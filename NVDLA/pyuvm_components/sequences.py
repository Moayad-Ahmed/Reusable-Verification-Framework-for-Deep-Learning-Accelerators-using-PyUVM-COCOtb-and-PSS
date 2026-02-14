from pyuvm import *
from pyuvm_components.seq_item import PdpTransaction
import numpy as np
import os
import yaml
from strategy.regs_configs import RegistrationConfigs
from strategy.pooling_strategy import PoolingStrategy
from strategy.Layer_Factory import LayerFactory
from utils.nvdla_utils import NvdlaBFM


class PdpTestSequence(uvm_sequence):
    # NVDLA_MEMORY_ATOMIC_SIZE for nv_small configuration
    ATOM = 8
    # Bytes-per-element lookup by data format
    BPE_MAP = {'INT8': 1, 'INT16': 2, 'FP16': 2}

    def __init__(self, name, input_file=None, config_file=None):
        super().__init__(name)
        self.input_file = input_file
        self.config_file = config_file
        self.pooling_strategy = PoolingStrategy()

    @classmethod
    def compute_pixel_layout(cls, config):
        """
        Compute the atom-aligned memory layout for a single spatial pixel.

        Returns:
            tuple: (bpe, pixel_bytes)
                - bpe:         bytes per element (1 for INT8, 2 for INT16/FP16)
                - pixel_bytes: total bytes per pixel, padded to atom boundary
        """
        data_format = config.get('data_format', 'INT8')
        bpe = cls.BPE_MAP.get(data_format, 1)
        channels = config.get('channels', 1)
        atoms_per_pixel = max(1, (channels * bpe + cls.ATOM - 1) // cls.ATOM)
        pixel_bytes = atoms_per_pixel * cls.ATOM
        return bpe, pixel_bytes

    def write_input_data_to_file(self, input_bytes, config):
        """
        Write input data to .dat file in NVDLA atom-aligned format.

        Each spatial pixel is padded to atom boundaries so that
        write_in_dram() can map every ATOM lines to one 64-bit AXI write.

        Args:
            input_bytes: List of signed integer values (row-major)
            config: Configuration dictionary (data_format, input_shape)
        """
        bpe, pixel_bytes = self.compute_pixel_layout(config)

        if self.input_file:
            with open(self.input_file, 'w') as f:
                for byte_val in input_bytes:
                    # Write the actual data byte(s)
                    value_bytes = byte_val.to_bytes(
                        bpe, byteorder='little', signed=True
                    )
                    for b in value_bytes:
                        f.write(f"{b:02x}\n")
                    # Pad the rest of the atom to pixel_bytes
                    for _ in range(pixel_bytes - bpe):
                        f.write("00\n")

    def load_yaml_config(self):
        """Load and parse the YAML configuration file"""
        if not self.config_file or not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Extract the first test case and first layer config
        test_suite = yaml_data.get('test_suite', [])
        if not test_suite:
            raise ValueError("No test suite found in YAML config")
        
        first_test = test_suite[0]
        layers = first_test.get('layers', [])
        if not layers:
            raise ValueError("No layers found in test configuration")
        
        layer_config = layers[0].get('config', {})
        return layer_config


    async def body(self):
        strategy = LayerFactory.create_strategy('pooling')
        # Load YAML configuration
        config = self.load_yaml_config()
        bfm = NvdlaBFM()

        for i in range(2):
            seq_item = PdpTransaction("pdp_tx", strategy)

            # Generate input data
            input_data = strategy.generate_input_data(config)
            # Compute generic golden output, then adapt for NVDLA hardware
            generic_output = strategy.compute_golden(input_data, config)
            expected_output = strategy.nvdla_avg_adjust(input_data, config)
            # Flatten to 1D byte array
            input_bytes = input_data.flatten().astype(np.int8).tolist()
            expected_bytes = expected_output.flatten().astype(np.int8).tolist()

            # Debug prints
            print(f"\n=== Iteration {i} ===")
            print(f"Expected output (raw): {expected_output}")
            print(f"Expected bytes:        {expected_bytes}")

            # Write input data to file
            self.write_input_data_to_file(input_bytes, config)

            # ----- Atom-aligned layout -----
            _, pixel_bytes = self.compute_pixel_layout(config)
            input_total_bytes = len(input_bytes) * pixel_bytes

            # ----- Input DRAM data -----
            seq_item.input_file = self.input_file
            seq_item.input_base_addr = 0
            seq_item.input_byte_count = input_total_bytes

            # ----- Compute destination base address -----
            # Place output after input data, 256-byte aligned, minimum 0x100
            seq_item.output_base_addr  = max(0x100, ((input_total_bytes + 255) // 256) * 256)

            # ----- PDP + PDP-RDMA register writes -----
            seq_item.reg_configs = RegistrationConfigs().pooling_configs(config, src_addr=seq_item.input_base_addr, dst_addr=seq_item.output_base_addr)

            # ----- Expected output info (from golden model) -----
            bpe, pixel_bytes = self.compute_pixel_layout(config)
            channels = config.get('channels', 1)
            out_h, out_w = expected_output.shape
            seq_item.output_num_pixels = out_h * out_w
            seq_item.output_pixel_bytes = pixel_bytes
            seq_item.output_data_bytes_per_pixel = channels * bpe
            seq_item.expected_output_data = expected_bytes

            await self.start_item(seq_item)
            await self.finish_item(seq_item)

            # Wait for the scoreboard to finish checking this iteration

            await bfm.iteration_done_queue.get()
            print(f"Iteration {i} checked by scoreboard.")


            