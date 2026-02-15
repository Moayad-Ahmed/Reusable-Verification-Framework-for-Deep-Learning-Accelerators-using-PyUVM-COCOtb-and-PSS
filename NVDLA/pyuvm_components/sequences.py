from pyuvm import *
from pyuvm_components.seq_item import PdpTransaction
import numpy as np
import os
import yaml
from strategy.regs_configs import RegistrationConfigs
from strategy.Layer_Factory import LayerFactory
from utils.nvdla_utils import NvdlaBFM


class PdpTestSequence(uvm_sequence):
    # NVDLA_MEMORY_ATOMIC_SIZE for nv_small configuration
    ATOM = 8
    # Bytes-per-element lookup by data format
    BPE_MAP = {'INT8': 1, 'INT16': 2, 'FP16': 2}
    ITERATIONS = 2

    def __init__(self, name, input_file=None, config_file=None):
        super().__init__(name)
        self.input_file = input_file
        self.config_file = config_file

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

        For multi-channel data, NVDLA packs all channels of a spatial position
        into one atom (when channels*bpe <= ATOM), then pads to atom boundary.
        Memory layout: for each (h, w): write all C channels, then pad.

        Args:
            input_bytes: Flattened list in channel-first order (C, H, W)
            config: Configuration dictionary (data_format, input_shape)
        """
        bpe, pixel_bytes = self.compute_pixel_layout(config)
        input_shape = config['input_shape']
        
        if len(input_shape) == 3:
            h, w, c = input_shape
        else:
            h, w = input_shape
            c = 1

        if self.input_file:
            with open(self.input_file, 'w') as f:
                # Data is in CHW order (channel-first), but memory needs HWC per-pixel layout
                # Reshape input_bytes from flat CHW to (C, H, W)
                if c == 1:
                    data_array = np.array(input_bytes).reshape((h, w))
                else:
                    data_array = np.array(input_bytes).reshape((c, h, w))
                
                # Write in HW order, with all channels per pixel
                for row in range(h):
                    for col in range(w):
                        # Write all channels for this spatial position
                        if c == 1:
                            pixel_data = [data_array[row, col]]
                        else:
                            pixel_data = [data_array[ch, row, col] for ch in range(c)]
                        
                        # Write each channel's bytes
                        for val in pixel_data:
                            value_bytes = int(val).to_bytes(
                                bpe, byteorder='little', signed=True
                            )
                            for b in value_bytes:
                                f.write(f"{b:02x}\n")
                        
                        # Pad the rest of the atom to pixel_bytes
                        bytes_written = c * bpe
                        for _ in range(pixel_bytes - bytes_written):
                            f.write("00\n")

    @classmethod
    def get_numpy_dtype(cls, config):
        data_format = config.get('data_format', 'INT8')
        if data_format in ('INT16'):
            return np.int16
        elif data_format in ('FP16'):
            return np.float16
        return np.int8

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
        reg_cfg = RegistrationConfigs()
        # Load YAML configuration
        config = self.load_yaml_config()
        bfm = NvdlaBFM()
        np_dtype = self.get_numpy_dtype(config)
        _, pixel_bytes = self.compute_pixel_layout(config)

        input_shape = config['input_shape']
        channels = input_shape[2] if len(input_shape) == 3 else 1
        num_spatial_pixels = input_shape[0] * input_shape[1]  # H × W
        input_total_bytes = num_spatial_pixels * pixel_bytes

        for i in range(self.ITERATIONS):
            seq_item = PdpTransaction("pdp_tx", strategy)

            # Generate input data
            input_data = strategy.generate_input_data(config)
            # Compute golden output adapted for NVDLA hardware
            expected_output = strategy.nvdla_pool_adjust(input_data, config)
            
            # Flatten to 1D byte array - convert from CHW (model format) to HWC (NVDLA memory layout)
            input_bytes = input_data.flatten().astype(np_dtype).tolist()
            
            # Convert expected output from CHW to HWC memory layout (pixel-by-pixel with all channels)
            input_shape = config['input_shape']
            is_multi_channel = len(input_shape) == 3 and input_shape[2] > 1
            
            if is_multi_channel and expected_output.ndim == 3:
                # Expected output is (C, H, W), need to convert to HWC memory layout
                c, h, w = expected_output.shape
                # Transpose from (C, H, W) to (H, W, C) then flatten
                expected_hwc = np.transpose(expected_output, (1, 2, 0))  # Now (H, W, C)
                expected_bytes = expected_hwc.flatten().astype(np_dtype).tolist()
            else:
                # Single channel: keep as-is
                expected_bytes = expected_output.flatten().astype(np_dtype).tolist()

            # Debug prints
            print(f"\n=== Iteration {i} ===")
            print(f"Expected output (raw): {expected_output}")
            print(f"Expected bytes:        {expected_bytes}")

            # Write input data to file
            self.write_input_data_to_file(input_bytes, config)

            # ----- Input DRAM data -----
            seq_item.input_file = self.input_file
            seq_item.input_base_addr = 0
            seq_item.input_byte_count = input_total_bytes

            # ----- Compute destination base address -----
            # Place output after input data, 256-byte aligned, minimum 0x100
            seq_item.output_base_addr  = max(0x100, ((input_total_bytes + 255) // 256) * 256)

            # ----- PDP + PDP-RDMA register writes -----
            seq_item.reg_configs = reg_cfg.pooling_configs(config, src_addr=seq_item.input_base_addr, dst_addr=seq_item.output_base_addr)

            # ----- Expected output info (from golden model) -----
            bpe, _ = self.compute_pixel_layout(config)
            
            # Handle both 2D (H, W) and 3D (C, H, W) output shapes
            if expected_output.ndim == 2:
                out_h, out_w = expected_output.shape
            else:
                _, out_h, out_w = expected_output.shape  # (C, H, W)
            
            seq_item.output_num_pixels = out_h * out_w
            seq_item.output_pixel_bytes = pixel_bytes
            seq_item.output_data_bytes_per_pixel = channels * bpe
            seq_item.expected_output_data = expected_bytes

            await self.start_item(seq_item)
            await self.finish_item(seq_item)

            # Wait for the scoreboard to finish checking this iteration

            await bfm.iteration_done_queue.get()
            print(f"Iteration {i} checked by scoreboard.")