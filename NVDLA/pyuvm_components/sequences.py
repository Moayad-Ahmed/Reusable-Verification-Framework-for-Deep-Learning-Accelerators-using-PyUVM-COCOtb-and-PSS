from pyuvm import *
from pyuvm_components.seq_item import PdpTransaction
import numpy as np
import os
import yaml
from strategy.regs_configs import RegistrationConfigs
from strategy.pooling_strategy import PoolingStrategy
from strategy.Layer_Factory import LayerFactory


class PdpTestSequence(uvm_sequence):
    def __init__(self, name, input_file=None, config_file=None):
        super().__init__(name)
        self.input_file = input_file
        self.config_file = config_file
        self.pooling_strategy = PoolingStrategy()

    def write_input_data_to_file(self, input_bytes, config):
        """
        Write input data to .dat file in hex format (64-bit aligned).
        
        This function converts input bytes to hex format based on the data format
        specified in the config, and writes them to a file with 64-bit alignment.
        
        Args:
            input_bytes: List of byte values to write
            config: Configuration dictionary containing data_format
        """
        # Determine bytes per element based on data format
        data_format = config.get('data_format', 'INT8')
        format_sizes = {
            'INT8': 1,   # 1 byte
            'INT16': 2,  # 2 bytes
            'FP16': 2,   # 2 bytes
        }
        bytes_per_element = format_sizes.get(data_format, 1)
        
        # Write generated data to .dat file in hex format (64-bit aligned)
        if self.input_file:
            with open(self.input_file, 'w') as f:
                for byte_val in input_bytes:
                    # Convert value to bytes (little-endian)
                    value_bytes = byte_val.to_bytes(bytes_per_element, byteorder='little', signed=False)
                    
                    # Write each byte of the value
                    for b in value_bytes:
                        f.write(f"{b:02x}\n")
                    
                    # Pad remaining bytes to reach 64 bits (8 bytes total)
                    padding_bytes = 8 - bytes_per_element
                    for _ in range(padding_bytes):
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
        seq_item = PdpTransaction("pdp_tx",strategy)

        # Load YAML configuration
        config = self.load_yaml_config()
        
        # Generate input data
        input_data = strategy.generate_input_data(config)
        # Compute golden output
        expected_output = strategy.compute_golden(input_data, config)
        # Flatten to 1D byte array
        input_bytes = input_data.flatten().astype(np.int8).tolist()
        expected_bytes = expected_output.flatten().astype(np.int8).tolist()

        # Write input data to file
        self.write_input_data_to_file(input_bytes, config)
        
        # ----- Input DRAM data -----
        seq_item.input_file = self.input_file
        seq_item.input_base_addr = 0
        # Each logical value is stored as 8 bytes (64 bits) in the .dat file
        seq_item.input_byte_count = len(input_bytes) * 8

        # ----- PDP + PDP-RDMA register writes -----
        # Organized by functional groups for better readability
        seq_item.reg_configs = RegistrationConfigs().pooling_configs()

        # ----- Expected output info (from golden model) -----
        seq_item.output_base_addr = 0x100     # DST_BASE_ADDR_LOW
        seq_item.output_length = len(expected_bytes)
        seq_item.expected_output_data = expected_bytes

        await self.start_item(seq_item)
        await self.finish_item(seq_item)