from pyuvm import *
from seq_item import GenericLayerTransaction
import yaml
import numpy as np
from Layer_Factory import LayerFactory
from math import floor
from cnn_utils import CNN_BFM


class ConfigDrivenSequence(uvm_sequence):
    def __init__(self, name, config_file):
        super().__init__(name)
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
    
    async def body(self):
        for test_suite in self.config['test_suite']:
            uvm_root().logger.info(f"Running test: {test_suite['name']}")
            
            for layer_spec in test_suite['layers']:
                # Create strategy from factory
                strategy = LayerFactory.create_strategy(layer_spec['type'])
                
                for i in range(10):
                    # Create transaction
                    seq_item = GenericLayerTransaction(
                        f"{layer_spec['type']}_test_{i}",
                        strategy
                    )

                    # Get configuration
                    seq_item.config = layer_spec['config']

                    # Generate input data
                    seq_item.input_data = strategy.generate_input_data(seq_item.config)

                    if seq_item.layer_type == 'convolution':
                        # Generate kernel weights for convolution
                        seq_item.kernel_weights = strategy.generate_kernel_weights(seq_item.config)

                        # Compute expected output using the SAME kernel weights
                        seq_item.expected_output = strategy.compute_golden(
                            seq_item.input_data, 
                            seq_item.config, 
                            seq_item.kernel_weights
                        )
                    elif seq_item.layer_type == 'fully_connected':
                        # Generate weights and biases for fully connected layer
                        seq_item.fc_weights = strategy.generate_input_weights(seq_item.config)
                        seq_item.fc_bias = strategy.generate_input_biases(seq_item.config)

                        # Compute expected output using the SAME weights and biases
                        seq_item.expected_output = strategy.compute_golden(
                            seq_item.input_data, 
                            seq_item.fc_weights, 
                            seq_item.fc_bias, 
                            seq_item.config
                        )
                    else:
                        seq_item.expected_output = strategy.compute_golden(
                             seq_item.input_data, seq_item.config)

                    await self.start_item(seq_item)
                    await self.finish_item(seq_item)


class ChainedLayerSequence(uvm_sequence):
    def __init__(self, name, config_file):
        super().__init__(name)
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
    
    async def body(self):
        for test_suite in self.config['test_suite']:
            uvm_root().logger.info(f"Running chained test: {test_suite['name']}")
            
            for i in range(10):
                previous_output = None
                
                for layer_idx, layer_spec in enumerate(test_suite['layers']):
                    # Create strategy from factory
                    strategy = LayerFactory.create_strategy(layer_spec['type'])

                    # Create transaction
                    seq_item = GenericLayerTransaction(
                        f"{layer_spec['type']}_chain_{i}_layer{layer_idx}",
                        strategy
                    )

                    # Get configuration
                    seq_item.config = layer_spec['config']

                    # First layer: generate input, others: use previous output
                    if layer_idx == 0:
                        seq_item.input_data = strategy.generate_input_data(seq_item.config)
                    elif layer_idx != 0 and seq_item.layer_type == 'fully_connected':
                        # For fully connected layers, we need to flatten the previous output
                        # Cast to int8 so the golden model uses signed interpretation matching the HW
                        seq_item.input_data = previous_output.flatten().astype(np.int8)
                    else:
                        seq_item.input_data = previous_output
                        #uvm_root().logger.info(f"Layer {layer_idx} using previous layer's output as input")
                        #uvm_root().logger.info(f"Input data for layer = {seq_item.input_data}")


                    if seq_item.layer_type == 'convolution':
                        # Generate kernel weights for convolution
                        seq_item.kernel_weights = strategy.generate_kernel_weights(seq_item.config)

                        # Compute expected output using the SAME kernel weights
                        seq_item.expected_output = strategy.compute_golden(
                            seq_item.input_data, 
                            seq_item.config, 
                            seq_item.kernel_weights
                        )
                    elif seq_item.layer_type == 'fully_connected':
                        # Generate weights and biases for fully connected layer
                        seq_item.fc_weights = strategy.generate_input_weights(seq_item.config)
                        seq_item.fc_bias = strategy.generate_input_biases(seq_item.config)

                        # Compute expected output using the SAME weights and biases
                        seq_item.expected_output = strategy.compute_golden(
                            seq_item.input_data, 
                            seq_item.fc_weights, 
                            seq_item.fc_bias, 
                            seq_item.config
                        )
                    else:
                        seq_item.expected_output = strategy.compute_golden(
                             seq_item.input_data, seq_item.config)

                    await self.start_item(seq_item)
                    await self.finish_item(seq_item)

                    # Use actual DUT output for next layer
                    #output_height = floor((seq_item.config['input_shape'][0] - seq_item.config['kernel_size']) / seq_item.config['stride']) + 1
                    #output_width = floor((seq_item.config['input_shape'][1] - seq_item.config['kernel_size']) / seq_item.config['stride'])  + 1

                    # Wait for the monitor to capture the actual DUT output
                    bfm = CNN_BFM()
                    actual_output = await bfm.wait_for_result(seq_item)

                    # actual_output is already a numpy array from the BFM
                    previous_output = actual_output
                    #uvm_root().logger.info(f"Previous output for layer = {previous_output}")
                    