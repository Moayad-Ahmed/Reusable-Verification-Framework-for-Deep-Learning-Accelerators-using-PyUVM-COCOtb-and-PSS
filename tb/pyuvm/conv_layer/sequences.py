from pyuvm import *
from seq_item import ConvolutionTransaction
from convolution_utils import ConvolutionBFM
from convolution_strategy import ConvolutionStrategy
import yaml

class ConvolutionSequence(uvm_sequence):
    """Basic convolution sequence"""
    def __init__(self, name, config_file):
        super().__init__(name)
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
    
    async def body(self):
        strategy = ConvolutionStrategy()
        
        for test_suite in self.config['test_suite']:
            uvm_root().logger.info(f"Running test: {test_suite['name']}")
            
            for layer_spec in test_suite['layers']:
                # Number of test iterations
                num_tests = layer_spec.get('num_tests', 1)
                
                for i in range(num_tests):
                    # Create transaction
                    seq_item = ConvolutionTransaction(
                        f"conv_test_{i}",
                        strategy
                    )

                    # Get configuration
                    seq_item.config = layer_spec['config']

                    # Generate input data
                    seq_item.input_data = strategy.generate_input_data(seq_item.config)

                    # Generate kernel weights
                    seq_item.kernel_weights = strategy.generate_kernel_weights(seq_item.config)

                    # Compute expected output using the SAME kernel weights
                    seq_item.expected_output = strategy.compute_golden(
                        seq_item.input_data, 
                        seq_item.config, 
                        seq_item.kernel_weights
                    )

                    uvm_root().logger.info(
                        f"Test {i}: "
                        f"Input shape: {seq_item.input_data.shape}, "
                        f"Kernel shape: {seq_item.kernel_weights.shape}, "
                        f"Expected output shape: {seq_item.expected_output.shape}"
                    )

                    await self.start_item(seq_item)
                    await self.finish_item(seq_item)