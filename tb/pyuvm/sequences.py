from pyuvm import *
from seq_item import GenericLayerTransaction
import yaml
from Layer_Factory import LayerFactory
from tb.pyuvm.pooling_strategy import PoolingStrategy


class ConfigDrivenSequence(uvm_sequence):
    """Generate tests from YAML config"""
    
    def __init__(self, name, config_file):
        super().__init__(name)
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
    
    async def body(self):
        for test_suite in self.config['test_suite']:
            uvm_root().logger.info(f"Running test: {test_suite['name']}")
            
            for layer_spec in test_suite['layers']:
                # Create strategy from factory
                LayerFactory.register_strategy('pooling',PoolingStrategy)
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

                    # Compute expected output
                    seq_item.expected_output = strategy.compute_golden(
                         seq_item.input_data, seq_item.config)

                    await self.start_item(seq_item)
                    await self.finish_item(seq_item)
