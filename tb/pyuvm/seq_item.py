from pyuvm import *

class GenericLayerTransaction(uvm_sequence_item):
    def __init__(self, name, layer_strategy):
        super().__init__(name)
        self.strategy = layer_strategy
        self.config = {}
        self.input_data = None
        self.expected_output = None
        self.actual_output = None
        self.layer_type = layer_strategy.get_layer_type()

    def __str__(self):
        return f"Generic Transaction: [{self.layer_type}]: config={self.config}"