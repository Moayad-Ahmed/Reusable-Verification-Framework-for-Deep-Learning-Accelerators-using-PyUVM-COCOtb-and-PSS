from pyuvm import *

class ConvolutionTransaction(uvm_sequence_item):
    def __init__(self, name, strategy):
        super().__init__(name)
        self.strategy = strategy
        self.config = {}
        self.input_data = None
        self.kernel_weights = None
        self.expected_output = None
        self.actual_output = None
        self.txn_id = None # Transaction ID for tracking
        self.layer_type = strategy.get_layer_type()

    def __str__(self):
        return f"Generic Transaction: [{self.layer_type}]: config={self.config}"
