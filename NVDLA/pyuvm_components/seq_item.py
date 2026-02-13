from pyuvm import *

class PdpTransaction(uvm_sequence_item):
    """
    Sequence item for NVDLA PDP transactions [responsible for pooling layer]

    Contains all information about a single PDP operation, including:
      - Input Data in DRAM                          (file path, base address, byte count)
      - A list of register configurations [CSB]     {(addr, data), ...}
      - Actual output address info                  (base address, byte length)
      - Actual output data info                     (actual CRC, actual output bytes)
      - Expected output data info                   (expected CRC, expected output bytes)
    """

    def __init__(self, name, layer_strategy):
        super().__init__(name)
        self.strategy = layer_strategy
        self.layer_type = layer_strategy.get_layer_type()
        self.layer_configs = {}
        # ----- Input Data (set by driver) -----
        self.input_file = None          # path to hex input-data file
        self.input_base_addr = 0        # DRAM address where input is loaded
        self.weights = None
        self.biases = None
        # ----- Configurations (set by driver) -----
        self.reg_configs = []       # [(addr, data), ...]
        # ----- Actual Output address info (set by driver) -----
        self.output_base_addr = 0       # DRAM address of result surface
        self.output_length = 0          # bytes to read / CRC
        # ----- Actual and expected output data (filled by monitor) -----
        self.actual_output_data = None
        self.expected_output_data = None

    def __str__(self):
        return f"Generic Transaction: [{self.layer_type}]: config={self.layer_configs}"