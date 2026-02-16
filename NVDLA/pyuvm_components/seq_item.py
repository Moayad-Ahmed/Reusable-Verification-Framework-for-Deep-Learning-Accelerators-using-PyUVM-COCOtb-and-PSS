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
        self.output_num_pixels = 0      # number of spatial output pixels
        self.output_pixel_bytes = 8     # total bytes per pixel in DRAM (atom-aligned)
        self.output_data_bytes_per_pixel = 1  # real data bytes per pixel (channels × bpe)
        # ----- Actual and expected output data (filled by monitor) -----
        self.actual_output_data = None
        self.expected_output_data = None

    def __str__(self):
        return f"Generic Transaction: [{self.layer_type}]: config={self.layer_configs}"


class ConvTransaction(uvm_sequence_item):
    """
    Sequence item for NVDLA Convolution transactions.

    Extends the common interface with weight-data fields so that both
    input features AND weights can be loaded into DRAM before the
    convolution pipeline is enabled.

    The driver / monitor / scoreboard use the same fields as PdpTransaction:
        input_file, input_base_addr, reg_configs, output_base_addr,
        output_num_pixels, output_pixel_bytes, output_data_bytes_per_pixel,
        actual_output_data, expected_output_data
    """

    def __init__(self, name, layer_strategy):
        super().__init__(name)
        self.strategy = layer_strategy
        self.layer_type = layer_strategy.get_layer_type()
        self.layer_configs = {}
        # ----- Input feature data -----
        self.input_file = None
        self.input_base_addr = 0
        # ----- Weight data -----
        self.weight_file = None         # path to hex weight-data file
        self.weight_base_addr = 0       # DRAM address for weights
        # ----- CSB register writes (may include 'poll' tuples) -----
        self.reg_configs = []           # [(addr, data), ...] or (addr, exp, 'poll')
        # ----- Output info -----
        self.output_base_addr = 0
        self.output_num_pixels = 0
        self.output_pixel_bytes = 8
        self.output_data_bytes_per_pixel = 1
        # ----- Results -----
        self.actual_output_data = None
        self.expected_output_data = None

    def __str__(self):
        return f"ConvTransaction: [{self.layer_type}]: config={self.layer_configs}"