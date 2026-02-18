from pyuvm import *


class BaseTransaction(uvm_sequence_item):
    """
    Base class for all NVDLA transactions.
    Holds the layer strategy reference and layer type string.
    """

    def __init__(self, name, layer_strategy):
        super().__init__(name)
        self.strategy      = layer_strategy
        self.layer_type    = layer_strategy.get_layer_type()
        self.layer_configs = {}


# ══════════════════════════════════════════════════════════════════════
#  DATA TRANSACTION  —  used by DataAgent / DataDriver
# ══════════════════════════════════════════════════════════════════════

class DataTransaction(BaseTransaction):

    def __init__(self, name, layer_strategy):
        super().__init__(name, layer_strategy)
        self.input_file       = None
        self.input_base_addr  = 0
        self.weight_file      = None   # None for pooling transactions
        self.weight_base_addr = 0      # 0 for pooling transactions

    def __str__(self):
        return (
            f"DataTransaction [{self.layer_type}]: "
            f"input={self.input_file}@0x{self.input_base_addr:x}  "
            f"weight={self.weight_file}@0x{self.weight_base_addr:x}"
        )


# ══════════════════════════════════════════════════════════════════════
#  CSB TRANSACTION  —  used by CsbAgent / CsbDriver / CsbMonitor
# ══════════════════════════════════════════════════════════════════════

class CsbTransaction(BaseTransaction):

    def __init__(self, name, layer_strategy):
        super().__init__(name, layer_strategy)
        self.reg_configs                 = []
        self.output_base_addr            = 0
        self.output_num_pixels           = 0
        self.output_pixel_bytes          = 8
        self.output_data_bytes_per_pixel = 1
        self.expected_output_data        = None
        self.actual_output_data          = None

    def __str__(self):
        return (
            f"CsbTransaction [{self.layer_type}]: "
            f"out@0x{self.output_base_addr:x}  "
            f"pixels={self.output_num_pixels}"
        )


# ══════════════════════════════════════════════════════════════════════
#  LEGACY ALIASES  —  kept so existing test YAML / imports still work
# ══════════════════════════════════════════════════════════════════════

class PdpTransaction(CsbTransaction):
    """Legacy alias for CsbTransaction used in pooling tests."""
    pass


class ConvTransaction(CsbTransaction):
    """Legacy alias for CsbTransaction used in convolution tests."""
    pass
