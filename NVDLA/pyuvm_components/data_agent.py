from pyuvm import *
from pyuvm_components.data_driver import DataDriver
from pyuvm_components.data_monitor import DataMonitor


class DataAgent(uvm_agent):
    """
    Agent responsible for loading input and weights into DRAM.

    """

    def build_phase(self):
        self.driver  = DataDriver("DataDriver", self)
        self.monitor = DataMonitor("DataMonitor", self)
        self.sqr     = uvm_sequencer("data_sqr", self)
        ConfigDB().set(None, "*", "DATA_SEQR", self.sqr)
        self.agt_ap  = uvm_analysis_port("data_agt_ap", self)

    def connect_phase(self):
        self.driver.seq_item_port.connect(self.sqr.seq_item_export)
        self.monitor.mon_ap.connect(self.agt_ap)
