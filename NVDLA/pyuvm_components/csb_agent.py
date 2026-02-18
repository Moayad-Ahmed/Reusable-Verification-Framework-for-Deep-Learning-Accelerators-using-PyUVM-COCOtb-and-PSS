from pyuvm import *
from pyuvm_components.csb_driver import CsbDriver
from pyuvm_components.csb_monitor import CsbMonitor


class CsbAgent(uvm_agent):
    """
    Agent responsible for Register configuration and output observation.
    """

    def build_phase(self):
        self.driver  = CsbDriver("CsbDriver", self)
        self.monitor = CsbMonitor("CsbMonitor", self)
        self.sqr     = uvm_sequencer("csb_sqr", self)
        ConfigDB().set(None, "*", "CSB_SEQR", self.sqr)
        self.agt_ap  = uvm_analysis_port("csb_agt_ap", self)

    def connect_phase(self):
        self.driver.seq_item_port.connect(self.sqr.seq_item_export)
        self.monitor.mon_ap.connect(self.agt_ap)
