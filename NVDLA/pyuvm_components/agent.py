from pyuvm import *
from pyuvm_components.monitor import NVDLA_Monitor
from pyuvm_components.driver import NVDLA_Driver


class NVDLA_Agent(uvm_agent):
    def build_phase(self):
        self.driver = NVDLA_Driver("NVDLA_Driver", self)
        self.monitor = NVDLA_Monitor("NVDLA_Monitor", self)
        self.sqr = uvm_sequencer("sqr", self)
        ConfigDB().set(None, "*", "SEQR", self.sqr)
        self.agt_ap = uvm_analysis_port("agt_ap", self)

    def connect_phase(self):
        self.driver.seq_item_port.connect(self.sqr.seq_item_export)
        self.monitor.mon_ap.connect(self.agt_ap)