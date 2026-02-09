from pyuvm import *
from pyuvm_components.monitor import My_Monitor
from pyuvm_components.driver import My_Driver

class My_Agent(uvm_agent):
    def build_phase(self):
        self.My_Monitor = My_Monitor("My_Monitor", self)
        self.My_Driver = My_Driver("My_Driver", self)
        self.sqr = uvm_sequencer("sqr", self)
        ConfigDB().set(None, "*", "SEQR", self.sqr)
        self.agt_ap = uvm_analysis_port("agt_ap", self)
   
    def connect_phase(self):
        self.My_Driver.seq_item_port.connect(self.sqr.seq_item_export)
        self.My_Monitor.mon_ap.connect(self.agt_ap)