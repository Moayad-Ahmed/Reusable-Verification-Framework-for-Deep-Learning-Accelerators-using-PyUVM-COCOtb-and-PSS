from pyuvm import *
from cnn_utils import CNN_BFM

class My_Monitor(uvm_monitor):

    def build_phase(self):
        self.bfm = CNN_BFM()
        self.mon_ap = uvm_analysis_port("mon_ap", self)
   
    async def run_phase(self):
        while True:
            seq_item = await self.bfm.get_result()
            self.mon_ap.write(seq_item)