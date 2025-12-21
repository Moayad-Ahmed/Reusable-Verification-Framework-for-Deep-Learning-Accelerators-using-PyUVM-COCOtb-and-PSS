from pyuvm import *
from pooling_utils import PoolingBFM

class My_Driver(uvm_driver):
    def start_of_simulation_phase(self):
        self.bfm = PoolingBFM()

    async def launch_tb(self):
        await self.bfm.reset()
        self.bfm.start_bfm()
    
    async def run_phase(self):
        await self.launch_tb()

        while True:
            seq_item = await self.seq_item_port.get_next_item()
            await self.bfm.send_config(seq_item)
            self.seq_item_port.item_done()