from pyuvm import *
from utils.nvdla_utils import NvdlaBFM


class DataMonitor(uvm_monitor):

    def build_phase(self):
        self.bfm    = NvdlaBFM()
        self.mon_ap = uvm_analysis_port("data_mon_ap", self)

    async def run_phase(self):
        while True:
            # Wait for DataDriver to signal a completed DRAM load
            seq_item = await self.bfm.data_observed_queue.get()
            # Broadcast the data-load event to any subscribers
            self.mon_ap.write(seq_item)
from pyuvm import *


class DataMonitor(uvm_monitor):
    """Monitor that tracks data loads into DRAM."""

    def build_phase(self):
        self.mon_ap = uvm_analysis_port("mon_ap", self)

    async def run_phase(self):
        """Monitor passive - just tracks data loads."""
        self.logger.info("DataMonitor running")
        # Can be extended to capture and broadcast data transactions
