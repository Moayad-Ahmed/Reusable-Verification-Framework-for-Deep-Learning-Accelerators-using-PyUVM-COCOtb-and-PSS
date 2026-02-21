from pyuvm import *
from utils.nvdla_utils import NvdlaBFM


class DataMonitor(uvm_monitor):
    """Monitor that tracks data loads into DRAM."""

    def build_phase(self):
        self.mon_ap = uvm_analysis_port("mon_ap", self)

    async def run_phase(self):
        """Monitor passive - just tracks data loads."""
        self.logger.info("DataMonitor running")
        # Can be extended to capture and broadcast data transactions
