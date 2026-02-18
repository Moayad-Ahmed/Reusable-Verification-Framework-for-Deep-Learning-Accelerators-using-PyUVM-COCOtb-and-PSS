from pyuvm import *
from utils.nvdla_utils import NvdlaBFM


class CsbMonitor(uvm_monitor):

    def build_phase(self):
        self.bfm    = NvdlaBFM()
        self.mon_ap = uvm_analysis_port("csb_mon_ap", self)

    async def run_phase(self):
        while True:
            # ---- Wait for CsbDriver to hand over the transaction ----
            seq_item = await self.bfm.output_config_queue.get()

            # ---- Wait for NVDLA interrupt (inference complete) ----
            await self.bfm.wait_for_interrupt()

            # ---- Read output surface from DRAM ----
            actual_output_data = await self.bfm.read_from_dram(
                seq_item.output_base_addr,
                seq_item.output_num_pixels,
                pixel_bytes=seq_item.output_pixel_bytes,
                data_bytes_per_pixel=seq_item.output_data_bytes_per_pixel,
            )

            # ---- Attach actual data and forward to scoreboard ----
            seq_item.actual_output_data = actual_output_data
            self.mon_ap.write(seq_item)
