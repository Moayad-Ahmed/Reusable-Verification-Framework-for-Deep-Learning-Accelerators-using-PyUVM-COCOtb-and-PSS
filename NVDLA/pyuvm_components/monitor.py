from pyuvm import *
from utils.nvdla_utils import NvdlaBFM
from pyuvm_components.seq_item import PdpTransaction


class NVDLA_Monitor(uvm_monitor):
    def build_phase(self):
        self.bfm = NvdlaBFM()
        self.mon_ap = uvm_analysis_port("mon_ap", self)

    async def run_phase(self):
        while True:
            # Wait until the driver tells us where the output address and length are
            recieved_seq_item = await self.bfm.output_config_queue.get()
            

            # Wait for the NVDLA interrupt for inference completion
            await self.bfm.wait_for_interrupt()

            # Read the output surface and compute CRC
            actual_output_data = await self.bfm.read_from_dram(
                recieved_seq_item.output_base_addr,
                recieved_seq_item.output_num_pixels,
                pixel_bytes=recieved_seq_item.output_pixel_bytes,
                data_bytes_per_pixel=recieved_seq_item.output_data_bytes_per_pixel
            )

            recieved_seq_item.actual_output_data = actual_output_data

            self.mon_ap.write(recieved_seq_item)