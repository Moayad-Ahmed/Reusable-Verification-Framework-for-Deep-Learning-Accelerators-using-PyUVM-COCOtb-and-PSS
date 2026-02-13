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
            layer_name, output_base_addr, output_length, expected_output_data, expected_crc = recieved_seq_item

            # Wait for the NVDLA interrupt for inference completion
            await self.bfm.wait_for_interrupt()

            # Read the output surface and compute CRC
            actual_output_data = await self.bfm.read_from_dram(output_base_addr, output_length)            
            actual_crc = self.bfm.calc_crc32(actual_output_data)

            # Build a result transaction and send it to the scoreboard
            if layer_name == "pooling":
                result = PdpTransaction("pdp_result")
            else:
                # Dummy line to avoid error -> replace with the new transaction class added in the future
                result = PdpTransaction("generic_result")
            
            result.layer_name = layer_name
            result.output_base_addr = output_base_addr
            result.output_length = output_length
            result.actual_crc = actual_crc
            result.actual_output_data = actual_output_data
            result.expected_crc = expected_crc
            result.expected_output_data = expected_output_data

            self.mon_ap.write(result)