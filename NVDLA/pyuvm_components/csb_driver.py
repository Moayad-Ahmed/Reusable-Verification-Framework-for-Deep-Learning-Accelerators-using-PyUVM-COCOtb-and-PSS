from pyuvm import *
from utils.nvdla_utils import NvdlaBFM


class CsbDriver(uvm_driver):

    def build_phase(self):
        self.bfm = NvdlaBFM()

    async def run_phase(self):
        while True:
            seq_item = await self.seq_item_port.get_next_item()

            # ---- Wait until DataDriver has finished loading DRAM ----
            # This is the key ordering guarantee: hardware registers are never
            # written before the input / weight data is present in DRAM.
            await self.bfm.data_ready_queue.get()

            # ---- Forward transaction to CsbMonitor ----
            # Monitor needs output_base_addr, pixel layout, and expected_output_data
            # so it knows where to read results and what to compare against.
            await self.bfm.output_config_queue.put(seq_item)

            # ---- Write all CSB registers in order ----
            for cfg in seq_item.reg_configs:
                if len(cfg) == 3 and cfg[2] == 'poll':
                    # Poll operation: read register until value matches
                    await self.bfm.poll_reg(cfg[0], cfg[1])
                else:
                    # Normal write: single non-posted CSB transaction
                    await self.bfm.reg_write(cfg[0], cfg[1])

            self.seq_item_port.item_done()
