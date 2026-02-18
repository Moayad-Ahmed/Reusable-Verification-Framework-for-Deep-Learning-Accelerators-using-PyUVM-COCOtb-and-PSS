from pyuvm import *
from utils.nvdla_utils import NvdlaBFM


class DataDriver(uvm_driver):
    
    def build_phase(self):
        self.bfm = NvdlaBFM()

    async def run_phase(self):
        await self.bfm.reset()

        while True:
            seq_item = await self.seq_item_port.get_next_item()
            await self.bfm.reset()

            # ---- Load input activations into DRAM ----
            try:
                await self.bfm.write_in_dram(
                    seq_item.input_file,
                    seq_item.input_base_addr,
                )
            except Exception as e:
                uvm_fatal("DATA_DRIVER", f"Error loading input data: {e}")

            # ---- Load weights into DRAM (convolution only) ----
            # weight_file is None for pooling transactions so skip safely
            if seq_item.weight_file is not None:
                try:
                    await self.bfm.write_in_dram(
                        seq_item.weight_file,
                        seq_item.weight_base_addr,
                    )
                except Exception as e:
                    uvm_fatal("DATA_DRIVER", f"Error loading weight data: {e}")

            # ---- Signal completion to both consumers ----
            # CsbDriver waits on data_ready_queue before writing registers
            await self.bfm.data_ready_queue.put(seq_item)
            # DataMonitor waits on data_observed_queue for passive observation
            await self.bfm.data_observed_queue.put(seq_item)

            self.seq_item_port.item_done()
