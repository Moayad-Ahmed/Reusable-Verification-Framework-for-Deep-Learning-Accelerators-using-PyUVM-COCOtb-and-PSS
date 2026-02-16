from pyuvm import *
from utils.nvdla_utils import NvdlaBFM


class NVDLA_Driver(uvm_driver):
    def build_phase(self):
        self.bfm = NvdlaBFM()

    async def run_phase(self):
        await self.bfm.reset()

        while True:
            seq_item = await self.seq_item_port.get_next_item()
            await self.bfm.reset()

            # Load input data into DRAM
            try:
                await self.bfm.write_in_dram(seq_item.input_file, seq_item.input_base_addr)
            except Exception as e:
                uvm_fatal("DRIVER", f" Error loading input data: {e}")

            # Load weight data into DRAM (convolution only)
            if hasattr(seq_item, 'weight_file') and seq_item.weight_file:
                try:
                    await self.bfm.write_in_dram(seq_item.weight_file,
                                                  seq_item.weight_base_addr)
                except Exception as e:
                    uvm_fatal("DRIVER", f" Error loading weight data: {e}")

            # Tell the monitor where to look for results and the expected output data and CRC
            await self.bfm.output_config_queue.put(seq_item)

            # Write all CSB registers (support poll operations)
            for cfg in seq_item.reg_configs:
                if len(cfg) == 3 and cfg[2] == 'poll':
                    await self.bfm.poll_reg(cfg[0], cfg[1])
                else:
                    await self.bfm.reg_write(cfg[0], cfg[1])

            self.seq_item_port.item_done()