from pyuvm import *
from utils.nvdla_utils import NvdlaBFM


class NVDLA_Driver(uvm_driver):
    def build_phase(self):
        self.bfm = NvdlaBFM()

    async def run_phase(self):
        await self.bfm.reset()

        while True:
            seq_item = await self.seq_item_port.get_next_item()

            # Load input data into DRAM
            try:
                await self.bfm.write_in_dram(seq_item.input_file, seq_item.input_base_addr)
            except Exception as e:
                uvm_fatal("DRIVER", f" Error loading input data: {e}")


            # Tell the monitor where to look for results and the expected output data and CRC
            await self.bfm.output_config_queue.put(
                (seq_item.layer_name, seq_item.output_base_addr, seq_item.output_length, 
                 seq_item.expected_output_data, seq_item.expected_crc)
            )

            # Write all CSB registers
            for addr, data in seq_item.register_writes.items():
                await self.bfm.reg_write(addr, data)

            self.seq_item_port.item_done()