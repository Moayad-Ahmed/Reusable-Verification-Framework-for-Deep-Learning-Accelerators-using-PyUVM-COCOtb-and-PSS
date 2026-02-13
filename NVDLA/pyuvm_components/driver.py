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
            if seq_item.input_file:
                self.bfm.load_memory_from_file(
                    seq_item.input_file,
                    seq_item.input_base_addr,
                    seq_item.input_byte_count,
                )

            # Tell the monitor where to look for results and the expected output data and CRC
            await self.bfm.output_config_queue.put(
                (seq_item.layer_name, seq_item.output_base_addr, seq_item.output_length, 
                 seq_item.expected_output_data, seq_item.expected_crc)
            )

            # Write all CSB registers
            for addr, data in seq_item.register_writes:
                await self.bfm.csb_write(addr, data)

            self.seq_item_port.item_done()