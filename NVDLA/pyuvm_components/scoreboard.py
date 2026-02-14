from pyuvm import *
from utils.nvdla_utils import NvdlaBFM


def _u8(value):
    return int(value) & 0xFF


def _s8(value):
    value_u8 = _u8(value)
    return value_u8 - 256 if value_u8 >= 128 else value_u8


def _fmt_hex(byte_list):
    return [f"0x{_u8(b):02x}" for b in byte_list]


class NVDLA_Scoreboard(uvm_scoreboard):
    def build_phase(self):
        self.score_fifo = uvm_tlm_analysis_fifo("score_fifo", self)
        self.score_export = self.score_fifo.analysis_export
        self.score_get_port = uvm_get_port("score_get_port", self)
        self.bfm = NvdlaBFM()
        self.passed = 0
        self.failed = 0

    def connect_phase(self):
        self.score_get_port.connect(self.score_fifo.get_export)

    async def run_phase(self):
        while True:
            result = await self.score_fifo.get()

            actual_data = [_u8(b) for b in result.actual_output_data]
            expected_data = [_u8(b) for b in result.expected_output_data]

            self.logger.info("PDP output @ 0x%08x", result.output_base_addr)
            self.logger.info("Output data bytes (hex): %s", _fmt_hex(actual_data))
            self.logger.info("Output data bytes (signed): %s", [_s8(b) for b in actual_data])

            # Compare against the golden model data
            if actual_data == expected_data:
                self.logger.info("Expected data (hex)=%s", _fmt_hex(expected_data))
                self.logger.info("Actual data match expected values. Test PASSED.")
                self.passed += 1
            else:
                self.logger.error("Expected data (hex)=%s", _fmt_hex(expected_data))
                self.logger.error("Actual data (hex)=%s", _fmt_hex(actual_data))
                self.logger.error("Expected data (signed)=%s", [_s8(b) for b in expected_data])
                self.logger.error("Actual data (signed)=%s", [_s8(b) for b in actual_data])
                self.logger.error("Actual data do NOT match expected values. Test FAILED.")
                self.failed += 1

            # Signal the sequence that this iteration's check is complete
            await self.bfm.iteration_done_queue.put(True)

    def check_phase(self):
        self.logger.info(f"Check phase completed: {self.passed} correct cases, {self.failed} failed cases")

        assert self.failed == 0, f"Test failed with {self.failed} failed cases"
        assert self.passed > 0, f"Test failed with no passing cases"