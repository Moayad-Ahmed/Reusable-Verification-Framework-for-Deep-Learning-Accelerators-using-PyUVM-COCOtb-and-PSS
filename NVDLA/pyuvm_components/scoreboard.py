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
        self.score_fifo      = uvm_tlm_analysis_fifo("score_fifo", self)
        self.score_export    = self.score_fifo.analysis_export
        self.score_get_port  = uvm_get_port("score_get_port", self)
        self.bfm             = NvdlaBFM()
        self.passed          = 0
        self.failed          = 0

    def connect_phase(self):
        self.score_get_port.connect(self.score_fifo.get_export)

    async def run_phase(self):
        while True:
            result = await self.score_fifo.get()

            actual_data   = [_u8(b) for b in result.actual_output_data]
            expected_data = [_u8(b) for b in result.expected_output_data]

            self.logger.info("Output @ 0x%08x", result.output_base_addr)
            self.logger.info("Actual   (hex):    %s", _fmt_hex(actual_data))
            self.logger.info("Actual   (signed): %s", [_s8(b) for b in actual_data])

            if actual_data == expected_data:
                self.logger.info("Expected (hex): %s", _fmt_hex(expected_data))
                self.logger.info("PASSED — actual data matches expected.")
                self.passed += 1
            else:
                self.logger.error("Expected (hex): %s", _fmt_hex(expected_data))
                self.logger.error("Actual   (hex): %s", _fmt_hex(actual_data))
                self.logger.error("FAILED — actual data does NOT match expected.")
                self.failed += 1

            # Signal the virtual sequence that this iteration's check is complete
            await self.bfm.iteration_done_queue.put(True)

    def check_phase(self):
        self.logger.info(
            "Check phase: %d passed, %d failed", self.passed, self.failed
        )
        assert self.failed == 0, f"Test FAILED with {self.failed} failed case(s)"
        assert self.passed  > 0, f"Test FAILED — no passing cases recorded"
