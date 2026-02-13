from pyuvm import *
from utils.nvdla_utils import NvdlaBFM


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

            data = result.actual_output_data

            self.logger.info("PDP output @ 0x%08x", result.output_base_addr)
            self.logger.info("Output data bytes: %s", [f"0x{b:02x}" for b in data])

            # Compare against the golden model data
            if data == result.expected_output_data:
                self.logger.info("Expected data=%s", [f"0x{b:02x}" for b in result.expected_output_data])
                self.logger.info("Actual data match expected values. Test PASSED.")
                self.passed += 1
            else:
                self.logger.error("Expected data=%s", [f"0x{b:02x}" for b in result.expected_output_data])
                self.logger.error("Actual data=%s", [f"0x{b:02x}" for b in data])
                self.logger.error("Actual data do NOT match expected values. Test FAILED.")
                self.failed += 1

            # Signal the sequence that this iteration's check is complete
            await self.bfm.iteration_done_queue.put(True)

    def check_phase(self):
        self.logger.info(f"Check phase completed: {self.passed} correct cases, {self.failed} failed cases")

        assert self.failed == 0, f"Test failed with {self.failed} failed cases"
        assert self.passed > 0, f"Test failed with no passing cases"