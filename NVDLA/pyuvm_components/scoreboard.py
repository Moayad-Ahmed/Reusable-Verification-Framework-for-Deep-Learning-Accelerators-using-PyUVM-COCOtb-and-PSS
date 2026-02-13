from pyuvm import *


class NVDLA_Scoreboard(uvm_scoreboard):
    def build_phase(self):
        self.score_fifo = uvm_tlm_analysis_fifo("score_fifo", self)
        self.score_export = self.score_fifo.analysis_export
        self.score_get_port = uvm_get_port("score_get_port", self)

    def connect_phase(self):
        self.score_get_port.connect(self.score_fifo.get_export)

    def check_phase(self):
        passed = 0
        failed = 0

        while self.score_get_port.can_get():
            success, result = self.score_get_port.try_get()
            if not success:
                break

            crc = result.actual_crc
            data = result.actual_output_data

            self.logger.info("PDP output @ 0x%08x,  CRC=0x%08x", result.output_base_addr, crc)
            self.logger.info("Output data bytes: %s", [f"0x{b:02x}" for b in data])

            # Compare against the golden model CRC and data
            if crc == result.expected_crc and data == result.expected_output_data:
                self.logger.info("Expected CRC=0x%08x, data=%s", result.expected_crc, [f"0x{b:02x}" for b in result.expected_output_data])
                self.logger.info("Actual CRC and data match expected values. Test PASSED.")
                passed += 1
            else:
                self.logger.error("Expected CRC=0x%08x, data=%s", result.expected_crc, [f"0x{b:02x}" for b in result.expected_output_data])
                self.logger.error("Actual CRC=0x%08x, data=%s", crc, [f"0x{b:02x}" for b in data])
                self.logger.error("Actual CRC and data do NOT match expected values. Test FAILED.")
                failed += 1

        self.logger.info(f"Check phase completed: {passed} correct cases, {failed} failed cases")

        assert not failed != 0, f"Test failed with {failed} failed cases"
        assert not passed == 0, f"Test failed with no passing cases"