from pyuvm import *
import numpy as np

class My_Scoreboard(uvm_scoreboard):
    def build_phase(self):
        self.score_fifo = uvm_tlm_analysis_fifo("score_fifo", self)
        self.score_export = self.score_fifo.analysis_export
        self.score_get_port = uvm_get_port("score_get_port", self)

    def connect_phase(self):
        self.score_get_port.connect(self.score_fifo.get_export)

    def check_phase(self):
        correct_cases = 0
        failed_cases = 0
        while self.score_get_port.can_get():
            success, seq_item = self.score_get_port.try_get()
            if not success:
                break

            if not np.array_equal(seq_item.expected_output, seq_item.actual_output):
                self.logger.error(f"Output Mismatch for layer: {seq_item.layer_type}")
                self.logger.error(f"  Expected:\n{seq_item.expected_output}")
                self.logger.error(f"  Actual:\n{seq_item.actual_output}")                
                failed_cases += 1
            else:
                self.logger.info(f"Output Match for layer: {seq_item.layer_type}")
                self.logger.info(f"  Expected:\n{seq_item.expected_output}")
                self.logger.info(f"  Actual:\n{seq_item.actual_output}")
                correct_cases += 1

        assert failed_cases == 0, f"Test failed with {failed_cases} failed cases"

        self.logger.info(f"Check phase completed: {correct_cases} correct cases, {failed_cases} failed cases")