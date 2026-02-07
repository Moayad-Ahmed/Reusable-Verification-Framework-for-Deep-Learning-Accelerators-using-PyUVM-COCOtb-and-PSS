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

            if seq_item.strategy.get_layer_type() == "pooling":
                if not np.array_equal(seq_item.expected_output, seq_item.actual_output):
                    self.logger.error(f"Output Mismatch: Got {seq_item.actual_output}, Expected {seq_item.expected_output}, Error in layer {seq_item.layer_type}")
                    failed_cases += 1
                else:
                    self.logger.info(f"Output matched: Got {seq_item.actual_output}, Expected {seq_item.expected_output}")
                    correct_cases += 1
            elif seq_item.strategy.get_layer_type() == "convolution":
                if not np.array_equal(seq_item.expected_output, seq_item.actual_output):
                    self.logger.error(f"Output Mismatch:")
                    self.logger.error(f"  Expected shape: {seq_item.expected_output.shape}")
                    self.logger.error(f"  Actual shape: {seq_item.actual_output.shape}")
                    self.logger.error(f"  Expected:\n{seq_item.expected_output}")
                    self.logger.error(f"  Actual:\n{seq_item.actual_output}")
                    self.logger.error(f"  Config: {seq_item.config}")
                    failed_cases += 1
                else:
                    self.logger.info(f"Output Matched!")
                    self.logger.info(f"  Shape: {seq_item.actual_output.shape}")
                    self.logger.info(f"  Config: kernel_size={seq_item.config['kernel_size']}, "
                                   f"stride={seq_item.config.get('stride', 1)}, "
                                   f"padding={seq_item.config.get('padding', 0)}, "
                                   f"channels={seq_item.config.get('input_channels', 1)}->"
                                   f"{seq_item.config.get('output_channels', 1)}")
                    self.logger.info(f"  Expected:\n{seq_item.expected_output}")
                    self.logger.info(f"  Actual:\n{seq_item.actual_output}")
                    correct_cases += 1

        assert failed_cases == 0, f"Test failed with {failed_cases} failed cases"

        self.logger.info(f"Check phase completed: {correct_cases} correct cases, {failed_cases} failed cases")