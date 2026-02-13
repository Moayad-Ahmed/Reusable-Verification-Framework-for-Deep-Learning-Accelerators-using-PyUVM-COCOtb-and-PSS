import cocotb
import pyuvm
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge
from pyuvm import *
from pyuvm_components.env import NVDLA_Env
from pyuvm_components.sequences import PdpTestSequence

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "..", "input_files")
CONFIG_DIR = os.path.join(BASE_DIR, "..", "yaml")


def input_file_path(filename):
    return os.path.join(INPUT_DIR, filename)


def config_file_path(filename):
    return os.path.join(CONFIG_DIR, filename)


@pyuvm.test()
class PdpBasicTest(uvm_test):
    """Runs the pdp_1x1x1_3x3_ave_int8_0 test through NVDLA PDP"""

    def build_phase(self):
        self.env = NVDLA_Env("NVDLA_Env", self)

    def end_of_elaboration_phase(self):
        self.sqr = ConfigDB().get(None, "", "SEQR")

    async def run_phase(self):
        # Start both clocks (20 ns period = 50 MHz clock)
        cocotb.start_soon(
            Clock(cocotb.top.dla_core_clk, 20, unit="ns").start()
        )
        cocotb.start_soon(
            Clock(cocotb.top.dla_csb_clk, 20, unit="ns").start()
        )

        pdp_test = PdpTestSequence(
            "pdp_test",
            input_file=input_file_path("pdp_1x1x1_3x3_ave_int8_0_in.dat"),
            config_file=config_file_path("nvdla_pooling_config.yaml")
        )

        self.raise_objection()

        await pdp_test.start(self.sqr)

        # Wait for NVDLA interrupt to fire
        while cocotb.top.dla_intr.value != 1:
            await RisingEdge(cocotb.top.dla_core_clk)

        # Wait some extra cycles to ensure that the scoreboard has completed its checks
        await ClockCycles(cocotb.top.dla_core_clk, 1000)

        self.drop_objection()
