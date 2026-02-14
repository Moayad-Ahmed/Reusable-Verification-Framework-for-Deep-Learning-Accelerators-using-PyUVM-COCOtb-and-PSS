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


# ------------------------------------------------------------------ #
#  Base class — common clock setup, env, objection logic             #
# ------------------------------------------------------------------ #

class PdpTestBase(uvm_test):
    """Base class for all PDP pooling tests."""

    # Subclasses override these two class attributes
    YAML_FILE = None        # e.g. "4x4_max_k2_s2.yaml"
    DAT_FILE  = None        # e.g. "4x4_max_k2_s2_in.dat"

    def build_phase(self):
        self.env = NVDLA_Env("NVDLA_Env", self)

    def end_of_elaboration_phase(self):
        self.sqr = ConfigDB().get(None, "", "SEQR")

    async def run_phase(self):
        cocotb.start_soon(
            Clock(cocotb.top.dla_core_clk, 20, unit="ns").start()
        )
        cocotb.start_soon(
            Clock(cocotb.top.dla_csb_clk, 20, unit="ns").start()
        )

        pdp_test = PdpTestSequence(
            "pdp_test",
            input_file=input_file_path(self.DAT_FILE),
            config_file=config_file_path(self.YAML_FILE),
        )

        self.raise_objection()
        await pdp_test.start(self.sqr)

        while cocotb.top.dla_intr.value != 1:
            await RisingEdge(cocotb.top.dla_core_clk)

        await ClockCycles(cocotb.top.dla_core_clk, 1000)
        self.drop_objection()


# ------------------------------------------------------------------ #
#  Concrete test classes — one per YAML configuration                #
# ------------------------------------------------------------------ #

@pyuvm.test()
class PdpBasicTest(PdpTestBase):
    """Default test — uses nvdla_pooling_config.yaml"""
    YAML_FILE = "nvdla_pooling_config.yaml"
    DAT_FILE  = "pdp_default_in.dat"


@pyuvm.test()
class Pdp_4x4_MAX_k2_s2(PdpTestBase):
    """4x4 Max Pooling, kernel=2, stride=2 → 2x2 output"""
    YAML_FILE = "4x4_max_k2_s2.yaml"
    DAT_FILE  = "pdp_4x4_max_k2_s2_in.dat"


@pyuvm.test()
class Pdp_6x6_AVG_k2_s2(PdpTestBase):
    """6x6 Avg Pooling, kernel=2, stride=2 → 3x3 output"""
    YAML_FILE = "6x6_avg_k2_s2.yaml"
    DAT_FILE  = "pdp_6x6_avg_k2_s2_in.dat"


@pyuvm.test()
class Pdp_4x4_MIN_k2_s1(PdpTestBase):
    """4x4 Min Pooling, kernel=2, stride=1 → 3x3 output"""
    YAML_FILE = "4x4_min_k2_s1.yaml"
    DAT_FILE  = "pdp_4x4_min_k2_s1_in.dat"


@pyuvm.test()
class Pdp_8x8_MAX_k4_s4(PdpTestBase):
    """8x8 Max Pooling, kernel=4, stride=4 → 2x2 output"""
    YAML_FILE = "8x8_max_k4_s4.yaml"
    DAT_FILE  = "pdp_8x8_max_k4_s4_in.dat"


@pyuvm.test()
class Pdp_8x8_AVG_k2_s2(PdpTestBase):
    """8x8 Avg Pooling, kernel=2, stride=2 → 4x4 output"""
    YAML_FILE = "8x8_avg_k2_s2.yaml"
    DAT_FILE  = "pdp_8x8_avg_k2_s2_in.dat"


@pyuvm.test()
class Pdp_5x5_MAX_k3_s1_pad1(PdpTestBase):
    """5x5 Max Pooling, kernel=3, stride=1, pad=1 → 5x5 output"""
    YAML_FILE = "5x5_max_k3_s1_pad1.yaml"
    DAT_FILE  = "pdp_5x5_max_k3_s1_pad1_in.dat"


@pyuvm.test()
class Pdp_6x6_MAX_k3_s1_pad2_valm64(PdpTestBase):
    """6x6 Max Pooling, kernel=3, stride=1, pad=2, pad_value=-64 → 8x8 output"""
    YAML_FILE = "6x6_max_k3_s1_pad2_valm64.yaml"
    DAT_FILE  = "pdp_6x6_max_k3_s1_pad2_valm64_in.dat"

@pyuvm.test()
class Pdp_4x4_AVG_k3_s2_pad3(PdpTestBase):
    """4x4 Avg Pooling, kernel=3, stride=2, pad=3, pad_value=-64 → 4x4 output"""
    YAML_FILE = "4x4_avg_k3_s2_pad3.yaml"
    DAT_FILE  = "pdp_4x4_avg_k3_s2_pad3_in.dat"

@pyuvm.test()
class Pdp_4x4_max_k2_s2_4ch(PdpTestBase):
    """4x4 Max Pooling, kernel=2, stride=2, 4 channels"""
    YAML_FILE = "4x4_max_k2_s2_4ch.yaml"
    DAT_FILE  = "pdp_4x4_max_k2_s2_4ch_in.dat"

