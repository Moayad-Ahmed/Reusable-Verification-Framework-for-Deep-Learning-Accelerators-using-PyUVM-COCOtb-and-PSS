import cocotb
import pyuvm
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge
from pyuvm import *
from pyuvm_components.env import NVDLA_Env
from pyuvm_components.sequences import (
    NVDLAVirtualSequencer,
    PdpTestSequence,
    ConvTestSequence,
    FcTestSequence,
)
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(BASE_DIR, "..", "input_files")
CONFIG_DIR = os.path.join(BASE_DIR, "..", "yaml")


def input_file_path(filename):
    return os.path.join(INPUT_DIR, filename)


def config_file_path(filename):
    return os.path.join(CONFIG_DIR, filename)


# ══════════════════════════════════════════════════════════════════════
#  BASE TEST
# ══════════════════════════════════════════════════════════════════════

class NVDLATestBase(uvm_test):
    """
    Common base class for all NVDLA tests.

    build_phase         : creates NVDLA_Env (DataAgent + CsbAgent + Scoreboard)
    end_of_elaboration  : builds the NVDLAVirtualSequencer which retrieves both
                          sub-sequencers from ConfigDB (registered by the agents)
    run_phase           : starts clocks, runs the virtual sequence on vseqr,
                          waits for dla_intr, then drops objection
    """

    YAML_FILE = None
    DAT_FILE  = None

    def build_phase(self):
        self.env = NVDLA_Env("NVDLA_Env", self)

    def end_of_elaboration_phase(self):
        # Instantiate the virtual sequencer and manually wire in both
        # sub-sequencer references from ConfigDB.
        # This must happen in end_of_elaboration_phase (not build_phase)
        # so that both agents have already registered their sequencers.
        self.vseqr = NVDLAVirtualSequencer("vseqr", self)
        self.vseqr.data_sqr = ConfigDB().get(self, "", "DATA_SEQR")
        self.vseqr.csb_sqr  = ConfigDB().get(self, "", "CSB_SEQR")

    async def _start_clocks(self):
        """Start DLA core and CSB clocks at 20 ns period (50 MHz)."""
        cocotb.start_soon(Clock(cocotb.top.dla_core_clk, 20, unit="ns").start())
        cocotb.start_soon(Clock(cocotb.top.dla_csb_clk,  20, unit="ns").start())

    def _create_sequence(self):
        """Subclasses return the correct virtual sequence instance."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _create_sequence()"
        )

    async def run_phase(self):
        await self._start_clocks()
        seq = self._create_sequence()

        self.raise_objection()
        # Start the virtual sequence on the virtual sequencer.
        # Internally it dispatches DataTransactions to data_sqr and
        # CsbTransactions to csb_sqr in the correct order each iteration.
        await seq.start(self.vseqr)

        # Wait for final NVDLA interrupt then allow output to settle
        while cocotb.top.dla_intr.value != 1:
            await RisingEdge(cocotb.top.dla_core_clk)

        await ClockCycles(cocotb.top.dla_core_clk, 1000)
        self.drop_objection()


# ══════════════════════════════════════════════════════════════════════
#  PDP BASE
# ══════════════════════════════════════════════════════════════════════

class PdpTestBase(NVDLATestBase):
    """Base class for all PDP pooling tests."""

    def _create_sequence(self):
        return PdpTestSequence(
            "pdp_test",
            input_file=input_file_path(self.DAT_FILE),
            config_file=config_file_path(self.YAML_FILE),
        )


# ── Concrete PDP tests ─────────────────────────────────────────────

@pyuvm.test()
class PdpBasicTest(PdpTestBase):
    """Default pooling test — nvdla_pooling_config.yaml"""
    YAML_FILE = "nvdla_pooling_config.yaml"
    DAT_FILE  = "pdp_default_in.dat"


@pyuvm.test()
class Pdp_4x4_MAX_k2_s2(PdpTestBase):
    """4x4 Max Pooling, kernel=2, stride=2 -> 2x2 output"""
    YAML_FILE = "4x4_max_k2_s2.yaml"
    DAT_FILE  = "pdp_4x4_max_k2_s2_in.dat"


@pyuvm.test()
class Pdp_6x6_AVG_k2_s2(PdpTestBase):
    """6x6 Avg Pooling, kernel=2, stride=2 -> 3x3 output"""
    YAML_FILE = "6x6_avg_k2_s2.yaml"
    DAT_FILE  = "pdp_6x6_avg_k2_s2_in.dat"


@pyuvm.test()
class Pdp_4x4_MIN_k2_s1(PdpTestBase):
    """4x4 Min Pooling, kernel=2, stride=1 -> 3x3 output"""
    YAML_FILE = "4x4_min_k2_s1.yaml"
    DAT_FILE  = "pdp_4x4_min_k2_s1_in.dat"


@pyuvm.test()
class Pdp_8x8_MAX_k4_s4(PdpTestBase):
    """8x8 Max Pooling, kernel=4, stride=4 -> 2x2 output"""
    YAML_FILE = "8x8_max_k4_s4.yaml"
    DAT_FILE  = "pdp_8x8_max_k4_s4_in.dat"


@pyuvm.test()
class Pdp_8x8_AVG_k2_s2(PdpTestBase):
    """8x8 Avg Pooling, kernel=2, stride=2 -> 4x4 output"""
    YAML_FILE = "8x8_avg_k2_s2.yaml"
    DAT_FILE  = "pdp_8x8_avg_k2_s2_in.dat"


@pyuvm.test()
class Pdp_5x5_MAX_k3_s1_pad1(PdpTestBase):
    """5x5 Max Pooling, kernel=3, stride=1, pad=1 -> 5x5 output"""
    YAML_FILE = "5x5_max_k3_s1_pad1.yaml"
    DAT_FILE  = "pdp_5x5_max_k3_s1_pad1_in.dat"


@pyuvm.test()
class Pdp_6x6_MAX_k3_s1_pad2_valm64(PdpTestBase):
    """6x6 Max Pooling, kernel=3, stride=1, pad=2, pad_value=-64"""
    YAML_FILE = "6x6_max_k3_s1_pad2_valm64.yaml"
    DAT_FILE  = "pdp_6x6_max_k3_s1_pad2_valm64_in.dat"


@pyuvm.test()
class Pdp_4x4_AVG_k3_s2_pad3(PdpTestBase):
    """4x4 Avg Pooling, kernel=3, stride=2, pad=3"""
    YAML_FILE = "4x4_avg_k3_s2_pad3.yaml"
    DAT_FILE  = "pdp_4x4_avg_k3_s2_pad3_in.dat"


@pyuvm.test()
class Pdp_4x4_max_k2_s2_4ch(PdpTestBase):
    """4x4 Max Pooling, kernel=2, stride=2, 4 channels"""
    YAML_FILE = "4x4_max_k2_s2_4ch.yaml"
    DAT_FILE  = "pdp_4x4_max_k2_s2_4ch_in.dat"


# ══════════════════════════════════════════════════════════════════════
#  CONV BASE
# ══════════════════════════════════════════════════════════════════════

class ConvTestBase(NVDLATestBase):
    """
    Base class for all NVDLA convolution tests.

    Convolution pipeline: CDMA -> CSC -> CMAC_A/B -> CACC -> SDP (passthrough).
    The result in DRAM is the raw convolution-truncation output.
    """
    WT_FILE = None

    def _create_sequence(self):
        return ConvTestSequence(
            "conv_test",
            input_file=input_file_path(self.DAT_FILE),
            weight_file=input_file_path(self.WT_FILE),
            config_file=config_file_path(self.YAML_FILE),
        )


# ── Concrete convolution tests ────────────────────────────────────────

@pyuvm.test()
class Conv_DC_1x1x8_k1(ConvTestBase):
    """1×1 DC conv: 8 input channels, 1 kernel, no truncation"""
    YAML_FILE = "dc_1x1x8_k1_simple.yaml"
    DAT_FILE  = "conv_1x1x8_k1_in.dat"
    WT_FILE   = "conv_1x1x8_k1_wt.dat"


@pyuvm.test()
class Conv_DC_2x1x8_k1(ConvTestBase):
    """2×1 DC conv: 8 input channels, 1 kernel, no truncation"""
    YAML_FILE = "dc_2x1x8_k1_simple.yaml"
    DAT_FILE  = "conv_2x1x8_k1_in.dat"
    WT_FILE   = "conv_2x1x8_k1_wt.dat"


# ══════════════════════════════════════════════════════════════════════
#  FC (FULLY-CONNECTED) BASE
# ══════════════════════════════════════════════════════════════════════

class FcTestBase(NVDLATestBase):
    """
    Base class for all NVDLA fully-connected tests.

    FC layers are mapped to the convolution pipeline as 1×1 convolutions:
        CDMA -> CSC -> CMAC_A/B -> CACC -> SDP (passthrough) -> DRAM
    """
    WT_FILE = None

    def _create_sequence(self):
        return FcTestSequence(
            "fc_test",
            input_file=input_file_path(self.DAT_FILE),
            weight_file=input_file_path(self.WT_FILE),
            config_file=config_file_path(self.YAML_FILE),
        )


# ── Concrete FC tests ──────────────────────────────────────────────


@pyuvm.test()
class FC_16in_8out(FcTestBase):
    """Fully-connected: 16 inputs, 8 outputs, INT8, no truncation"""
    YAML_FILE = "fc_16in_8out_simple.yaml"
    DAT_FILE  = "fc_16in_8out_in.dat"
    WT_FILE   = "fc_16in_8out_wt.dat"


@pyuvm.test()
class FC_64in_32out(FcTestBase):
    """Fully-connected: 64 inputs, 32 outputs, INT8, truncate=4"""
    YAML_FILE = "fc_64in_32out.yaml"
    DAT_FILE  = "fc_64in_32out_in.dat"
    WT_FILE   = "fc_64in_32out_wt.dat"

@pyuvm.test()
class FC_8in_1out(FcTestBase):
    """Fully-connected: 8 inputs, 1 output, INT8, no truncation"""
    YAML_FILE = "fc_8in_1out_simple.yaml"
    DAT_FILE  = "fc_8in_1out_in.dat"
    WT_FILE   = "fc_8in_1out_wt.dat"
