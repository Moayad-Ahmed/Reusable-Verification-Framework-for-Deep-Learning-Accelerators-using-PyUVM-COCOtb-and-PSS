import cocotb 
import pyuvm
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
from pyuvm import *
from env import My_Env
from sequences import ConvolutionSequence

@pyuvm.test()
class Conv_Basic_Test(uvm_test):
    """Test single-channel 3x3 convolution"""
    def build_phase(self):
        self.My_Env = My_Env("My_Env", self)

    def end_of_elaboration_phase(self):
        self.sqr = ConfigDB().get(None, "", "SEQR")
        
    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        conv_seq = ConvolutionSequence("conv_seq", "single_channel_conv.yaml")

        self.raise_objection()
        await conv_seq.start(self.sqr)
        await ClockCycles(cocotb.top.clk, 10)
        self.drop_objection()


@pyuvm.test()
class Conv_MultiChannel_Test(Conv_Basic_Test):
    """Test multi-channel convolution"""
    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        conv_seq = ConvolutionSequence("multi_conv_seq", "multi_channel_conv.yaml")

        self.raise_objection()
        await conv_seq.start(self.sqr)
        await ClockCycles(cocotb.top.clk, 10)
        self.drop_objection()