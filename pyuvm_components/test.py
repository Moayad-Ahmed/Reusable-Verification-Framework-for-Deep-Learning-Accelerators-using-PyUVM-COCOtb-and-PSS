import cocotb 
import pyuvm
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
from pyuvm import *
from env import My_Env
from sequences import ConfigDrivenSequence, ChainedLayerSequence

@pyuvm.test()
class Pool_Basic_Test(uvm_test):
    def build_phase(self):
        self.My_Env = My_Env("My_Env", self)

    def end_of_elaboration_phase(self):
        self.sqr = ConfigDB().get(None, "", "SEQR")
        
    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        one_layer_max_pool = ConfigDrivenSequence("one_layer_max_pool", "yaml_files/one_layer_max_pool.yaml")

        self.raise_objection()

        await one_layer_max_pool.start(self.sqr)
        # Wait for pipeline to drain - ensure all responses are captured
        await ClockCycles(cocotb.top.clk, 10)
        
        self.drop_objection()
        

@pyuvm.test()
class Successive_Min_Avg_Pool(Pool_Basic_Test):
    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        Successive_Min_Avg_Pool = ConfigDrivenSequence("Successive_Min_Avg_Pool", "yaml_files/successive_min_avg_pool.yaml")

        self.raise_objection()

        await Successive_Min_Avg_Pool.start(self.sqr)
        
        # Wait for pipeline to drain - ensure all responses are captured
        await ClockCycles(cocotb.top.clk, 10)
        
        self.drop_objection()


@pyuvm.test()
class Chained_Pool_Layers_Test(Pool_Basic_Test):
    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        chained_layer_seq = ChainedLayerSequence("chained_layer_seq", "yaml_files/Chained_Pool_Layers.yaml")

        self.raise_objection()

        await chained_layer_seq.start(self.sqr)
        
        # Wait for pipeline to drain - ensure all responses are captured
        await ClockCycles(cocotb.top.clk, 10)
        
        self.drop_objection()

@pyuvm.test()
class Conv_Basic_Test(uvm_test):
    """Test single-channel 3x3 convolution"""
    def build_phase(self):
        self.My_Env = My_Env("My_Env", self)

    def end_of_elaboration_phase(self):
        self.sqr = ConfigDB().get(None, "", "SEQR")
        
    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        conv_seq = ConfigDrivenSequence("conv_seq", "yaml_files/single_channel_conv.yaml")

        self.raise_objection()
        await conv_seq.start(self.sqr)
        await ClockCycles(cocotb.top.clk, 10)
        self.drop_objection()


@pyuvm.test()
class Conv_MultiChannel_Test(Conv_Basic_Test):
    """Test multi-channel convolution"""
    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        multi_conv_seq = ConfigDrivenSequence("multi_conv_seq", "yaml_files/multi_channel_conv.yaml")

        self.raise_objection()
        await multi_conv_seq.start(self.sqr)
        await ClockCycles(cocotb.top.clk, 10)
        self.drop_objection()


@pyuvm.test()
class Conv_Pool_Test(uvm_test):
    """Test a simple 2-layer network: convolution followed by pooling"""
    def build_phase(self):
        self.My_Env = My_Env("My_Env", self)

    def end_of_elaboration_phase(self):
        self.sqr = ConfigDB().get(None, "", "SEQR")
        
    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        conv_pool_seq = ChainedLayerSequence("conv_pool_seq", "yaml_files/mix_conv_pool.yaml")

        self.raise_objection()
        await conv_pool_seq.start(self.sqr)
        await ClockCycles(cocotb.top.clk, 10)
        self.drop_objection()

@pyuvm.test()
class FullyConnected_Test(uvm_test):
    """Test a simple fully connected layer"""
    def build_phase(self):
        self.My_Env = My_Env("My_Env", self)

    def end_of_elaboration_phase(self):
        self.sqr = ConfigDB().get(None, "", "SEQR")
        
    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        fully_connected_seq = ConfigDrivenSequence("fully_connected_seq", "yaml_files/fully_connected.yaml")

        self.raise_objection()
        await fully_connected_seq.start(self.sqr)
        await ClockCycles(cocotb.top.clk, 10)
        self.drop_objection()

@pyuvm.test()
class Conv_Pool_FC_Test(uvm_test):
    """Test a 3-layer network: convolution followed by pooling followed by fully connected layer"""
    def build_phase(self):
        self.My_Env = My_Env("My_Env", self)

    def end_of_elaboration_phase(self):
        self.sqr = ConfigDB().get(None, "", "SEQR")
        
    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        conv_pool_fc_seq = ChainedLayerSequence("conv_pool_fc_seq", "yaml_files/conv_pool_fc.yaml")

        self.raise_objection()
        await conv_pool_fc_seq.start(self.sqr)
        await ClockCycles(cocotb.top.clk, 10)
        self.drop_objection()
