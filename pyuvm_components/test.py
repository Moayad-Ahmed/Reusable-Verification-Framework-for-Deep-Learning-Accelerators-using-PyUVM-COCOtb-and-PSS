import cocotb 
import pyuvm
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
from pyuvm import *
from pyuvm_components.env import My_Env
from pyuvm_components.sequences import ConfigDrivenSequence, ChainedLayerSequence

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_DIR = os.path.join(BASE_DIR, "..", "yaml_files")

def yaml_file_path(filename):
    return os.path.join(YAML_DIR, filename)

@pyuvm.test()
class Pool_Basic_Test(uvm_test):
    def build_phase(self):
        self.My_Env = My_Env("My_Env", self)

    def end_of_elaboration_phase(self):
        self.sqr = ConfigDB().get(None, "", "SEQR")
        
    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        one_layer_max_pool = ConfigDrivenSequence("one_layer_max_pool", yaml_file_path("one_layer_max_pool.yaml"))

        self.raise_objection()

        await one_layer_max_pool.start(self.sqr)
        # Wait for pipeline to drain - ensure all responses are captured
        await ClockCycles(cocotb.top.clk, 10)
        
        self.drop_objection()
        

@pyuvm.test()
class Successive_Min_Avg_Pool(Pool_Basic_Test):
    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        Successive_Min_Avg_Pool = ConfigDrivenSequence("Successive_Min_Avg_Pool", yaml_file_path("successive_min_avg_pool.yaml"))

        self.raise_objection()

        await Successive_Min_Avg_Pool.start(self.sqr)
        
        # Wait for pipeline to drain - ensure all responses are captured
        await ClockCycles(cocotb.top.clk, 10)
        
        self.drop_objection()


@pyuvm.test()
class Chained_Pool_Layers_Test(Pool_Basic_Test):
    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        chained_layer_seq = ChainedLayerSequence("chained_layer_seq", yaml_file_path("Chained_Pool_Layers.yaml"))

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
        conv_seq = ConfigDrivenSequence("conv_seq", yaml_file_path("single_channel_conv.yaml"))

        self.raise_objection()
        await conv_seq.start(self.sqr)
        await ClockCycles(cocotb.top.clk, 10)
        self.drop_objection()


@pyuvm.test()
class Conv_MultiChannel_Test(Conv_Basic_Test):
    """Test multi-channel convolution"""
    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        multi_conv_seq = ConfigDrivenSequence("multi_conv_seq", yaml_file_path("multi_channel_conv.yaml"))

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
        conv_pool_seq = ChainedLayerSequence("conv_pool_seq", yaml_file_path("mix_conv_pool.yaml"))

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
        fully_connected_seq = ConfigDrivenSequence("fully_connected_seq", yaml_file_path("fully_connected.yaml"))

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
        conv_pool_fc_seq = ChainedLayerSequence("conv_pool_fc_seq", yaml_file_path("conv_pool_fc.yaml"))

        self.raise_objection()
        await conv_pool_fc_seq.start(self.sqr)
        await ClockCycles(cocotb.top.clk, 10)
        self.drop_objection()


@pyuvm.test()
class Activation_ReLU_Test(uvm_test):
    """Test ReLU activation function on a 4x4 matrix"""
    def build_phase(self):
        self.My_Env = My_Env("My_Env", self)

    def end_of_elaboration_phase(self):
        self.sqr = ConfigDB().get(None, "", "SEQR")

    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        relu_seq = ConfigDrivenSequence("relu_seq", yaml_file_path("one_layer_activation_relu.yaml"))

        self.raise_objection()
        await relu_seq.start(self.sqr)
        await ClockCycles(cocotb.top.clk, 10)
        self.drop_objection()


@pyuvm.test()
class Activation_Sigmoid_Test(uvm_test):
    """Test Sigmoid activation function on a 4x4 matrix"""
    def build_phase(self):
        self.My_Env = My_Env("My_Env", self)

    def end_of_elaboration_phase(self):
        self.sqr = ConfigDB().get(None, "", "SEQR")

    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        sigmoid_seq = ConfigDrivenSequence("sigmoid_seq", yaml_file_path("one_layer_activation_sigmoid.yaml"))

        self.raise_objection()
        await sigmoid_seq.start(self.sqr)
        await ClockCycles(cocotb.top.clk, 10)
        self.drop_objection()


@pyuvm.test()
class Activation_Tanh_Test(uvm_test):
    """Test Tanh activation function on a 4x4 matrix"""
    def build_phase(self):
        self.My_Env = My_Env("My_Env", self)

    def end_of_elaboration_phase(self):
        self.sqr = ConfigDB().get(None, "", "SEQR")

    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        tanh_seq = ConfigDrivenSequence("tanh_seq", yaml_file_path("one_layer_activation_tanh.yaml"))

        self.raise_objection()
        await tanh_seq.start(self.sqr)
        await ClockCycles(cocotb.top.clk, 10)
        self.drop_objection()


@pyuvm.test()
class Activation_Softmax_Test(uvm_test):
    """Test Softmax activation function on a 4x4 matrix"""
    def build_phase(self):
        self.My_Env = My_Env("My_Env", self)

    def end_of_elaboration_phase(self):
        self.sqr = ConfigDB().get(None, "", "SEQR")

    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        softmax_seq = ConfigDrivenSequence("softmax_seq", yaml_file_path("one_layer_activation_softmax.yaml"))

        self.raise_objection()
        await softmax_seq.start(self.sqr)
        await ClockCycles(cocotb.top.clk, 10)
        self.drop_objection()


@pyuvm.test()
class Conv_Activation_Pool_FC_Test(uvm_test):
    """Test Conv Activation Pool FC Layer"""
    def build_phase(self):
        self.My_Env = My_Env("My_Env", self)

    def end_of_elaboration_phase(self):
        self.sqr = ConfigDB().get(None, "", "SEQR")

    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        conv_act_pool_fc_seq = ChainedLayerSequence("conv_act_pool_fc_seq", yaml_file_path("conv_act_pool_fc.yaml"))

        self.raise_objection()
        await conv_act_pool_fc_seq.start(self.sqr)
        await ClockCycles(cocotb.top.clk, 10)
        self.drop_objection()

