import cocotb 
import pyuvm
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
from pyuvm import *
from env import My_Env
from sequences import ConfigDrivenSequence

@pyuvm.test()
class My_Test(uvm_test):
    def build_phase(self):
        self.My_Env = My_Env("My_Env", self)

    def end_of_elaboration_phase(self):
        self.sqr = ConfigDB().get(None, "", "SEQR")
        
    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        one_layer_max_pool = ConfigDrivenSequence("config_driven_seq", "one_layer_max_pool.yaml")

        self.raise_objection()

        await one_layer_max_pool.start(self.sqr)
        # Wait for pipeline to drain - ensure all responses are captured
        await ClockCycles(cocotb.top.clk, 10)
        
        self.drop_objection()

@pyuvm.test()
class My_Test_Min_Avg_Pool(My_Test):
    async def run_phase(self):
        cocotb.start_soon(Clock(cocotb.top.clk, 2, "ns").start())
        one_layer_min_avg_pool = ConfigDrivenSequence("config_driven_seq", "one_layer_min_avg_pool.yaml")

        self.raise_objection()

        await one_layer_min_avg_pool.start(self.sqr)
        
        # Wait for pipeline to drain - ensure all responses are captured
        await ClockCycles(cocotb.top.clk, 10)
        
        self.drop_objection()