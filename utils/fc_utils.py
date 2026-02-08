import logging
import math
from math import *

import cocotb
from cocotb.queue import Queue, QueueEmpty
from cocotb.triggers import RisingEdge

from pyuvm import utility_classes

import numpy as np

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class FullyConnectedBFM(metaclass=utility_classes.Singleton):
    def __init__(self):
        self.dut = cocotb.top
        self.drv_queue = Queue(maxsize=1)
        self.mtr_queue = Queue(maxsize=0)
        self.seq_item_queue = Queue(maxsize=0)
        self.result_ready_queue = Queue(maxsize=0)  # Queue for sequence to wait on

    async def send_config(self, seq_item):
        await self.seq_item_queue.put(seq_item)
        driver_tuple = (1, seq_item.input_data, seq_item.fc_weights, seq_item.fc_bias, seq_item.config)
        await self.drv_queue.put(driver_tuple)

    async def get_result(self):
        result = await self.mtr_queue.get()
        seq_item = await self.seq_item_queue.get()
            
        seq_item.actual_output = result
        # Notify waiting sequences that result is ready
        await self.result_ready_queue.put(seq_item.actual_output)
        return seq_item

    async def wait_for_result(self):
        """Wait for the monitor to capture the actual DUT output and return it"""
        result = await self.result_ready_queue.get()
        return result

    async def reset(self):
        await RisingEdge(self.dut.clk)
        # Initialize configurable inputs during reset
        self.dut.rst_n.value = 1
        self.dut.en.value = 0
        self.dut.in_vec.value = 0
        self.dut.weights.value = 0  
        self.dut.bias.value = 0
        
        await RisingEdge(self.dut.clk)
        self.dut.rst_n.value = 1
        await RisingEdge(self.dut.clk)


    async def driver_bfm(self):
        # Initialize with default values
        self.dut.rst_n.value = 1
        self.dut.en.value = 0
        self.dut.in_vec.value = 0
        self.dut.weights.value = 0  
        self.dut.bias.value = 0
        
        while True:
            await RisingEdge(self.dut.clk)
            try:
                (en, data_in, weights, bias, config) = self.drv_queue.get_nowait()
                self.dut.en.value = en
                
                self.dut.in_vec.value = data_in
                self.dut.bias.value = bias
                self.dut.weights.value = weights
                
                
            except QueueEmpty:
                # Deassert valid_in when no new transaction
                self.dut.en.value = 0
    
    async def monitor_bfm(self):
        while True:
            await RisingEdge(self.dut.clk)
            if self.dut.valid.value == 1:
                result = (self.dut.out_vec.value)
                await self.mtr_queue.put(result)

            

    def start_bfm(self):
        cocotb.start_soon(self.driver_bfm())
        cocotb.start_soon(self.monitor_bfm())