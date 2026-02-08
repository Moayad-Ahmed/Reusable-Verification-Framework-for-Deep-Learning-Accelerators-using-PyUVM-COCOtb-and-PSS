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
    
    def flatten_matrix_to_int(self, matrix):
        """Convert a 2D matrix to a packed integer for the DUT.
        Each element is 8 bits, packed in row-major order (E1, E2, ..., E9)."""
        # Flatten the matrix in row-major order
        if isinstance(matrix, np.ndarray):
            flat = matrix.flatten().tolist()
        else:
            flat = [elem for row in matrix for elem in row]
        
        # Pack elements into a single integer (E1 at LSB, E9 at MSB)
        packed_value = 0
        for i, elem in enumerate(flat):
            packed_value |= (int(elem) & 0xFF) << (i * 8)
        return packed_value
    
    def int_to_output_matrix(self, packed_value, shape):
        """Convert a packed integer from the DUT back to a 2D matrix.
        Each element is 8 bits, packed in row-major order."""
        # Convert to integer if it's a cocotb BinaryValue/LogicArray
        if hasattr(packed_value, 'to_unsigned'):
            packed_value = packed_value.to_unsigned()
        else:
            packed_value = int(packed_value)
        
        # Extract elements from packed integer (interpret as signed int8)
        num_elements = shape[0] * shape[1]
        flat = []
        for i in range(num_elements):
            elem = (packed_value >> (i * 8)) & 0xFF
            if elem > 127:
                elem -= 256
            flat.append(elem)
        
        # Reshape to 2D array
        return np.array(flat, dtype=np.int8).reshape(shape)

    async def reset(self):
        await RisingEdge(self.dut.clk)
        # Initialize configurable inputs during reset
        self.dut.rst_n.value = 1
        self.dut.fc_en.value = 0
        self.dut.fc_in_vec.value = 0
        self.dut.fc_weights.value = 0  
        self.dut.fc_bias.value = 0
        self.dut.fc_actual_input_size.value = 0
        self.dut.fc_actual_output_size.value = 0
        
        await RisingEdge(self.dut.clk)
        self.dut.rst_n.value = 1
        await RisingEdge(self.dut.clk)


    async def driver_bfm(self):
        # Initialize with default values
        self.dut.rst_n.value = 1
        self.dut.fc_en.value = 0
        self.dut.fc_in_vec.value = 0
        self.dut.fc_weights.value = 0  
        self.dut.fc_bias.value = 0
        self.dut.fc_actual_input_size.value = 0
        self.dut.fc_actual_output_size.value = 0
        
        while True:
            await RisingEdge(self.dut.clk)
            try:
                (en, data_in, weights, bias, config) = self.drv_queue.get_nowait()
                self.dut.fc_en.value = en
                
                self.dut.fc_actual_input_size.value = int(config['input_size'])
                self.dut.fc_actual_output_size.value = int(config['output_size'])
                
                self.dut.fc_in_vec.value = self.flatten_matrix_to_int(data_in)
                self.dut.fc_bias.value = self.flatten_matrix_to_int(bias)
                self.dut.fc_weights.value = self.flatten_matrix_to_int(weights)
            except QueueEmpty:
                # Deassert valid_in when no new transaction
                self.dut.fc_en.value = 0
    
    async def monitor_bfm(self):
        while True:
            await RisingEdge(self.dut.clk)
            if self.dut.fc_valid.value == 1:
                output_size = int(self.dut.fc_actual_output_size.value)
                result = self.int_to_output_matrix(self.dut.fc_out_vec.value, (1, output_size)).flatten()
                await self.mtr_queue.put(result)

            

    def start_bfm(self):
        cocotb.start_soon(self.driver_bfm())
        cocotb.start_soon(self.monitor_bfm())