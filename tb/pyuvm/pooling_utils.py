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

class PoolingBFM(metaclass=utility_classes.Singleton):
    def __init__(self):
        self.dut = cocotb.top
        self.drv_queue = Queue(maxsize=1)
        self.mtr_queue = Queue(maxsize=0)
        self.seq_item_queue = Queue(maxsize=0)

    async def send_config(self, seq_item):
        if seq_item.layer_type == 'pooling':
            await self.seq_item_queue.put(seq_item)
            driver_tuple = (1, seq_item.input_data, seq_item.config['pool_type'])
            await self.drv_queue.put(driver_tuple)

    async def get_result(self):
        result = await self.mtr_queue.get()
        seq_item = await self.seq_item_queue.get()
        output_height = floor((seq_item.config['input_shape'][0] - seq_item.config['kernel_size']) / seq_item.config['stride']) + 1
        output_width = floor((seq_item.config['input_shape'][1] - seq_item.config['kernel_size']) / seq_item.config['stride'])  + 1
        seq_item.actual_output = self.int_to_output_matrix(result, shape=(output_height, output_width))
        return seq_item
    
    async def reset(self):
        await RisingEdge(self.dut.clk)
        self.dut.rst_n.value = 0
        self.dut.valid_in.value = 0
        self.dut.data_in.value = 0
        self.dut.pool_mode.value = 0
        await RisingEdge(self.dut.clk)
        self.dut.rst_n.value = 1
        await RisingEdge(self.dut.clk)

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
        
        # Extract elements from packed integer
        num_elements = shape[0] * shape[1]
        flat = []
        for i in range(num_elements):
            elem = (packed_value >> (i * 8)) & 0xFF
            flat.append(elem)
        
        # Reshape to 2D array
        return np.array(flat).reshape(shape)

    async def driver_bfm(self):
        self.dut.rst_n.value = 1
        self.dut.valid_in.value = 0
        self.dut.data_in.value = 0
        self.dut.pool_mode.value = 0
        while True:
            try:
                await RisingEdge(self.dut.clk)
                (valid_in, data_in, pool_mode) = self.drv_queue.get_nowait()
                self.dut.valid_in.value = valid_in
                # Convert matrix to packed integer for DUT
                packed_data = self.flatten_matrix_to_int(data_in)
                self.dut.data_in.value = packed_data
                if pool_mode == 'max':
                    self.dut.pool_mode.value = 0
                elif pool_mode == 'avg':
                    self.dut.pool_mode.value = 1
                elif pool_mode == 'min':
                    self.dut.pool_mode.value = 2
            except QueueEmpty:
                pass
            
            
    async def monitor_bfm(self):
        while True:
            await RisingEdge(self.dut.clk)
            if self.dut.valid_out.value == 1:
                result = (self.dut.data_out.value)
                await self.mtr_queue.put(result)
    

    def start_bfm(self):
        cocotb.start_soon(self.driver_bfm())
        cocotb.start_soon(self.monitor_bfm())