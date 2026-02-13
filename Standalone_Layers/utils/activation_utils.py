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

class ActivationBFM(metaclass=utility_classes.Singleton):
    """
    Bus Functional Model for Activation Layer DUT
    
    DUT Interface:
    - clk, rst_n: Clock and reset
    - func_sel[1:0]: 00=ReLU, 01=Sigmoid, 10=Tanh, 11=Softmax
    - data_in[7:0]: Input data as 8-bit unsigned integer [0-255]
    - valid_in: Input valid signal
    - vector_in[3:0][7:0]: Vector input for softmax (4 elements)
    - data_out[7:0]: Output data as 8-bit unsigned integer [0-255]
    - valid_out: Output valid signal
    """
    
    def __init__(self):
        self.dut = cocotb.top
        self.drv_queue = Queue(maxsize=1)
        self.mtr_queue = Queue(maxsize=0)
        self.seq_item_queue = Queue(maxsize=0)
        self.result_ready_queue = Queue(maxsize=0)
        
        # Get matrix size from DUT parameter
        self.matrix_size = int(self.dut.ACT_MATRIX_SIZE.value)
        
    def activation_type_to_func_sel(self, activation_type):
        """
        Convert activation type string to func_sel encoding
        """
        mapping = {
            'relu': 0b00,
            'sigmoid': 0b01,
            'tanh': 0b10,
            'softmax': 0b11
        }
        return mapping.get(activation_type.lower(), 0b00)
    
    def flatten_matrix_to_array(self, matrix):
        """Convert a 2D matrix to a flat array for the DUT.
        Each element is 8 bits (signed)."""
        # Flatten the matrix in row-major order
        if isinstance(matrix, np.ndarray):
            flat = matrix.flatten().tolist()
        else:
            flat = [elem for row in matrix for elem in row]
        
        # Ensure each element is 8-bit signed
        result = []
        for elem in flat:
            val = int(elem) & 0xFF
            # Convert to signed: if > 127, subtract 256
            if val > 127:
                val -= 256
            result.append(val)
        return result

    def array_to_output_matrix(self, array, shape):
        """Convert a flat array from the DUT back to a 2D matrix.
        Each element is 8 bits (signed)."""
        # Extract elements
        flat = []
        for elem in array:
            # Handle cocotb BinaryValue/LogicArray
            if hasattr(elem, 'integer'):
                val = elem.integer
            elif hasattr(elem, 'value'):
                val = int(elem.value)
            else:
                val = int(elem)
            # Convert to signed 8-bit
            val = val & 0xFF
            if val > 127:
                val -= 256
            flat.append(val)
        
        # Reshape to 2D array as int8
        return np.array(flat, dtype=np.int8).reshape(shape)
    
    async def send_config(self, seq_item):
        """Send activation configuration and input data to driver"""
        if seq_item.layer_type == 'activation':
            await self.seq_item_queue.put(seq_item)
            # driver_tuple: (valid_in, input_data, config)
            driver_tuple = (1, seq_item.input_data, seq_item.config)
            await self.drv_queue.put(driver_tuple)

    async def get_result(self):
        """
        Get result from monitor and convert to comparable format
        Returns seq_item with actual_output in same format as golden model
        """
        result_data = await self.mtr_queue.get()
        seq_item = await self.seq_item_queue.get()
        
        # Convert DUT output to integer matrix for comparison
        input_shape = seq_item.config['input_shape']
        seq_item.actual_output = self.array_to_output_matrix(
            result_data, 
            shape=input_shape
        )
        
        # Notify waiting sequences that result is ready
        await self.result_ready_queue.put(seq_item.actual_output)
        return seq_item

    async def wait_for_result(self):
        """Wait for the monitor to capture the actual DUT output and return it"""
        result = await self.result_ready_queue.get()
        return result
    
    async def reset(self):
        """Reset the DUT to initial state. Skip if activation signals don't exist."""
        try:
            await RisingEdge(self.dut.clk)
            # Initialize all inputs during reset
            self.dut.rst_n.value = 0
            self.dut.valid_in.value = 0
            self.dut.func_sel.value = 0  # Default to ReLU
            self.dut.act_matrix_size.value = 0  # Initialize matrix size
            
            # Initialize all data_in array elements
            for i in range(self.matrix_size):
                self.dut.data_in[i].value = 0
            
            await RisingEdge(self.dut.clk)
            self.dut.rst_n.value = 1
            await RisingEdge(self.dut.clk)
        except (AttributeError, TypeError, IndexError):
            # Activation signals don't exist (pooling DUT loaded)
            pass

    async def driver_bfm(self):
        """
        Driver BFM - sends full matrix to DUT at once
        DUT processes all elements in parallel
        """
        # Initialize with default values
        self.dut.rst_n.value = 1
        self.dut.valid_in.value = 0
        self.dut.func_sel.value = 0
        self.dut.act_matrix_size.value = 0
        
        # Initialize all data_in array elements
        for i in range(self.matrix_size):
            self.dut.data_in[i].value = 0
        
        while True:
            await RisingEdge(self.dut.clk)
            try:
                (valid_in, data_in, config) = self.drv_queue.get_nowait()
                
                # Set activation function select
                activation_type = config['activation_type']
                self.dut.func_sel.value = self.activation_type_to_func_sel(activation_type)
                
                # Convert input matrix to flat array
                flat_array = self.flatten_matrix_to_array(data_in)
                actual_size = len(flat_array)
                
                # Drive the actual matrix size to the DUT
                self.dut.act_matrix_size.value = actual_size
                
                # Send all elements to DUT at once
                for i in range(min(actual_size, self.matrix_size)):
                    self.dut.data_in[i].value = flat_array[i]
                
                # Pad remaining elements with zeros if matrix is smaller
                for i in range(actual_size, self.matrix_size):
                    self.dut.data_in[i].value = 0
                
                self.dut.valid_in.value = valid_in
                
            except QueueEmpty:
                # Deassert valid_in when no new transaction
                self.dut.valid_in.value = 0
            
    async def monitor_bfm(self):
        """
        Monitor BFM - captures matrix output from DUT based on act_matrix_size
        """
        while True:
            await RisingEdge(self.dut.clk)
            
            if self.dut.valid_out.value == 1:
                # Read the actual matrix size driven to the DUT
                actual_size = int(self.dut.act_matrix_size.value)
                actual_size = min(actual_size, self.matrix_size)
                
                # Capture only the active output data elements
                output_data = []
                for i in range(actual_size):
                    output_data.append(int(self.dut.data_out[i].value))
                
                # Put captured data in monitor queue
                await self.mtr_queue.put(output_data)
    
    def start_bfm(self):
        """Start BFM only if activation signals exist"""
        try:
            # Check if activation-specific signal exists
            _ = self.dut.func_sel
            cocotb.start_soon(self.driver_bfm())
            cocotb.start_soon(self.monitor_bfm())
        except AttributeError:
            # Activation signals don't exist (pooling DUT loaded)
            pass


# # Utility functions for data format conversion and comparison
# def compare_outputs(golden, dut_actual, tolerance=0):
#     """
#     Compare golden model output with DUT actual output
    
#     Args:
#         golden: numpy array from strategy compute_golden() (uint8)
#         dut_actual: numpy array from BFM (uint8)
#         tolerance: acceptable difference (default 0 for exact match)
    
#     Returns:
#         (match, max_error) tuple
#     """
#     if golden.shape != dut_actual.shape:
#         logger.error(f"Shape mismatch: golden={golden.shape}, dut={dut_actual.shape}")
#         return False, float('inf')
    
#     # Compute element-wise error
#     error = np.abs(golden.astype(np.int16) - dut_actual.astype(np.int16))
#     max_error = np.max(error)
    
#     # Check if all errors are within tolerance
#     match = np.all(error <= tolerance)
    
#     if not match:
#         logger.error(f"Output mismatch! Max error: {max_error}")
#         logger.error(f"Golden:\n{golden}")
#         logger.error(f"DUT Actual:\n{dut_actual}")
#         logger.error(f"Error:\n{error}")
    
#     return match, max_error



