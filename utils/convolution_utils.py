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

class ConvolutionBFM(metaclass=utility_classes.Singleton):
    def __init__(self):
        self.dut = cocotb.top
        self.drv_queue = Queue(maxsize=1)
        self.mtr_queue = Queue(maxsize=0)
        self.seq_item_queue = Queue(maxsize=0)
        self.result_ready_queue = Queue(maxsize=0)  # Queue for sequence to wait on

    async def send_config(self, seq_item):
        await self.seq_item_queue.put(seq_item)
        driver_tuple = (1, seq_item.input_data, seq_item.kernel_weights, seq_item.config)
        await self.drv_queue.put(driver_tuple)

    async def get_result(self):
        result = await self.mtr_queue.get()
        seq_item = await self.seq_item_queue.get()
        
        # Calculate output dimensions
        kernel_size = seq_item.config['kernel_size']
        stride = seq_item.config.get('stride', 1)
        padding = seq_item.config.get('padding', 0)
        img_h = seq_item.config['input_shape'][0]
        img_w = seq_item.config['input_shape'][1]
        out_channels = seq_item.config.get('output_channels', 1)
        
        output_height = floor((img_h + 2*padding - kernel_size) / stride) + 1
        output_width = floor((img_w + 2*padding - kernel_size) / stride) + 1
        
        # Convert result based on output dimensions
        if out_channels > 1:
            shape = (out_channels, output_height, output_width)
        else:
            shape = (output_height, output_width)
            
        seq_item.actual_output = self.int_to_output_tensor(result, shape)
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
        self.dut.rst_n.value = 0
        self.dut.conv_valid_in.value = 0
        self.dut.conv_data_in.value = 0
        self.dut.conv_kernel_weights.value = 0
        self.dut.conv_kernel_size.value = 3
        self.dut.conv_stride.value = 1
        self.dut.conv_padding.value = 0
        self.dut.conv_img_height.value = 32
        self.dut.conv_img_width.value = 32
        self.dut.conv_in_channels.value = 1
        self.dut.conv_out_channels.value = 1
        self.dut.conv_activation.value = 0

        # Reset Pooling BFM inputs as well
        self.dut.pool_valid_in.value = 0
        self.dut.pool_data_in.value = 0
        self.dut.pool_mode.value = 0
        self.dut.pool_size.value = 2
        self.dut.pool_stride_h.value = 2
        self.dut.pool_stride_w.value = 2
        self.dut.pool_img_height.value = 32
        self.dut.pool_img_width.value = 32

        await RisingEdge(self.dut.clk)
        self.dut.rst_n.value = 1
        await RisingEdge(self.dut.clk)

    def flatten_tensor_to_int(self, tensor):
        """Convert a multi-dimensional tensor to a packed integer for the DUT.
        Each element is 8 bits, packed in C-H-W order (channel, height, width)."""
        if isinstance(tensor, np.ndarray):
            if tensor.ndim == 2 or tensor.ndim == 3:  # Single channel (H, W) or multi-channel (C, H, W)
                flat = tensor.flatten().tolist()
            else:
                raise ValueError(f"Unsupported tensor dimensions: {tensor.ndim}")
        else:
            flat = [elem for row in tensor for elem in row] if isinstance(tensor[0], list) else tensor
        
        # Pack elements into a single integer
        packed_value = 0
        for i, elem in enumerate(flat):
            packed_value |= (int(elem) & 0xFF) << (i * 8)
        return packed_value

    def flatten_kernel_to_int(self, kernel):
        """Convert kernel weights to packed integer.
        
        Kernel shape can be:
        - 2D: (K, K) - single in/out channel
        - 3D: (C_in, K, K) - multi input, single output
        - 4D: (C_out, C_in, K, K) - multi input/output
        
        Packed in order: [out_ch][in_ch][k_row][k_col]
        """
        if isinstance(kernel, np.ndarray):
            flat = kernel.flatten().tolist()
        else:
            # Flatten nested lists
            def flatten_recursive(lst):
                result = []
                for item in lst:
                    if isinstance(item, (list, np.ndarray)):
                        result.extend(flatten_recursive(item))
                    else:
                        result.append(item)
                return result
            flat = flatten_recursive(kernel)
        
        # Pack kernel weights
        packed_value = 0
        for i, weight in enumerate(flat):
            # Convert float weights to 8-bit signed integer
            weight_int = int(weight) & 0xFF
            packed_value |= weight_int << (i * 8)
        return packed_value

    def int_to_output_tensor(self, packed_value, shape):
        """Convert a packed integer from the DUT back to a tensor.
        
        Args:
            packed_value: Packed integer from DUT
            shape: Tuple defining output shape, e.g., (H, W) or (C, H, W)
        """
        # Convert to integer if it's a cocotb BinaryValue/LogicArray
        if hasattr(packed_value, 'to_unsigned'):
            packed_value = packed_value.to_unsigned()
        else:
            packed_value = int(packed_value)
        
        # Calculate total number of elements
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        
        # Extract elements from packed integer
        flat = []
        for i in range(num_elements):
            elem = (packed_value >> (i * 8)) & 0xFF
            flat.append(elem)
        
        # Reshape to target shape
        return np.array(flat).reshape(shape)

    async def driver_bfm(self):
        # Initialize with default values
        self.dut.rst_n.value = 1
        self.dut.conv_valid_in.value = 0
        self.dut.conv_data_in.value = 0
        self.dut.conv_kernel_weights.value = 0
        self.dut.conv_kernel_size.value = 3
        self.dut.conv_stride.value = 1
        self.dut.conv_padding.value = 0
        self.dut.conv_img_height.value = 32
        self.dut.conv_img_width.value = 32
        self.dut.conv_in_channels.value = 1
        self.dut.conv_out_channels.value = 1
        self.dut.conv_activation.value = 0

        # Initialize Pooling BFM inputs as well
        self.dut.pool_valid_in.value = 0
        self.dut.pool_data_in.value = 0
        self.dut.pool_mode.value = 0
        self.dut.pool_size.value = 2
        self.dut.pool_stride_h.value = 2
        self.dut.pool_stride_w.value = 2
        self.dut.pool_img_height.value = 32
        self.dut.pool_img_width.value = 32
        
        while True:
            await RisingEdge(self.dut.clk)
            try:
                (valid_in, data_in, kernel_weights, config) = self.drv_queue.get_nowait()
                self.dut.conv_valid_in.value = valid_in
                
                # Convert input data to packed integer
                packed_data = self.flatten_tensor_to_int(data_in)
                self.dut.conv_data_in.value = packed_data
                
                # Convert kernel weights to packed integer
                packed_kernel = self.flatten_kernel_to_int(kernel_weights)
                self.dut.conv_kernel_weights.value = packed_kernel
                
                # Set the configurable parameters from config
                self.dut.conv_kernel_size.value = config['kernel_size']
                self.dut.conv_stride.value = config.get('stride', 1)
                self.dut.conv_padding.value = config.get('padding', 0)
                self.dut.conv_img_height.value = config['input_shape'][0]
                self.dut.conv_img_width.value = config['input_shape'][1]
                self.dut.conv_in_channels.value = config.get('input_channels', 1)
                self.dut.conv_out_channels.value = config.get('output_channels', 1)
                
                # Set activation function
                activation_map = {'none': 0, 'relu': 1, 'sigmoid': 2}
                activation = config.get('activation', 'none')
                self.dut.conv_activation.value = activation_map.get(activation, 0)
                
            except QueueEmpty:
                # Deassert valid_in when no new transaction
                self.dut.conv_valid_in.value = 0
            
    async def monitor_bfm(self):
        while True:
            await RisingEdge(self.dut.clk)
            if self.dut.conv_valid_out.value == 1:
                result = (self.dut.conv_data_out.value)
                await self.mtr_queue.put(result)
    
    def start_bfm(self):
        cocotb.start_soon(self.driver_bfm())
        cocotb.start_soon(self.monitor_bfm())
