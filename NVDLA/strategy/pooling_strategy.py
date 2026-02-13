from LayerStrategy import LayerStrategy
import torch
import torch.nn as nn
import numpy as np
import random

class PoolingStrategy(LayerStrategy):
    """Strategy for pooling layers"""
    
    def __init__(self):
        self.pool_types = ['max', 'avg', 'min']
    
    def get_layer_type(self):
        return "pooling"
    
    def generate_input_data(self, config):
        """
        Example ->

        config = {
            'kernel_size': 2,
            'stride': 1,
            'pool_type': 'max',
            'input_shape': (3, 3),
            'data_range': (0, 255)
        }

        """
        h, w = config['input_shape']
        low, high = config['data_range']
        
        # Generate random input
        return np.random.randint(low, high, size=(h, w))
    
    def compute_golden(self, input_data, config):
        """
        Golden model that matches 8-bit Integer Hardware.
        Supports: MAX, MIN, and AVG (with floor rounding).
        """
        kernel_size = config['kernel_size']
        stride = config['stride']
        pool_type = config['pool_type']
        
        # 1. Convert to FLOAT for PyTorch operations
        # Even though hardware is int, PyTorch AvgPool requires float.
        # We will cast back to int at the very end.
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        
        # Add Batch and Channel dims: (H, W) -> (1, 1, H, W)
        if len(input_tensor.shape) == 2:
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        
        output = None
        
        with torch.no_grad():
            if pool_type == 'max':
                # Standard Max Pooling
                pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
                output = pool(input_tensor)
                
            elif pool_type == 'min':
                # TRICK: PyTorch has no MinPool. 
                # Mathematical equivalent: min(x) = -max(-x)
                pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
                output = -pool(-input_tensor)
                
            elif pool_type == 'avg':
                # Apply AvgPool
                pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
                output = pool(input_tensor)
                
                # CRITICAL FOR HARDWARE MATCHING:
                # Hardware does integer division (e.g., 50 // 4 = 12).
                # PyTorch does float division (e.g., 50 / 4 = 12.5).
                # We must FLOOR (truncate) the result to match hardware.
                output = torch.floor(output)
        
        # 2. Convert back to Numpy (remove batch and channel dims, keep spatial dims)
        result = output.squeeze(0).squeeze(0).numpy()
        
        # Ensure result is at least 2D (for 1x1 output cases)
        if result.ndim == 0:
            result = result.reshape(1, 1)
        
        # 3. Clip to Valid 8-bit Range
        # If HW is Unsigned (0 to 255):
        result = np.clip(result, 0, 255)
        
        # If HW is Signed (-128 to 127), uncomment this instead:
        # result = np.clip(result, -128, 127)
        
        # 4. Convert to integer array
        result = result.astype(int)
        
        return result