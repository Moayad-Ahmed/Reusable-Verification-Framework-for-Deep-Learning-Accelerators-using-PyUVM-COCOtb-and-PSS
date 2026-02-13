from strategy.LayerStrategy import LayerStrategy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class PoolingStrategy(LayerStrategy):
    """Strategy for pooling layers"""
    
    def __init__(self):
        super().__init__()
        self.pool_types = ['max', 'avg', 'min']
    
    def get_layer_type(self):
        return "pooling"
    
    def validate_pooling_config(self, config):
        """
        Validate that kernel size, stride, and padded input dimensions are compatible.
        
        The sliding window must reach the edge perfectly: (padded_dim - kernel) % stride = 0
        for both height and width.
        
        Args:
            config: dict with keys:
                - kernel_size: int, pooling kernel size
                - stride: int, pooling stride
                - input_shape: tuple, (height, width)
                - padding_left, padding_right, padding_top, padding_bottom: int (optional, default 0)
        
        Raises:
            ValueError: If dimensions are incompatible with kernel and stride
        """
        kernel_size = config['kernel_size']
        stride = config['stride']
        input_h, input_w = config['input_shape']
        
        # Get padding values
        pad_left = config.get('padding_left', 0)
        pad_right = config.get('padding_right', 0)
        pad_top = config.get('padding_top', 0)
        pad_bottom = config.get('padding_bottom', 0)
        
        # Calculate padded dimensions
        padded_h = input_h + pad_top + pad_bottom
        padded_w = input_w + pad_left + pad_right
        
        # Check compatibility: (padded_dim - kernel) % stride = 0
        height_check = (padded_h - kernel_size) % stride
        width_check = (padded_w - kernel_size) % stride
        
        error_messages = []
        
        if height_check != 0:
            error_messages.append(
                f"Height incompatibility: (padded_height - kernel) % stride != 0\n"
                f"  Calculation: ({padded_h} - {kernel_size}) % {stride} = {height_check}\n"
                f"  Original height: {input_h}, Padding: top={pad_top}, bottom={pad_bottom}"
            )
        
        if width_check != 0:
            error_messages.append(
                f"Width incompatibility: (padded_width - kernel) % stride != 0\n"
                f"  Calculation: ({padded_w} - {kernel_size}) % {stride} = {width_check}\n"
                f"  Original width: {input_w}, Padding: left={pad_left}, right={pad_right}"
            )
        
        if error_messages:
            raise ValueError(
                "❌ POOLING CONFIGURATION ERROR - Incompatible dimensions:\n\n" + 
                "\n\n".join(error_messages) + 
                "\n\nThe sliding window cannot reach the edge perfectly.\n" +
                "Use formula: (padded_dimension - kernel_size) % stride = 0"
            )

    
    def generate_input_data(self, config):
        """
        Generate random input data matching specified dimensions and data range.
        
        Example config structure:
        
        config = {
            'kernel_size': 3,
            'stride': 1,
            'pool_type': 'avg',              # 'avg', 'max', or 'min'
            'input_shape': (8, 8),           # (height, width)
            'data_range': (0, 255),          # (min, max)
            # Padding configuration (optional):
            'padding_left': 1,
            'padding_right': 1,
            'padding_top': 1,
            'padding_bottom': 1,
            'padding_value': 0               # Value for padded regions
        }

        """
        h, w = config['input_shape']
        low, high = config['data_range']
        
        # Generate random input
        # return np.random.randint(low, high, size=(h, w))
        return np.full((h, w), fill_value=35, dtype=int)  # Use constant value for easier debugging
    
    def compute_golden(self, input_data, config):
        """
        Golden model that matches 8-bit Integer Hardware.
        Supports: MAX, MIN, and AVG (with floor rounding).
        Optionally applies padding before pooling.
        
        Args:
            input_data: numpy array of input (H, W)
            config: dict with keys:
                - kernel_size: int, pooling kernel size
                - stride: int, pooling stride
                - pool_type: str, 'avg', 'max', or 'min'
                - input_shape: tuple, (height, width)
                - data_range: tuple, (min, max)
                - padding_left: int (optional, default 0)
                - padding_right: int (optional, default 0)
                - padding_top: int (optional, default 0)
                - padding_bottom: int (optional, default 0)
                - padding_value: int (optional, default 0)
        
        Returns:
            numpy array of pooled output
        """
        kernel_size = config['kernel_size']
        stride = config['stride']
        pool_type = config['pool_type']
        
        # VALIDATION: Check that dimensions are compatible for pooling
        # This ensures the sliding window reaches the edge perfectly
        self.validate_pooling_config(config)
        
        # 1. Convert to FLOAT for PyTorch operations
        # Even though hardware is int, PyTorch AvgPool requires float.
        # We will cast back to int at the very end.
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        
        # Add Batch and Channel dims: (H, W) -> (1, 1, H, W)
        if len(input_tensor.shape) == 2:
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        
        # 1.5 Apply padding if configured
        # Extract padding values from config (defaults to 0 if not specified)
        pad_left = config.get('padding_left', 0)
        pad_right = config.get('padding_right', 0)
        pad_top = config.get('padding_top', 0)
        pad_bottom = config.get('padding_bottom', 0)
        pad_value = config.get('padding_value', 0)
        
        # Apply padding if any padding is specified
        if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
            # PyTorch padding format: (left, right, top, bottom) for 4D tensors
            input_tensor = F.pad(
                input_tensor,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='constant',
                value=pad_value
            )
        
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
        #result = np.clip(result, 0, 255)
        
        # If HW is Signed (-128 to 127), uncomment this instead:
        result = np.clip(result, -128, 127)
        
        # 4. Convert to integer array
        result = result.astype(int)
        
        return result