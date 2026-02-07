from LayerStrategy import LayerStrategy
import torch
import torch.nn as nn
import numpy as np

class ConvolutionStrategy(LayerStrategy):
    """Strategy for convolution layers"""
    
    def __init__(self):
        pass
    
    def get_layer_type(self):
        return "convolution"
    
    def generate_input_data(self, config):
        """
        Generate input data for convolution layer
        
        config = {
            'input_channels': 3,
            'output_channels': 16,
            'kernel_size': 3,
            'stride': 1,
            'padding': 0,
            'input_shape': (28, 28),  # Height, Width
            'data_range': (0, 255),
            'weight_range': (-1, 1)   # For kernel weights
        }
        """
        h, w = config['input_shape']
        channels = config.get('input_channels', 1)
        low, high = config['data_range']
        
        # Generate random input: (channels, height, width)
        if channels > 1:
            return np.random.randint(low, high, size=(channels, h, w))
        else:
            return np.random.randint(low, high, size=(h, w))
    
    def generate_kernel_weights(self, config):
        """Generate kernel weights for convolution"""
        kernel_size = config['kernel_size']
        input_channels = config.get('input_channels', 1)
        output_channels = config.get('output_channels', 1)
        low, high = config.get('weight_range', (-1, 1))
        
        # Generate random kernel weights
        # Shape: (output_channels, input_channels, kernel_size, kernel_size)
        if input_channels > 1 and output_channels > 1:
            kernel = np.random.uniform(low, high, 
                                     size=(output_channels, input_channels, 
                                           kernel_size, kernel_size))
        elif input_channels > 1:
            kernel = np.random.uniform(low, high,
                                     size=(input_channels, kernel_size, kernel_size))
        else:
            kernel = np.random.uniform(low, high,
                                     size=(kernel_size, kernel_size))
        
        # Quantize to 8-bit if needed
        if config.get('quantize_weights', False):
            scale = 127.0 / max(abs(low), abs(high))
            kernel_q = np.clip(kernel * scale, -128, 127).astype(np.int8)
            return kernel_q
        
        return kernel
    
    def compute_golden(self, input_data, config, kernel_weights=None):
        """
        Golden model for convolution that matches hardware behavior.
        Args:
            input_data: Input tensor (H, W) or (C, H, W)
            config: Configuration dictionary
            kernel_weights: Pre-generated kernel weights (optional)
                           If None, will generate new weights
        """
        kernel_size = config['kernel_size']
        stride = config.get('stride', 1)
        padding = config.get('padding', 0)
        input_channels = config.get('input_channels', 1)
        output_channels = config.get('output_channels', 1)
        
        # Use provided kernel weights or generate new ones
        if kernel_weights is None:
            kernel = self.generate_kernel_weights(config)
        else:
            kernel = kernel_weights

        if kernel.dtype == np.int8:
            scale = 127.0 / max(abs(config.get('weight_range', (-1, 1))[0]), abs(config.get('weight_range', (-1, 1))[1]))
            kernel = kernel.astype(np.float32) / scale
        
        # Convert to PyTorch tensors
        if input_data.ndim == 2:  # Single channel
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        elif input_data.ndim == 3:  # Multiple channels
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported input dimension: {input_data.ndim}")
        
        # Handle kernel tensor conversion based on dimensions
        if kernel.ndim == 2:  # Single in/out channel
            weight_tensor = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        elif kernel.ndim == 3:  # Multiple in, single out
            weight_tensor = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0)
        elif kernel.ndim == 4:  # Multiple in/out channels
            weight_tensor = torch.tensor(kernel, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported kernel dimension: {kernel.ndim}")
        
        # Apply convolution
        with torch.no_grad():
            conv = nn.Conv2d(
                in_channels=input_tensor.shape[1],
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            )
            
            # Set the weights
            conv.weight.data = weight_tensor
            
            # Perform convolution
            output = conv(input_tensor)
            
            # Apply activation if specified
            activation_type = config.get('activation', 'none')
            if activation_type == 'relu':
                output = torch.relu(output)
            elif activation_type == 'sigmoid':
                # Simplified sigmoid: clamp to [0, 255] range
                output = torch.sigmoid(output) * 255.0
            
            # Quantize to 8-bit if needed
            if config.get('quantize_output', True):
                # Match Verilog integer division (truncation towards zero)
                # Verilog: scaled_sum = conv_sum / WEIGHT_SCALE (integer division)
                output = torch.trunc(output)  # Truncate instead of floor
                
                # Clamp to valid range (unsigned 8-bit: [0, 255])
                output = torch.clamp(output, 0, 255)
        
        # Convert back to numpy
        result = output.squeeze(0).numpy()
        
        # Handle different output dimensions
        if result.ndim == 2:
            # Single channel output: (H, W)
            return result.astype(int)
        elif result.ndim == 3 and result.shape[0] == 1:
            # Single channel output: (1, H, W) -> squeeze to (H, W)
            return result.squeeze(0).astype(int)
        elif result.ndim == 3:
            # Multi-channel output: (C, H, W)
            return result.astype(int)
        else:
            # Edge case: reshape if needed
            return result.reshape(1, 1).astype(int)