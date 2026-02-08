from LayerStrategy import LayerStrategy
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
        
        Hardware ONLY supports quantized weights using INTEGER arithmetic:
        1. Multiply input (uint8) * weight (int8) -> produces int32
        2. Accumulate all products as int32
        3. Divide accumulated sum by WEIGHT_SCALE (127) using integer division
        4. Clamp to [0, 255]
        
        Args:
            input_data: Input tensor (H, W) or (C, H, W)
            config: Configuration dictionary
            kernel_weights: Pre-generated kernel weights (optional)
        """
        # Use provided kernel weights or generate new ones
        if kernel_weights is None:
            kernel = self.generate_kernel_weights(config)
        else:
            kernel = kernel_weights
        
        # Hardware ONLY supports quantized int8 weights
        if kernel.dtype != np.int8:
            raise ValueError(
                f"Hardware only supports quantized int8 weights, but got {kernel.dtype}. "
                "Set 'quantize_weights': true in your config."
            )
        
        return self._compute_golden_integer_arithmetic(
            input_data, kernel, config
        )
    
    
    def _compute_golden_integer_arithmetic(self, input_data, kernel_int8, config):
        """
        Integer arithmetic path that matches Verilog hardware exactly.
        This manually implements convolution using only integer operations.
        """
        kernel_size = config['kernel_size']
        stride = config.get('stride', 1)
        padding = config.get('padding', 0)
        input_channels = config.get('input_channels', 1)
        output_channels = config.get('output_channels', 1)
        WEIGHT_SCALE = 127
        
        # Ensure input is proper shape
        if input_data.ndim == 2:
            input_data = input_data.reshape(1, *input_data.shape)
        
        # Get input dimensions
        in_c, in_h, in_w = (input_data.shape[0], *input_data.shape[1:]) if input_data.ndim == 3 else (1, *input_data.shape)
        
        # Calculate output dimensions
        out_h = (in_h + 2 * padding - kernel_size) // stride + 1
        out_w = (in_w + 2 * padding - kernel_size) // stride + 1
        
        # Initialize output
        if output_channels > 1:
            output = np.zeros((output_channels, out_h, out_w), dtype=np.int32)
        else:
            output = np.zeros((out_h, out_w), dtype=np.int32)
        
        # Ensure kernel is proper shape
        if kernel_int8.ndim == 2:
            kernel_int8 = kernel_int8.reshape(1, 1, *kernel_int8.shape)
        elif kernel_int8.ndim == 3:
            kernel_int8 = kernel_int8.reshape(1, *kernel_int8.shape)
        
        # Perform convolution with INTEGER arithmetic (matching Verilog)
        for oc in range(output_channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    conv_sum = 0  # Integer accumulator
                    
                    for ic in range(input_channels):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                # Calculate input position
                                ih = oh * stride + kh - padding
                                iw = ow * stride + kw - padding
                                
                                # Check bounds (padding)
                                if 0 <= ih < in_h and 0 <= iw < in_w:
                                    # Get input value (unsigned int8)
                                    input_val = int(input_data[ic, ih, iw])
                                    
                                    # Get weight value (signed int8)
                                    weight_val = int(kernel_int8[oc, ic, kh, kw])
                                    
                                    # INTEGER multiply and accumulate
                                    product = input_val * weight_val
                                    conv_sum += product
                    
                    # Truncation toward zero
                    scaled_value = conv_sum // WEIGHT_SCALE
                    
                    # Clamp to [0, 255]
                    if scaled_value > 255:
                        scaled_value = 255
                    elif scaled_value < 0:
                        scaled_value = 0
                    
                    # Store result
                    if output_channels > 1:
                        output[oc, oh, ow] = scaled_value
                    else:
                        output[oh, ow] = scaled_value
        
        # Return as int (matching hardware unsigned 8-bit)
        return output.astype(np.int32)