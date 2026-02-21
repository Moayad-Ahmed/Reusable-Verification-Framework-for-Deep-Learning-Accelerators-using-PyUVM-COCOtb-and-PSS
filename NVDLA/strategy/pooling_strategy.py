from strategy.LayerStrategy import LayerStrategy
import torch
import torch.nn as nn
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
                - input_shape: tuple, (height, width) or (height, width, channels)
                - padding_left, padding_right, padding_top, padding_bottom: int (optional, default 0)
        
        Raises:
            ValueError: If dimensions are incompatible with kernel and stride
        """
        kernel_size = config['kernel_size']
        stride = config['stride']
        input_shape = config['input_shape']
        input_h = input_shape[0]
        input_w = input_shape[1]
        
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

        if padded_h < kernel_size:
            error_messages.append(
                f"Height too small: padded_height ({padded_h}) < kernel_size ({kernel_size})\n"
                f"  Original height: {input_h}, Padding: top={pad_top}, bottom={pad_bottom}"
            )

        if padded_w < kernel_size:
            error_messages.append(
                f"Width too small: padded_width ({padded_w}) < kernel_size ({kernel_size})\n"
                f"  Original width: {input_w}, Padding: left={pad_left}, right={pad_right}"
            )
        
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
                "POOLING CONFIGURATION ERROR - Incompatible dimensions:\n\n" + 
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
            'input_shape': (8, 8) or (8, 8, 3),  # (height, width) or (height, width, channels)
            'data_range': (0, 255),          # (min, max)
            # Padding configuration (optional):
            'padding_left': 1,
            'padding_right': 1,
            'padding_top': 1,
            'padding_bottom': 1,
            'padding_value': 0               # Value for padded regions
        }

        """
        input_shape = config['input_shape']
        low, high = config['data_range']
        
        # Support both [H, W] and [H, W, C] formats
        if len(input_shape) == 2:
            h, w = input_shape
            # Single channel: return 2D array (H, W)
            return np.random.randint(low, high, size=(h, w))
        else:
            h, w, c = input_shape
            # Multi-channel: return 3D array (C, H, W) - channel-first for PyTorch compatibility
            return np.random.randint(low, high, size=(c, h, w))
        #return np.full((h, w), fill_value=35, dtype=int)  # Use constant value for easier debugging
    
    # ------------------------------------------------------------------ #
    #                        GENERIC GOLDEN MODEL                         #
    # ------------------------------------------------------------------ #

    def compute_golden(self, input_data, config):
        """
        Golden model that matches 8-bit Integer Hardware.
        Supports: MAX, MIN, and AVG (with floor rounding).
        Optionally applies padding before pooling.
        Handles both 2D (H, W) and 3D (C, H, W) input data.
        """
        kernel_size = config['kernel_size']
        stride = config['stride']
        pool_type = config['pool_type']

        # VALIDATION: Check that dimensions are compatible for pooling
        self.validate_pooling_config(config)

        # 1. Optionally pad the input
        pad_left   = config.get('padding_left', 0)
        pad_right  = config.get('padding_right', 0)
        pad_top    = config.get('padding_top', 0)
        pad_bottom = config.get('padding_bottom', 0)
        pad_value  = config.get('padding_value', 0)

        # Determine if input is 2D (H, W) or 3D (C, H, W)
        is_3d = len(input_data.shape) == 3

        if pad_left or pad_right or pad_top or pad_bottom:
            if is_3d:
                # 3D input (C, H, W): pad only spatial dimensions
                padded = np.pad(
                    input_data,
                    ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant',
                    constant_values=pad_value,
                )
            else:
                # 2D input (H, W)
                padded = np.pad(
                    input_data,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant',
                    constant_values=pad_value,
                )
        else:
            padded = input_data

        # 2. Convert to FLOAT for PyTorch operations
        # Even though hardware is int, PyTorch AvgPool requires float.
        # We will cast back to int at the very end.
        input_tensor = torch.tensor(padded, dtype=torch.float32)

        # Add Batch dim and ensure Channel dim exists: (H, W) -> (1, 1, H, W) or (C, H, W) -> (1, C, H, W)
        if len(input_tensor.shape) == 2:
            # 2D: add batch and channel dims
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        elif len(input_tensor.shape) == 3:
            # 3D: add batch dim only
            input_tensor = input_tensor.unsqueeze(0)

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
                output = torch.round(output)

        # 3. Convert back to Numpy (remove batch dim, keep channel and spatial dims)
        result = output.squeeze(0).numpy()  # (C, H, W) or (1, H, W)

        # For single-channel output, squeeze channel dim to get 2D
        if result.shape[0] == 1 and not is_3d:
            result = result.squeeze(0)  # (H, W)
            # Ensure result is at least 2D (for 1x1 output cases)
            if result.ndim == 0:
                result = result.reshape(1, 1)
        # For multi-channel, keep as (C, H, W)

        # 4. Clip to signed INT8 range and convert to integer array
        result = np.clip(result, -128, 127).astype(int)

        return result

    # ------------------------------------------------------------------ #
    #              NVDLA HARDWARE-SPECIFIC AVG ADJUSTMENT                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _nvdla_hw_round(product_fixed):
        """
        Round-half-away-from-zero on a Q(N).16 fixed-point value.

        Replicates the three-way rounding mux in the NVDLA PDP RTL
        (NV_NVDLA_PDP_CORE_cal2d.v):
          - Positive / zero : integer_part + bit[15]  (round-half-up)
          - Negative, frac == -0.5 : round away from zero (toward -inf)
          - Negative, |frac| <  0.5 : round toward zero  (add 1)
          - Negative, |frac| >  0.5 : truncate (keep floor)
        """
        int_part = product_fixed >> 16        # arithmetic right-shift
        frac     = product_fixed & 0xFFFF     # 16-bit fractional part

        if product_fixed >= 0:
            return int_part + (1 if frac >= 0x8000 else 0)
        else:
            if   frac == 0x8000:  return int_part       # exactly -0.5 → away from zero
            elif frac >  0x8000:  return int_part + 1   # |frac| < 0.5 → toward zero
            else:                 return int_part       # |frac| > 0.5 → truncate

    @classmethod
    def nvdla_pool_adjust(cls, input_data, config):
        """
        Re-compute AVG pooling output using NVDLA's exact fixed-point
        arithmetic so the golden reference matches the hardware.

        NVDLA computes average pooling in two sequential stages:
          1. Multiply the window sum by ``recip_width``  (Q0.16), round.
          2. Multiply that intermediate by ``recip_height`` (Q0.16), round.

        Both stages use ``_nvdla_hw_round`` (round-half-away-from-zero).

                For MAX/MIN, NVDLA border behavior can differ from a generic
                zero-padded software model. We apply NVDLA-oriented padding
                defaults for border windows so expected data matches hardware:
                    - MAX: use a very small pad value (INT8 min by default)
                    - MIN: use a very large pad value (INT8 max by default)

        Args:
            input_data: the *original* (un-padded) numpy array (H, W)
                        so the function can recompute integer sums.
            config: same config dict used by ``compute_golden``.

        Returns:
            numpy int array (out_H, out_W) matching NVDLA hardware output.
        """
        pool_type = config['pool_type']

        # NVDLA border behavior for MAX/MIN (edge windows)
        if pool_type == 'max':
            hw_cfg = dict(config)
            hw_cfg['padding_value'] = config.get('nvdla_max_pad_value', -128)
            return cls().compute_golden(input_data, hw_cfg)
        if pool_type == 'min':
            hw_cfg = dict(config)
            hw_cfg['padding_value'] = config.get('nvdla_min_pad_value', 127)
            return cls().compute_golden(input_data, hw_cfg)

        kernel_size = config['kernel_size']
        stride      = config['stride']

        # Pad the input (same logic as compute_golden)
        pad_left   = config.get('padding_left', 0)
        pad_right  = config.get('padding_right', 0)
        pad_top    = config.get('padding_top', 0)
        pad_bottom = config.get('padding_bottom', 0)
        pad_value  = config.get('padding_value', 0)

        # Determine if input is 2D (H, W) or 3D (C, H, W)
        is_3d = len(input_data.shape) == 3

        if pad_left or pad_right or pad_top or pad_bottom:
            if is_3d:
                # 3D input (C, H, W): pad only spatial dimensions
                padded = np.pad(
                    input_data,
                    ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant',
                    constant_values=pad_value,
                )
            else:
                # 2D input (H, W)
                padded = np.pad(
                    input_data,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant',
                    constant_values=pad_value,
                )
        else:
            padded = input_data

        # Extract spatial dimensions
        if is_3d:
            num_channels, in_h, in_w = padded.shape
        else:
            in_h, in_w = padded.shape
            num_channels = 1
            
        out_h = (in_h - kernel_size) // stride + 1
        out_w = (in_w - kernel_size) // stride + 1

        # NVDLA reciprocal registers: round(65536 / kernel_dim)
        recip_w = round(65536 / kernel_size)
        recip_h = round(65536 / kernel_size)

        # Process each channel independently
        if is_3d:
            result = np.zeros((num_channels, out_h, out_w), dtype=int)
            for ch in range(num_channels):
                for r in range(out_h):
                    for c in range(out_w):
                        window = padded[ch, r * stride : r * stride + kernel_size,
                                        c * stride : c * stride + kernel_size]
                        s = int(np.sum(window))

                        # Stage 0 — divide by kernel width (fixed-point multiply + round)
                        step1 = cls._nvdla_hw_round(s * recip_w)
                        # Stage 1 — divide by kernel height
                        step2 = cls._nvdla_hw_round(step1 * recip_h)

                        result[ch, r, c] = step2
        else:
            result = np.zeros((out_h, out_w), dtype=int)
            for r in range(out_h):
                for c in range(out_w):
                    window = padded[r * stride : r * stride + kernel_size,
                                    c * stride : c * stride + kernel_size]
                    s = int(np.sum(window))

                    # Stage 0 — divide by kernel width (fixed-point multiply + round)
                    step1 = cls._nvdla_hw_round(s * recip_w)
                    # Stage 1 — divide by kernel height
                    step2 = cls._nvdla_hw_round(step1 * recip_h)

                    result[r, c] = step2

        result = np.clip(result, -128, 127).astype(int)
        if result.ndim == 0:
            result = result.reshape(1, 1)
        return result