"""
Convolution-layer strategy for NVDLA nv_small (INT8 Direct-Convolution).

Golden-model pipeline:
    1. result = conv2d(input, weight)            -- full-precision INT32 sum
    2. result = round_half_up(result, truncate)  -- CACC truncation with rounding
    3. result = clamp(result, -128, 127)         -- saturate to INT8

Weight memory format (nv_small, uncompressed, DC):
    for each kernel_group [0 .. ceil(K/atomK)):
      for s in [0..S):
        for r in [0..R):
          for cg in [0..ceil(C/atomC)):
            for k in [0..atomK):       # kernel within group
              for c in [0..atomC):     # channel within group
                weight[kg*atomK+k][s][r][cg*atomC+c]
    Padding with zeros for non-existent kernels/channels.
"""

from strategy.LayerStrategy import LayerStrategy
import numpy as np


class ConvolutionStrategy(LayerStrategy):
    """Strategy for NVDLA nv_small INT8 direct-convolution layers."""

    # nv_small atomic parameters
    ATOM_C = 8
    ATOM_K = 8

    def get_layer_type(self):
        return "convolution"

    # ------------------------------------------------------------------ #
    #  Validation                                                         #
    # ------------------------------------------------------------------ #
    def validate_conv_config(self, config):
        """Raise ValueError when mandatory keys are missing or out-of-range."""
        required = ['input_shape', 'num_channels', 'num_kernels']
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required conv config key: {key}")

        if config.get('data_format', 'INT8') != 'INT8':
            raise ValueError("nv_small convolution supports INT8 only")

    # ------------------------------------------------------------------ #
    #  Input-data generation                                              #
    # ------------------------------------------------------------------ #
    def generate_input_data(self, config):
        """
        Return a numpy INT8 array in CHW order.

        Config keys used:
            input_shape : [H, W, C] or [H, W]
            data_range  : [min, max]  (default [-5, 5])
        """
        input_shape = config['input_shape']
        data_range  = config.get('data_range', [-5, 5])

        if len(input_shape) == 3:
            h, w, c = input_shape
        else:
            h, w = input_shape
            c = config.get('num_channels', 1)

        return np.random.randint(
            data_range[0], data_range[1] + 1, size=(c, h, w)
        ).astype(np.int8)

    # ------------------------------------------------------------------ #
    #  Weight-data generation                                             #
    # ------------------------------------------------------------------ #
    def generate_weight_data(self, config):
        """
        Return a numpy INT8 array with shape (K, C, KH, KW).

        Config keys used:
            kernel_h, kernel_w, num_channels, num_kernels
            weight_range : [min, max]  (default [-3, 3])
        """
        kh = config.get('kernel_h', 1)
        kw = config.get('kernel_w', 1)
        c  = config['num_channels']
        k  = config['num_kernels']
        wr = config.get('weight_range', [-3, 3])

        return np.random.randint(
            wr[0], wr[1] + 1, size=(k, c, kh, kw)
        ).astype(np.int8)

    # ------------------------------------------------------------------ #
    #  NVDLA weight formatting                                            #
    # ------------------------------------------------------------------ #
    def format_weights_for_nvdla(self, weight_data):
        """
        Re-arrange weight_data (K, C, KH, KW) into the nv_small DRAM layout.
        Optimized: uses NumPy reshape/transpose instead of 6 nested loops.

        Returns:
            list[int]  — flat unsigned-byte list, padded to ≥128 bytes.
        """
        K, C, KH, KW = weight_data.shape
        AC, AK = self.ATOM_C, self.ATOM_K
        
        # Pad K and C to atom boundaries
        K_padded = ((K + AK - 1) // AK) * AK
        C_padded = ((C + AC - 1) // AC) * AC
        
        # Create padded weight array and copy data
        weight_padded = np.zeros((K_padded, C_padded, KH, KW), dtype=np.int8)
        weight_padded[:K, :C, :, :] = weight_data
        
        # Reshape to group structure: (nkg, AK, ncg, AC, KH, KW)
        weight_grouped = weight_padded.reshape(
            K_padded // AK, AK, C_padded // AC, AC, KH, KW
        )
        
        # Transpose to NVDLA memory order: (nkg, KH, KW, ncg, AK, AC)
        # From (kg, ki, cg, ci, s, r) to (kg, s, r, cg, ki, ci)
        weight_reordered = weight_grouped.transpose(0, 4, 5, 2, 1, 3)
        
        # Flatten to byte list and convert to unsigned
        buf = weight_reordered.astype(np.uint8).flatten().tolist()
        
        # Pad to at least 128 bytes (DRAM alignment)
        target = max(128, ((len(buf) + 127) // 128) * 128)
        buf.extend([0] * (target - len(buf)))
        return buf

    # ------------------------------------------------------------------ #
    #  Golden model                                                       #
    # ------------------------------------------------------------------ #
    def compute_golden(self, input_data, config, weight_data=None):
        """
        INT8 direct-convolution golden model (vectorized).
        
        Optimized: uses NumPy einsum for vectorized inner product computation.
        10-100x faster than nested-loop version for typical convolution sizes.

        Args:
            input_data  : np.int8 array (C, H, W)
            config      : layer configuration dict
            weight_data : np.int8 array (K, C, KH, KW)

        Returns:
            np.int8 array (K, out_H, out_W)
        """
        if weight_data is None:
            raise ValueError("weight_data is required for convolution golden model")

        # Extract configuration parameters
        clip_truncate = config.get('clip_truncate', 0)
        stride_h = config.get('stride_h', 1)
        stride_w = config.get('stride_w', 1)
        pad_t    = config.get('padding_top', 0)
        pad_b    = config.get('padding_bottom', 0)
        pad_l    = config.get('padding_left', 0)
        pad_r    = config.get('padding_right', 0)
        pad_val  = config.get('padding_value', 0)
        dil_y    = config.get('dilation_y', 1)
        dil_x    = config.get('dilation_x', 1)

        C, H, W         = input_data.shape
        K, _, KH, KW    = weight_data.shape

        # Effective kernel footprint with dilation
        eff_kh = (KH - 1) * dil_y + 1
        eff_kw = (KW - 1) * dil_x + 1

        # Pad input using np.pad (more efficient than manual padding)
        pad_width = ((0, 0), (pad_t, pad_b), (pad_l, pad_r))
        padded = np.pad(input_data.astype(np.int32), pad_width, 
                        mode='constant', constant_values=pad_val)

        padded_h, padded_w = padded.shape[1:]
        out_h = (padded_h - eff_kh) // stride_h + 1
        out_w = (padded_w - eff_kw) // stride_w + 1

        # Pre-convert weights to int64 once (instead of repeatedly in loops)
        weight_i64 = weight_data.astype(np.int64)
        output = np.zeros((K, out_h, out_w), dtype=np.int64)

        # Vectorized computation: use einsum for inner product
        # This eliminates the innermost 3 loops (c, kh, kw)
        for oh in range(out_h):
            for ow in range(out_w):
                h_start = oh * stride_h
                w_start = ow * stride_w
                
                # Extract dilated receptive field: (C, KH, KW)
                receptive_field = np.zeros((C, KH, KW), dtype=np.int32)
                for kh_idx in range(KH):
                    for kw_idx in range(KW):
                        receptive_field[:, kh_idx, kw_idx] = padded[:, 
                                                               h_start + kh_idx * dil_y,
                                                               w_start + kw_idx * dil_x]
                
                # Compute convolution for all kernels at this position using einsum
                # weight_i64: (K, C, KH, KW), receptive_field: (C, KH, KW)
                # einsum('kcij,cij->k') computes all kernel outputs in one shot
                output[:, oh, ow] = np.einsum('kcij,cij->k', weight_i64,
                                              receptive_field.astype(np.int64))

        # CACC truncation with round-half-up (matches NVDLA RTL behavior)
        # The hardware adds a rounding bias before shifting:
        #   guard  = bit at position (truncate-1)
        #   sticky = OR of bits below guard
        #   round_up = guard AND (positive OR sticky)
        if clip_truncate > 0:
            guard = (output >> (clip_truncate - 1)) & 1
            if clip_truncate > 1:
                sticky_mask = (1 << (clip_truncate - 1)) - 1
                sticky = (output & sticky_mask).astype(bool).astype(np.int64)
            else:
                sticky = np.zeros_like(output)
            is_positive = (output >= 0).astype(np.int64)
            round_up = guard & (is_positive | sticky)
            output = (output >> clip_truncate) + round_up

        # Saturate to INT8
        output = np.clip(output, -128, 127).astype(np.int8)
        return output
