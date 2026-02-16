"""
Convolution-layer strategy for NVDLA nv_small (INT8 Direct-Convolution).

Golden-model pipeline:
    1. result = conv2d(input, weight)            — full-precision INT32 sum
    2. result = result >> clip_truncate           — CACC right-shift
    3. result = clamp(result, -128, 127)          — saturate to INT8

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

        Returns:
            list[int]  — flat unsigned-byte list, padded to ≥128 bytes.
        """
        K, C, KH, KW = weight_data.shape
        AC, AK = self.ATOM_C, self.ATOM_K
        num_kg = (K  + AK - 1) // AK
        num_cg = (C  + AC - 1) // AC

        buf = []
        for kg in range(num_kg):
            for s in range(KH):
                for r in range(KW):
                    for cg in range(num_cg):
                        for ki in range(AK):
                            for ci in range(AC):
                                k_idx = kg * AK + ki
                                c_idx = cg * AC + ci
                                if k_idx < K and c_idx < C:
                                    buf.append(int(weight_data[k_idx, c_idx, s, r]) & 0xFF)
                                else:
                                    buf.append(0)

        # Pad to at least 128 bytes (DRAM alignment)
        target = max(128, ((len(buf) + 127) // 128) * 128)
        buf.extend([0] * (target - len(buf)))
        return buf

    # ------------------------------------------------------------------ #
    #  Golden model                                                       #
    # ------------------------------------------------------------------ #
    def compute_golden(self, input_data, config, weight_data=None):
        """
        INT8 direct-convolution golden model.

        Args:
            input_data  : np.int8 array (C, H, W)
            config      : layer configuration dict
            weight_data : np.int8 array (K, C, KH, KW)

        Returns:
            np.int8 array (K, out_H, out_W)
        """
        if weight_data is None:
            raise ValueError("weight_data is required for convolution golden model")

        clip_truncate = config.get('clip_truncate', 0)
        stride_h = config.get('stride_h', 1)
        stride_w = config.get('stride_w', 1)
        pad_l    = config.get('padding_left', 0)
        pad_r    = config.get('padding_right', 0)
        pad_t    = config.get('padding_top', 0)
        pad_b    = config.get('padding_bottom', 0)
        pad_val  = config.get('padding_value', 0)
        dil_x    = config.get('dilation_x', 1)
        dil_y    = config.get('dilation_y', 1)

        C, H, W         = input_data.shape
        K, _, KH, KW    = weight_data.shape

        # Effective kernel footprint with dilation
        eff_kh = (KH - 1) * dil_y + 1
        eff_kw = (KW - 1) * dil_x + 1

        # Pad input (INT32 to avoid overflow)
        padded_h = H + pad_t + pad_b
        padded_w = W + pad_l + pad_r
        padded = np.full((C, padded_h, padded_w), pad_val, dtype=np.int32)
        padded[:, pad_t:pad_t + H, pad_l:pad_l + W] = input_data.astype(np.int32)

        out_h = (padded_h - eff_kh) // stride_h + 1
        out_w = (padded_w - eff_kw) // stride_w + 1

        output = np.zeros((K, out_h, out_w), dtype=np.int64)

        for k in range(K):
            for oh in range(out_h):
                for ow in range(out_w):
                    acc = np.int64(0)
                    for kh in range(KH):
                        for kw_i in range(KW):
                            ih = oh * stride_h + kh * dil_y
                            iw = ow * stride_w + kw_i * dil_x
                            for c in range(C):
                                acc += np.int64(padded[c, ih, iw]) * np.int64(weight_data[k, c, kh, kw_i])
                    output[k, oh, ow] = acc

        # CACC truncation (arithmetic right shift)
        if clip_truncate > 0:
            output = output >> clip_truncate

        # Saturate to INT8
        output = np.clip(output, -128, 127).astype(np.int8)
        return output
