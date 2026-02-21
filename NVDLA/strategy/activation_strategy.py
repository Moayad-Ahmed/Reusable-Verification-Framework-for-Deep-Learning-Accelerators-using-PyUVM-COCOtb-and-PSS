"""
Activation-layer strategy for NVDLA nv_small (INT8, standalone SDP).

Supported activations:
    - relu      : max(0, x)
    - prelu     : x if x >= 0 else x * leak_factor
    - sigmoid   : 1 / (1 + exp(-x))  via LUT
    - tanh      : tanh(x)            via LUT
    - clamp     : clip(x, min_val, max_val)

The SDP is used in NON-flying (standalone) mode — data comes from
DRAM via SDP_RDMA, passes through the SDP pipeline, and exits to
DRAM via WDMA.

Golden-model pipeline (must match hardware exactly):
    1. SDP receives INT8 input → internal 32-bit
    2. Passes through BS → BN → EW sub-processors
    3. Output CVT: out = saturate((data - offset) * scale >> shift)
    4. Result saturated to INT8 [-128, 127]
"""

from strategy.LayerStrategy import LayerStrategy
import numpy as np


class ActivationStrategy(LayerStrategy):
    """Strategy for NVDLA nv_small INT8 activation layers via standalone SDP."""

    ATOM = 8   # nv_small memory atomic size in bytes

    # ------------------------------------------------------------------ #
    #  Layer type                                                         #
    # ------------------------------------------------------------------ #
    def get_layer_type(self):
        return "activation"

    # ------------------------------------------------------------------ #
    #  Validation                                                         #
    # ------------------------------------------------------------------ #
    SUPPORTED_ACTIVATIONS = {'relu', 'prelu', 'sigmoid', 'tanh', 'clamp'}

    def validate_config(self, config):
        """Raise ValueError when mandatory keys are missing or invalid."""
        if 'activation_type' not in config:
            raise ValueError("Missing required key: 'activation_type'")
        act = config['activation_type']
        if act not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{act}'. "
                f"Choose from: {self.SUPPORTED_ACTIVATIONS}"
            )
        if 'input_shape' not in config:
            raise ValueError("Missing required key: 'input_shape'")
        if act == 'prelu' and 'leak_factor' not in config:
            raise ValueError("PReLU requires 'leak_factor' in config")
        if act == 'clamp':
            if 'clamp_min' not in config or 'clamp_max' not in config:
                raise ValueError("Clamp requires 'clamp_min' and 'clamp_max'")
        if config.get('data_format', 'INT8') != 'INT8':
            raise ValueError("nv_small SDP activation supports INT8 only")

    # ------------------------------------------------------------------ #
    #  Input-data generation                                              #
    # ------------------------------------------------------------------ #
    def generate_input_data(self, config):
        """
        Return a numpy INT8 array.

        Config keys used:
            input_shape : [H, W] or [H, W, C]
            data_range  : [min, max]  (default [-128, 127])
        """
        input_shape = config['input_shape']
        data_range  = config.get('data_range', [-128, 127])

        if len(input_shape) == 3:
            h, w, c = input_shape
        else:
            h, w = input_shape
            c = 1

        if c == 1:
            return np.random.randint(
                data_range[0], data_range[1] + 1, size=(h, w)
            ).astype(np.int8)
        else:
            # Return (C, H, W) for multi-channel — consistent with conv/pool
            return np.random.randint(
                data_range[0], data_range[1] + 1, size=(c, h, w)
            ).astype(np.int8)

    # ================================================================== #
    #                        GOLDEN MODELS                                #
    # ================================================================== #

    def compute_golden(self, input_data, config):
        """
        Compute expected activation output matching NVDLA hardware.

        Args:
            input_data : np.int8 array (H, W) or (C, H, W)
            config     : dict with activation parameters

        Returns:
            np.int8 array, same shape as input
        """
        self.validate_config(config)
        act = config['activation_type']

        if act == 'relu':
            return self._golden_relu(input_data, config)
        elif act == 'prelu':
            return self._golden_prelu(input_data, config)
        elif act == 'sigmoid':
            return self._golden_sigmoid(input_data, config)
        elif act == 'tanh':
            return self._golden_tanh(input_data, config)
        elif act == 'clamp':
            return self._golden_clamp(input_data, config)
        else:
            raise ValueError(f"Unsupported activation: {act}")

    # ------------------------------------------------------------------ #
    #  ReLU golden model                                                  #
    # ------------------------------------------------------------------ #
    def _golden_relu(self, input_data, config):
        """
        ReLU: max(0, x)

        Hardware path: BS stage with ALU bypass, MUL bypass, ReLU ON.
        Output CVT is identity (offset=0, scale=1, shift=0).
        """
        data = input_data.astype(np.int32)
        # ReLU: clamp negative to 0
        result = np.maximum(data, 0)
        # Output CVT identity → saturate to INT8
        return np.clip(result, -128, 127).astype(np.int8)

    # ------------------------------------------------------------------ #
    #  PReLU (Leaky ReLU) golden model                                    #
    # ------------------------------------------------------------------ #
    def _golden_prelu(self, input_data, config):
        """
        PReLU: x if x >= 0 else x * leak_factor

        Hardware path: BS stage MUL in PReLU mode.
        leak_factor is a float (e.g., 0.1) quantized to INT16 with a shift.

        The hardware computes:
            positive: pass through
            negative: x * operand >> shift_value

        The right-shift uses round-to-nearest (ties away from zero),
        matching the NV_NVDLA_HLS_shiftrightsu hardware module.
        """
        leak_float = config['leak_factor']
        mul_shift  = config.get('prelu_shift', 8)   # right-shift after multiply

        # Quantize leak factor to INT16 fixed-point
        leak_quant = int(round(leak_float * (1 << mul_shift)))
        leak_quant = np.clip(leak_quant, -32768, 32767)

        data = input_data.astype(np.int32)

        # For negative values: multiply then shift with HW-matching rounding
        products = data * leak_quant
        shifted = self._hw_shift_right(products, mul_shift)

        result = np.where(data >= 0, data, shifted)
        return np.clip(result, -128, 127).astype(np.int8)

    @staticmethod
    def _hw_shift_right(values, shift):
        """
        Replicate NV_NVDLA_HLS_shiftrightsu rounding behaviour.

        Round-to-nearest with ties away from zero:
          - Positive values: round half up   (0.5 → +1)
          - Negative values: round half down (−0.5 → −1), but
            fractions > 0.5 round toward zero (add 1 to floor).
        """
        if shift == 0:
            return values

        # Arithmetic right-shift (floor for negatives)
        integer_part = values >> shift
        remainder    = values - (integer_part << shift)

        # Decompose remainder into guard bit and sticky bits
        guard  = (remainder >> (shift - 1)) & 1
        if shift > 1:
            sticky = remainder & ((1 << (shift - 1)) - 1)
        else:
            sticky = np.zeros_like(remainder)

        # HW rounding rule:
        #   positive → point5 = guard
        #   negative → point5 = guard & (sticky != 0)
        is_neg = values < 0
        point5 = np.where(is_neg,
                          guard & (sticky != 0).astype(np.int32),
                          guard)

        return integer_part + point5

    # ------------------------------------------------------------------ #
    #  Sigmoid golden model (LUT-based)                                   #
    # ------------------------------------------------------------------ #
    def _golden_sigmoid(self, input_data, config):
        """
        Sigmoid: 1 / (1 + exp(-x))

        Hardware path: EW stage LUT.
        We replicate the NVDLA LUT linear interpolation exactly.

        For INT8 inputs, the internal 32-bit representation of x is just
        sign-extended x. The LUT covers a configurable input range.
        """
        lut_frac_bits = config.get('lut_frac_bits', 15)
        lut_range_min = config.get('lut_range_min', -8.0)
        lut_range_max = config.get('lut_range_max', 8.0)
        cvt_offset    = config.get('cvt_offset', 0)
        cvt_scale     = config.get('cvt_scale', 1)
        cvt_shift     = config.get('cvt_shift', 0)

        # Build the same LO table (257 entries) the hardware uses
        lo_table = self.generate_lo_lut_sigmoid(lut_frac_bits,
                                                 lut_range_min, lut_range_max)

        data = input_data.astype(np.float64)

        # Map input values to LUT index space
        # input range → [0, 256] linearly
        scale = 256.0 / (lut_range_max - lut_range_min)
        idx_float = (data - lut_range_min) * scale
        idx_float = np.clip(idx_float, 0.0, 256.0)

        # Integer index and fraction for interpolation
        idx_lo = np.floor(idx_float).astype(np.int32)
        idx_lo = np.clip(idx_lo, 0, 255)
        idx_hi = idx_lo + 1

        fraction = idx_float - idx_lo.astype(np.float64)

        # Table lookup + linear interpolation
        y0 = lo_table[idx_lo].astype(np.float64)
        y1 = lo_table[idx_hi].astype(np.float64)
        result = y0 + fraction * (y1 - y0)

        # Convert from LUT fixed-point to INT8 via output CVT.
        # Hardware CVT: out = shiftrightsu((data - offset) * scale, shift)
        # The same round-to-nearest rounding as the BS PReLU truncation.
        cvt_data = np.round((result - cvt_offset) * cvt_scale).astype(np.int64)
        if cvt_shift > 0:
            cvt_data = self._hw_shift_right(cvt_data, cvt_shift)

        return np.clip(cvt_data, -128, 127).astype(np.int8)

    # ------------------------------------------------------------------ #
    #  Tanh golden model (LUT-based)                                      #
    # ------------------------------------------------------------------ #
    def _golden_tanh(self, input_data, config):
        """
        Tanh: tanh(x)

        Same LUT mechanism as sigmoid, different table values and range.
        """
        lut_frac_bits = config.get('lut_frac_bits', 15)
        lut_range_min = config.get('lut_range_min', -4.0)
        lut_range_max = config.get('lut_range_max', 4.0)
        cvt_offset    = config.get('cvt_offset', 0)
        cvt_scale     = config.get('cvt_scale', 1)
        cvt_shift     = config.get('cvt_shift', 0)

        lo_table = self.generate_lo_lut_tanh(lut_frac_bits,
                                              lut_range_min, lut_range_max)

        data = input_data.astype(np.float64)
        scale = 256.0 / (lut_range_max - lut_range_min)
        idx_float = (data - lut_range_min) * scale
        idx_float = np.clip(idx_float, 0.0, 256.0)

        idx_lo = np.floor(idx_float).astype(np.int32)
        idx_lo = np.clip(idx_lo, 0, 255)
        idx_hi = idx_lo + 1

        fraction = idx_float - idx_lo.astype(np.float64)

        y0 = lo_table[idx_lo].astype(np.float64)
        y1 = lo_table[idx_hi].astype(np.float64)
        result = y0 + fraction * (y1 - y0)

        # Output CVT with hardware-matching round-to-nearest shift
        cvt_data = np.round((result - cvt_offset) * cvt_scale).astype(np.int64)
        if cvt_shift > 0:
            cvt_data = self._hw_shift_right(cvt_data, cvt_shift)

        return np.clip(cvt_data, -128, 127).astype(np.int8)

    # ------------------------------------------------------------------ #
    #  Clamp golden model                                                 #
    # ------------------------------------------------------------------ #
    def _golden_clamp(self, input_data, config):
        """
        Clamp: clip(x, min_val, max_val)

        Hardware path: BS stage ALU=MAX(min_val), BN stage ALU=MIN(max_val).
        """
        min_val = config['clamp_min']
        max_val = config['clamp_max']
        data = input_data.astype(np.int32)
        result = np.clip(data, min_val, max_val)
        return np.clip(result, -128, 127).astype(np.int8)

    # ================================================================== #
    #                        LUT TABLE GENERATORS                         #
    # ================================================================== #

    @staticmethod
    def generate_lo_lut_sigmoid(frac_bits=15, x_min=-8.0, x_max=8.0):
        """
        Generate 257-entry LO table for sigmoid function.
        Values are INT16 fixed-point with `frac_bits` fractional bits.

        Returns:
            np.int16 array of shape (257,)
        """
        x_vals = np.linspace(x_min, x_max, 257)
        sigmoid_vals = 1.0 / (1.0 + np.exp(-x_vals))
        scale = (1 << frac_bits)
        lut = np.clip(np.round(sigmoid_vals * scale), -32768, 32767).astype(np.int16)
        return lut

    @staticmethod
    def generate_lo_lut_tanh(frac_bits=15, x_min=-4.0, x_max=4.0):
        """
        Generate 257-entry LO table for tanh function.
        Values are INT16 fixed-point with `frac_bits` fractional bits.

        Returns:
            np.int16 array of shape (257,)
        """
        x_vals = np.linspace(x_min, x_max, 257)
        tanh_vals = np.tanh(x_vals)
        scale = (1 << frac_bits)
        lut = np.clip(np.round(tanh_vals * scale), -32768, 32767).astype(np.int16)
        return lut

    @staticmethod
    def generate_le_lut_sigmoid(frac_bits=15, x_min=-8.0, x_max=8.0):
        """
        Generate 65-entry LE table for sigmoid (linear spacing for simplicity).

        Returns:
            np.int16 array of shape (65,)
        """
        x_vals = np.linspace(x_min, x_max, 65)
        sigmoid_vals = 1.0 / (1.0 + np.exp(-x_vals))
        scale = (1 << frac_bits)
        lut = np.clip(np.round(sigmoid_vals * scale), -32768, 32767).astype(np.int16)
        return lut

    @staticmethod
    def generate_le_lut_tanh(frac_bits=15, x_min=-4.0, x_max=4.0):
        """
        Generate 65-entry LE table for tanh (linear spacing for simplicity).

        Returns:
            np.int16 array of shape (65,)
        """
        x_vals = np.linspace(x_min, x_max, 65)
        tanh_vals = np.tanh(x_vals)
        scale = (1 << frac_bits)
        lut = np.clip(np.round(tanh_vals * scale), -32768, 32767).astype(np.int16)
        return lut

    # ================================================================== #
    #                        HELPER METHODS                              #
    # ================================================================== #

    @staticmethod
    def quantize_leak_factor(leak_float, shift=8):
        """
        Quantize a floating-point leak factor for PReLU into INT16 + shift.

        Args:
            leak_float: float, e.g. 0.1
            shift: right-shift bits (default 8)

        Returns:
            (int16_operand, shift) tuple
        """
        operand = int(round(leak_float * (1 << shift)))
        operand = max(-32768, min(32767, operand))
        return operand, shift
