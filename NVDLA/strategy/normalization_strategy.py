"""
Normalization-layer strategy (Local Response Normalization / LRN).

Provides two golden-model functions:
    1. compute_golden()        — Generic LRN (standard formula, framework-like)
    2. compute_golden_nvdla()  — NVDLA CDP-specific pipeline model

Generic LRN formula (across channels):
    output[c, h, w] = input[c, h, w] / (k + alpha * sum(input[j, h, w]^2))^beta
    where j ranges over [c - floor(n/2), c + floor(n/2)]

NVDLA CDP pipeline (INT8, nv_small):
    1. Input Conversion:  cvt = saturate_9bit((input + datin_offset) * datin_scale >> datin_shifter)
    2. Square-Sum:        sqsum[c] = sum(cvt[j]^2) for j in [c-half, c+half]   (zero-padded edges)
    3. LUT Interpolation: lut_out = LUT_lookup(sqsum)   (LE/LO tables with linear interpolation)
    4. Multiply:          mul_out = cvt[c] * lut_out[c]
    5. Output Conversion: output = saturate_int8((mul_out + datout_offset) * datout_scale >> datout_shifter)

    Steps 2-4 can be independently bypassed via D_FUNC_BYPASS.

Memory format (nv_small):
    Data cube is stored in CHW order with 8-byte ATOM alignment.
    Each surface holds atomic_m channels (8 for INT8, 4 for INT16/FP16).
    Surfaces are stored contiguously: surface_stride = line_stride × height.
"""

from strategy.LayerStrategy import LayerStrategy
import numpy as np


class NormalizationStrategy(LayerStrategy):
    """Strategy for Local Response Normalization (LRN) layers."""

    # nv_small constants
    ATOM_SIZE = 8       # bytes
    ICVTO_BITS = 9      # internal conversion output width (signed)

    # Normalization length encoding
    _NORMALZ_LEN = {3: 0, 5: 1, 7: 2, 9: 3}

    def get_layer_type(self):
        return "normalization"

    # ------------------------------------------------------------------ #
    #  Validation                                                         #
    # ------------------------------------------------------------------ #
    def validate_norm_config(self, config):
        """
        Validate normalization layer configuration.

        Required keys:
            input_shape   : [H, W, C] or [H, W]
            normalz_len   : int (3, 5, 7, or 9)

        Optional keys with defaults:
            data_format   : 'INT8' (default)
            data_range    : [min, max]  (default [-5, 5])
            k, alpha, beta: LRN formula parameters (for generic model)
            datin_offset, datin_scale, datin_shifter   : input conversion params
            datout_offset, datout_scale, datout_shifter : output conversion params
            lut_le_table, lut_lo_table : LUT data arrays
            sqsum_bypass, mul_bypass   : bypass flags
        """
        required = ['input_shape', 'normalz_len']
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required normalization config key: {key}")

        normalz_len = config['normalz_len']
        if normalz_len not in self._NORMALZ_LEN:
            raise ValueError(
                f"normalz_len must be 3, 5, 7, or 9 — got {normalz_len}"
            )

        data_fmt = config.get('data_format', 'INT8')
        if data_fmt not in ('INT8', 'INT16', 'FP16'):
            raise ValueError(f"Unsupported data_format: {data_fmt}")

    # ------------------------------------------------------------------ #
    #  Input-data generation                                              #
    # ------------------------------------------------------------------ #
    def generate_input_data(self, config):
        """
        Return a numpy INT8 array in CHW order.

        Config keys used:
            input_shape : [H, W, C] or [H, W]
            num_channels: int (used if input_shape is 2D)
            data_range  : [min, max]  (default [-5, 5])
        """
        input_shape = config['input_shape']
        data_range = config.get('data_range', [-5, 5])

        if len(input_shape) == 3:
            h, w, c = input_shape
        else:
            h, w = input_shape
            c = config.get('num_channels', 1)

        return np.random.randint(
            data_range[0], data_range[1] + 1, size=(c, h, w)
        ).astype(np.int8)

    # ================================================================== #
    #  GOLDEN MODEL 1: Generic LRN (standard / framework-like)            #
    # ================================================================== #

    def compute_golden(self, input_data, config):
        """
        Standard Local Response Normalization across channels.

        Formula (per element):
            output[c, h, w] = input[c, h, w] / (k + alpha * sum(input[j, h, w]^2))^beta

        where j ranges over [c - floor(n/2), c + floor(n/2)], with
        out-of-bounds channels treated as zero.

        This matches the LRN formulation from AlexNet / Caffe / PyTorch's
        nn.LocalResponseNorm (cross-channel mode).

        Args:
            input_data : np.array (C, H, W) — signed integer or float input
            config     : dict with keys:
                normalz_len : int (3, 5, 7, or 9) — neighborhood size
                k           : float (default 1.0)  — bias constant
                alpha       : float (default 1e-4) — scaling factor
                beta        : float (default 0.75) — exponent
                output_dtype: str (default 'int8') — 'int8', 'int16', or 'float'

        Returns:
            np.array (C, H, W) — same spatial shape, clipped to output dtype
        """
        self.validate_norm_config(config)

        normalz_len = config['normalz_len']
        k     = config.get('k', 1.0)
        alpha = config.get('alpha', 1e-4)
        beta  = config.get('beta', 0.75)
        output_dtype = config.get('output_dtype', 'int8')

        half_n = normalz_len // 2

        # Work in float64 for precision
        inp = input_data.astype(np.float64)
        C, H, W = inp.shape
        output = np.zeros_like(inp, dtype=np.float64)

        # Compute per-channel square sum using sliding window
        # Precompute input^2
        inp_sq = inp ** 2

        for c in range(C):
            # Neighborhood: [c - half_n, c + half_n], clipped to [0, C-1]
            c_lo = max(0, c - half_n)
            c_hi = min(C - 1, c + half_n)

            # Sum of squares across the neighborhood (H, W)
            sqsum = np.sum(inp_sq[c_lo:c_hi + 1], axis=0)

            # LRN formula
            denominator = (k + alpha * sqsum) ** beta
            output[c] = inp[c] / denominator

        # Quantize output
        if output_dtype == 'int8':
            output = np.clip(np.round(output), -128, 127).astype(np.int8)
        elif output_dtype == 'int16':
            output = np.clip(np.round(output), -32768, 32767).astype(np.int16)
        else:
            output = output.astype(np.float32)

        return output

    # ================================================================== #
    #  GOLDEN MODEL 2: NVDLA CDP-Specific Pipeline                        #
    # ================================================================== #

    @staticmethod
    def _saturate_signed(value, bits):
        """Saturate a value to a signed range of the given bit width."""
        lo = -(1 << (bits - 1))
        hi = (1 << (bits - 1)) - 1
        return int(max(lo, min(hi, value)))

    @staticmethod
    def _arithmetic_right_shift(value, shift):
        """
        Arithmetic right shift matching Verilog >>> behavior.
        For non-negative shift: shift right with sign extension.
        For shift=0: return value unchanged.
        """
        if shift <= 0:
            return value
        # Python's >> on int is already arithmetic for negative numbers
        return int(value) >> shift

    @staticmethod
    def _shiftrightsu(value, shift):
        """
        Match NVDLA NV_NVDLA_HLS_shiftrightsu: arithmetic right shift
        with round-half-toward-negative-infinity rounding.

        Positive values: round half up (add guard bit).
        Negative values: round up only if strictly past halfway
                         (guard AND sticky), truncate at exact halfway.
        """
        if shift <= 0:
            return int(value)
        value = int(value)
        int_part = value >> shift  # arithmetic (Python handles sign)
        guard = (value >> (shift - 1)) & 1
        if shift > 1:
            sticky = (value & ((1 << (shift - 1)) - 1)) != 0
        else:
            sticky = False
        if value < 0:
            point5 = 1 if (guard and sticky) else 0
        else:
            point5 = guard
        return int_part + point5

    @staticmethod
    def _lut_linear_interpolate(x, table, start, end, index_select=0):
        """
        Linear interpolation within a LUT table — hardware-matching
        integer arithmetic.

        Matches NVDLA CDP interpolation unit (NV_NVDLA_CDP_DP_INTP_unit):
            result = X0 + round_half_toward_neg_inf((X1-X0)*frac_16, 16)

        In the NVDLA CDP hardware (LINEAR mode), the table index is
        computed as:  index = (x - start) >> index_select
        Each table segment covers 2^index_select input units.

        Args:
            x             : input value (int)
            table         : list/array of signed 16-bit LUT entries
            start         : LUT range start value
            end           : LUT range end value (not used in computation)
            index_select  : right-shift amount for index (default 0)

        Returns:
            int — interpolated LUT value (signed 16-bit range)
        """
        num_segments = len(table) - 1
        if num_segments <= 0:
            return int(table[0]) if len(table) > 0 else 0

        x_offset = int(x) - int(start)
        if x_offset < 0:
            return int(table[0])  # underflow

        idx = x_offset >> index_select  # integer part = table index

        if idx >= num_segments:
            return int(table[num_segments])  # overflow

        lo_val = int(table[idx])
        hi_val = int(table[idx + 1])
        diff = hi_val - lo_val

        if diff == 0:
            return lo_val

        # Fraction: lower index_select bits, normalized to 16-bit
        frac_bits = x_offset & ((1 << index_select) - 1) if index_select > 0 else 0
        if index_select <= 16:
            frac_16 = frac_bits << (16 - index_select) if index_select > 0 else 0
        else:
            frac_16 = frac_bits >> (index_select - 16)

        # Product and round-half-toward-negative-infinity (matches INTP unit)
        product = diff * frac_16
        shift = 16
        int_part = product >> shift
        guard = (product >> (shift - 1)) & 1
        if shift > 1:
            sticky = (product & ((1 << (shift - 1)) - 1)) != 0
        else:
            sticky = False
        if product < 0:
            point5 = 1 if (guard and sticky) else 0
        else:
            point5 = guard
        interpolated = int_part + point5

        result = lo_val + interpolated
        # Saturate to signed 16-bit
        return max(-32768, min(32767, result))

    def compute_golden_nvdla(self, input_data, config):
        """
        NVDLA CDP hardware-accurate golden model.

        Replicates the exact CDP datapath pipeline:
            CvtIn → BufferIn → SqSum → LUT Interpolation → Mul → CvtOut → WDMA

        The model uses integer arithmetic with explicit saturation and shifting
        to match RTL behavior as closely as possible.

        Args:
            input_data : np.int8 array (C, H, W) — raw input from memory
            config     : dict with keys:

                --- Normalization ---
                normalz_len   : int (3, 5, 7, or 9)

                --- Input Conversion ---
                datin_offset  : int (signed, 8-bit for INT8)   default 0
                datin_scale   : int (signed 16-bit)            default 1
                datin_shifter : int (unsigned 5-bit, 0-31)     default 0

                --- Output Conversion ---
                datout_offset  : int (signed 32-bit)           default 0
                datout_scale   : int (signed 16-bit)           default 1
                datout_shifter : int (unsigned 6-bit, 0-63)    default 0

                --- LUT Tables ---
                lut_le_table  : list of 65 signed 16-bit values (LE table)
                lut_lo_table  : list of 257 signed 16-bit values (LO table)
                lut_le_start  : int — LE table input range start  (default 0)
                lut_le_end    : int — LE table input range end    (default 65535)
                lut_lo_start  : int — LO table input range start  (default 0)
                lut_lo_end    : int — LO table input range end    (default 65535)
                lut_priority  : str — 'LE' or 'LO' for hybrid/oflow/uflow
                                      (default 'LE')

                --- Bypass ---
                sqsum_bypass  : bool (default False) — skip square-sum
                mul_bypass    : bool (default False) — skip multiply

                --- Slope extrapolation (for out-of-range LUT inputs) ---
                lut_le_slope_uflow_scale : int (signed 16-bit, default 0)
                lut_le_slope_uflow_shift : int (signed 5-bit,  default 0)
                lut_le_slope_oflow_scale : int (signed 16-bit, default 0)
                lut_le_slope_oflow_shift : int (signed 5-bit,  default 0)
                lut_lo_slope_uflow_scale : int (signed 16-bit, default 0)
                lut_lo_slope_uflow_shift : int (signed 5-bit,  default 0)
                lut_lo_slope_oflow_scale : int (signed 16-bit, default 0)
                lut_lo_slope_oflow_shift : int (signed 5-bit,  default 0)

        Returns:
            np.int8 array (C, H, W) — NVDLA-matching output
        """
        self.validate_norm_config(config)

        normalz_len = config['normalz_len']
        half_n = normalz_len // 2

        # --- Input Conversion parameters ---
        datin_offset  = config.get('datin_offset', 0)
        datin_scale   = config.get('datin_scale', 1)
        datin_shifter = config.get('datin_shifter', 0)

        # --- Output Conversion parameters ---
        datout_offset  = config.get('datout_offset', 0)
        datout_scale   = config.get('datout_scale', 1)
        datout_shifter = config.get('datout_shifter', 0)

        # --- Bypass flags ---
        sqsum_bypass = config.get('sqsum_bypass', False)
        mul_bypass   = config.get('mul_bypass', False)

        # --- LUT tables and ranges ---
        # Default: identity-like LUT (output = 1 for any input → pass-through)
        lut_le_table = config.get('lut_le_table', None)
        lut_lo_table = config.get('lut_lo_table', None)
        lut_le_start = config.get('lut_le_start', 0)
        lut_le_end   = config.get('lut_le_end', 65535)
        lut_lo_start = config.get('lut_lo_start', 0)
        lut_lo_end   = config.get('lut_lo_end', 65535)
        lut_priority = config.get('lut_priority', 'LE')

        # Slope extrapolation parameters
        le_slope_uflow_scale = config.get('lut_le_slope_uflow_scale', 0)
        le_slope_uflow_shift = config.get('lut_le_slope_uflow_shift', 0)
        le_slope_oflow_scale = config.get('lut_le_slope_oflow_scale', 0)
        le_slope_oflow_shift = config.get('lut_le_slope_oflow_shift', 0)
        lo_slope_uflow_scale = config.get('lut_lo_slope_uflow_scale', 0)
        lo_slope_uflow_shift = config.get('lut_lo_slope_uflow_shift', 0)
        lo_slope_oflow_scale = config.get('lut_lo_slope_oflow_scale', 0)
        lo_slope_oflow_shift = config.get('lut_lo_slope_oflow_shift', 0)

        C, H, W = input_data.shape
        icvto_bits = self.ICVTO_BITS  # 9 for nv_small INT8

        # ============================================================== #
        #  STAGE 1: Input Conversion (CvtIn)                              #
        #  Formula: cvt = saturate_icvto((input + offset) * scale >> shift)#
        # ============================================================== #
        inp_i32 = input_data.astype(np.int32)
        cvt_in = np.zeros((C, H, W), dtype=np.int32)

        for c in range(C):
            for h in range(H):
                for w in range(W):
                    val = int(inp_i32[c, h, w]) + datin_offset
                    val = val * datin_scale
                    val = self._arithmetic_right_shift(val, datin_shifter)
                    val = self._saturate_signed(val, icvto_bits)
                    cvt_in[c, h, w] = val

        # ============================================================== #
        #  STAGE 2: Square-Sum (with channel sliding window)              #
        #  For each (c, h, w): sqsum = sum(cvt[j, h, w]^2)              #
        #  where j in [c - half_n, c + half_n], zero-padded at edges     #
        # ============================================================== #
        sqsum = np.zeros((C, H, W), dtype=np.int64)

        if not sqsum_bypass:
            cvt_sq = cvt_in.astype(np.int64) ** 2
            for c in range(C):
                c_lo = max(0, c - half_n)
                c_hi = min(C - 1, c + half_n)
                sqsum[c] = np.sum(cvt_sq[c_lo:c_hi + 1], axis=0)
        else:
            # Bypass: feed converted input directly to LUT
            sqsum = cvt_in.astype(np.int64)

        # ============================================================== #
        #  STAGE 3: LUT Interpolation                                     #
        #  Maps sqsum → normalization factor via LE and LO tables         #
        # ============================================================== #
        lut_out = np.zeros((C, H, W), dtype=np.int64)

        # --- LUT index_select (power-of-2 segment size) ---
        le_index_select = config.get('lut_le_index_select', 0)
        lo_index_select = config.get('lut_lo_index_select', 0)

        for c in range(C):
            for h in range(H):
                for w in range(W):
                    x = int(sqsum[c, h, w])
                    lut_out[c, h, w] = self._lut_lookup(
                        x, lut_le_table, lut_lo_table,
                        lut_le_start, lut_le_end,
                        lut_lo_start, lut_lo_end,
                        lut_priority,
                        le_slope_uflow_scale, le_slope_uflow_shift,
                        le_slope_oflow_scale, le_slope_oflow_shift,
                        lo_slope_uflow_scale, lo_slope_uflow_shift,
                        lo_slope_oflow_scale, lo_slope_oflow_shift,
                        le_index_select, lo_index_select,
                    )

        # ============================================================== #
        #  STAGE 4: Multiply (input × LUT output)                        #
        #  mul_out[c] = cvt_in[c] * lut_out[c]                          #
        # ============================================================== #
        if not mul_bypass:
            mul_out = cvt_in.astype(np.int64) * lut_out
        else:
            # Bypass multiply: LUT output goes directly to output conversion
            mul_out = lut_out.copy()

        # ============================================================== #
        #  STAGE 5: Output Conversion (CvtOut)                            #
        #  output = saturate_int8((mul_out + offset) * scale >> shifter)  #
        # ============================================================== #
        output = np.zeros((C, H, W), dtype=np.int8)

        for c in range(C):
            for h in range(H):
                for w in range(W):
                    # RTL does: sub = (mul_out - offset); scaled = sub * scale;
                    #           output = shiftrightsu(scaled, shifter)
                    val = int(mul_out[c, h, w]) - datout_offset
                    val = val * datout_scale
                    val = self._shiftrightsu(val, datout_shifter)
                    val = self._saturate_signed(val, 8)  # INT8 output
                    output[c, h, w] = val

        return output

    # ------------------------------------------------------------------ #
    #  LUT lookup helper with underflow/overflow slope extrapolation       #
    # ------------------------------------------------------------------ #
    def _lut_lookup(self, x,
                    le_table, lo_table,
                    le_start, le_end,
                    lo_start, lo_end,
                    priority,
                    le_slope_uflow_scale, le_slope_uflow_shift,
                    le_slope_oflow_scale, le_slope_oflow_shift,
                    lo_slope_uflow_scale, lo_slope_uflow_shift,
                    lo_slope_oflow_scale, lo_slope_oflow_shift,
                    le_index_select=0, lo_index_select=0):
        """
        Perform LUT lookup with LE/LO table priority resolution and
        slope-based extrapolation for out-of-range inputs.

        The NVDLA CDP has two LUT tables:
            LE table (65 entries)  — fine-grained, linear or exponential mapping
            LO table (257 entries) — coarse, always linear mapping

        In LINEAR mode the table index is: (x - start) >> index_select.
        A "hit" means 0 <= index < num_segments (64 for LE, 256 for LO).
        """
        def _is_hit(table, start, index_select):
            if table is None:
                return False
            x_off = int(x) - int(start)
            if x_off < 0:
                return False
            idx = x_off >> index_select
            return idx < (len(table) - 1)

        le_hit = _is_hit(le_table, le_start, le_index_select)
        lo_hit = _is_hit(lo_table, lo_start, lo_index_select)

        # Both tables cover the input
        if le_hit and lo_hit:
            if priority == 'LE':
                return self._lut_linear_interpolate(x, le_table, le_start, le_end, le_index_select)
            else:
                return self._lut_linear_interpolate(x, lo_table, lo_start, lo_end, lo_index_select)

        # Only LE covers it
        if le_hit:
            return self._lut_linear_interpolate(x, le_table, le_start, le_end, le_index_select)

        # Only LO covers it
        if lo_hit:
            return self._lut_linear_interpolate(x, lo_table, lo_start, lo_end, lo_index_select)

        # Neither covers it — slope extrapolation
        return self._slope_extrapolate(
            x, le_table, lo_table,
            le_start, le_end, lo_start, lo_end,
            priority,
            le_slope_uflow_scale, le_slope_uflow_shift,
            le_slope_oflow_scale, le_slope_oflow_shift,
            lo_slope_uflow_scale, lo_slope_uflow_shift,
            lo_slope_oflow_scale, lo_slope_oflow_shift,
        )

    def _slope_extrapolate(self, x,
                           le_table, lo_table,
                           le_start, le_end, lo_start, lo_end,
                           priority,
                           le_slope_uflow_scale, le_slope_uflow_shift,
                           le_slope_oflow_scale, le_slope_oflow_shift,
                           lo_slope_uflow_scale, lo_slope_uflow_shift,
                           lo_slope_oflow_scale, lo_slope_oflow_shift):
        """
        Slope-based extrapolation when input x is outside LUT range.

        NVDLA calculates:
            If underflow (x < start):
                result = base_value + (x - start) * slope_scale >> slope_shift
            If overflow (x > end):
                result = base_value + (x - end) * slope_scale >> slope_shift

        The priority register selects which table's slope to use.
        """
        use_le = (priority == 'LE') and (le_table is not None)

        if use_le:
            table = le_table
            start, end = le_start, le_end
            uflow_scale, uflow_shift = le_slope_uflow_scale, le_slope_uflow_shift
            oflow_scale, oflow_shift = le_slope_oflow_scale, le_slope_oflow_shift
        elif lo_table is not None:
            table = lo_table
            start, end = lo_start, lo_end
            uflow_scale, uflow_shift = lo_slope_uflow_scale, lo_slope_uflow_shift
            oflow_scale, oflow_shift = lo_slope_oflow_scale, lo_slope_oflow_shift
        else:
            # No tables configured at all — return 0
            return 0.0

        if x < start:
            # Underflow extrapolation
            base = int(table[0])
            delta = int(x) - int(start)
            slope_val = delta * int(uflow_scale)
            if uflow_shift > 0:
                slope_val = self._arithmetic_right_shift(int(slope_val), uflow_shift)
            return base + int(slope_val)
        else:
            # Overflow extrapolation
            base = int(table[-1])
            delta = int(x) - int(end)
            slope_val = delta * int(oflow_scale)
            if oflow_shift > 0:
                slope_val = self._arithmetic_right_shift(int(slope_val), oflow_shift)
            return base + int(slope_val)

    # ------------------------------------------------------------------ #
    #  LUT generation utilities                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def generate_lrn_lut(k, alpha, beta, normalz_len, num_entries,
                         sqsum_min=0, sqsum_max=None, index_select=None):
        """
        Generate a LUT that approximates the LRN normalization factor:

            f(sqsum) = 1 / (k + alpha * sqsum) ^ beta

        This is the function that the CDP LUT should store so that:
            output[c] ≈ input[c] × LUT(sqsum[c])

        In the NVDLA hardware, the LUT index for LINEAR mode is computed as:
            index = (sqsum - start) >> index_select
        Each table segment covers 2^index_select sqsum units.

        Args:
            k            : float — LRN bias
            alpha        : float — LRN scaling factor
            beta         : float — LRN exponent
            normalz_len  : int   — normalization window size (3, 5, 7, or 9)
            num_entries  : int   — number of LUT entries (65 for LE, 257 for LO)
            sqsum_min    : int   — minimum expected sqsum value (default 0)
            sqsum_max    : int   — maximum expected sqsum value
                                   (default: 256^2 × normalz_len for INT8)
            index_select : int   — right-shift for index (default: auto-computed)

        Returns:
            tuple: (table, start, end, index_select)
                table        : list of int (signed 16-bit LUT values, Q8.8)
                start        : int — LUT range start
                end          : int — LUT range end (last entry's sqsum value)
                index_select : int — the shift used
        """
        import math

        if sqsum_max is None:
            sqsum_max = (256 ** 2) * normalz_len

        num_segments = num_entries - 1  # 64 for LE, 256 for LO

        if index_select is None:
            # Auto-compute: choose smallest shift so that the full sqsum
            # range maps to <= num_segments indices.
            if sqsum_max <= num_segments:
                index_select = 0
            else:
                index_select = math.ceil(math.log2(max(1, sqsum_max) / num_segments))

        segment_size = 1 << index_select

        table = []
        for i in range(num_entries):
            # Each entry i corresponds to sqsum = sqsum_min + i * segment_size
            sqsum = sqsum_min + i * segment_size
            denom = (k + alpha * sqsum) ** beta
            factor = 1.0 / denom if denom != 0 else 0.0
            fp_val = int(np.clip(np.round(factor * 256), -32768, 32767))
            table.append(fp_val)

        start = sqsum_min
        end = sqsum_min + num_segments * segment_size

        return table, start, end, index_select

    @staticmethod
    def generate_identity_lut(num_entries):
        """
        Generate an identity LUT where every entry = 256 (i.e., 1.0 in Q8.8).
        Useful for bypass/debug testing.

        Returns:
            list of int — all entries set to 256 (1.0 in Q8.8 fixed-point)
        """
        return [256] * num_entries
