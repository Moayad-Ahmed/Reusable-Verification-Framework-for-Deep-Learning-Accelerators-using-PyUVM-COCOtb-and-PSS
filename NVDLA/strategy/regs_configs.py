import math
from strategy.Layers_regs_addresses import PDP_REG
from strategy.Layers_regs_addresses import PDP_RDMA_REG
from strategy.Layers_regs_addresses import CDP_REG, CDP_RDMA_REG
from strategy.Layers_regs_addresses import (
    GLB_REG, CDMA_REG, CSC_REG, CMAC_A_REG, CMAC_B_REG, CACC_REG, SDP_REG,SDP_RDMA_REG,
)


# Lookup tables for encoding
_POOL_METHOD = {'avg': 0x0, 'max': 0x1, 'min': 0x2}
_DATA_FMT    = {'INT8': 0x0, 'INT16': 0x1, 'FP16': 0x2}
_BYTES_PER_EL = {'INT8': 1, 'INT16': 2, 'FP16': 2}


class RegistrationConfigs:

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _recip_fixed(kernel_dim):
        """Reciprocal in Q0.16 fixed-point: round(65536 / kernel_dim)."""
        return round(65536 / kernel_dim) if kernel_dim > 0 else 0x10000

    @staticmethod
    def _partial_width_reg(total_width):
        """
        Build D_PARTIAL_WIDTH_IN / _OUT for split_num=0 (single split).
        FIRST = total_width-1, MID = 0, LAST = 0  (only FIRST matters).
        Layout: [9:0]=FIRST-1, [19:10]=LAST-1, [29:20]=MID-1
        With a single split, first = total_width - 1.
        """
        first = (total_width - 1) & 0x3FF
        return first  # MID=0, LAST=0 → upper bits stay 0

    @staticmethod
    def _output_dim(input_dim, pad_before, pad_after, kernel, stride):
        """Compute output size: (input + pad_before + pad_after - kernel) / stride + 1"""
        return (input_dim + pad_before + pad_after - kernel) // stride + 1

    # ----------------------------------------------------------- main function
    @staticmethod
    def _split_addr(addr):
        """Split a 64-bit address into (low_32, high_32) for NVDLA registers."""
        return addr & 0xFFFFFFFF, (addr >> 32) & 0xFFFFFFFF

    def pooling_configs(self, layer_configs, src_addr=0x0, dst_addr=0x100):
        """
        Build PDP + PDP_RDMA register list from a YAML layer config dict.

        Expected keys in layer_configs:
            kernel_size     : int
            stride          : int
            pool_type       : str  ('avg' | 'max' | 'min')
            input_shape     : [height, width]
            data_range      : [min, max]
            data_format     : str  ('INT8' | 'INT16' | 'FP16')
            padding_left    : int  (default 0)
            padding_right   : int  (default 0)
            padding_top     : int  (default 0)
            padding_bottom  : int  (default 0)
            padding_value   : int  (default 0)

        Args:
            layer_configs: dict with keys listed above
            src_addr: source base address in DRAM (input data)
            dst_addr: destination base address in DRAM (output data,
                    must not overlap the input region)
        """
        # ---------- extract config values ----------
        kernel   = layer_configs['kernel_size']
        stride   = layer_configs['stride']
        pool_type = layer_configs['pool_type']
        
        # Extract input_shape - support both [H,W] and [H,W,C] formats
        input_shape = layer_configs['input_shape']
        in_h = input_shape[0]
        in_w = input_shape[1]
        channels = input_shape[2] if len(input_shape) == 3 else 1
        
        data_fmt = layer_configs.get('data_format', 'INT8')

        pad_l = layer_configs.get('padding_left', 0)
        pad_r = layer_configs.get('padding_right', 0)
        pad_t = layer_configs.get('padding_top', 0)
        pad_b = layer_configs.get('padding_bottom', 0)
        pad_val = layer_configs.get('padding_value', 0)

        bpe = _BYTES_PER_EL[data_fmt]

        # NVDLA memory atomic size (nv_small = 8 bytes).
        # Each spatial pixel stores ceil(C*bpe / ATOM) atoms of data.
        ATOM = 8
        atoms_per_pixel = max(1, (channels * bpe + ATOM - 1) // ATOM)
        pixel_bytes = atoms_per_pixel * ATOM  # bytes per pixel in memory

        # ---------- derived dimensions ----------
        out_w = self._output_dim(in_w, pad_l, pad_r, kernel, stride)
        out_h = self._output_dim(in_h, pad_t, pad_b, kernel, stride)

        # ---------- register field encodings ----------
        # Kernel/stride: value-1 encoding
        k_enc = kernel - 1       # 0-7
        s_enc = stride - 1       # 0-15

        # PDP D_POOLING_KERNEL_CFG
        #   [3:0]=kernel_width-1, [11:8]=kernel_height-1,
        #   [19:16]=stride_width-1, [23:20]=stride_height-1
        pdp_kernel_cfg = (
            (k_enc & 0xF)
            | ((k_enc & 0xF) << 8)
            | ((s_enc & 0xF) << 16)
            | ((s_enc & 0xF) << 20)
        )

        # PDP_RDMA D_POOLING_KERNEL_CFG
        #   [3:0]=kernel_width-1, [7:4]=stride_width-1
        rdma_kernel_cfg = (k_enc & 0xF) | ((s_enc & 0xF) << 4)

        # PDP D_OPERATION_MODE_CFG
        #   [1:0]=method, [4]=flying(1=OFF), [15:8]=split_num(0=1 split)
        method_enc = _POOL_METHOD[pool_type]
        flying = 1  # OFF_FLYING (read from memory via RDMA)
        split_num = 0
        op_mode_cfg = method_enc | (flying << 4) | (split_num << 8)

        # PDP D_POOLING_PADDING_CFG
        #   [2:0]=L, [6:4]=T, [10:8]=R, [14:12]=B
        pdp_pad_cfg = (
            (pad_l & 0x7)
            | ((pad_t & 0x7) << 4)
            | ((pad_r & 0x7) << 8)
            | ((pad_b & 0x7) << 12)
        )

        # PDP_RDMA D_POOLING_PADDING_CFG  [3:0] = max(pad_l, pad_r)
        rdma_pad_cfg = max(pad_l, pad_r) & 0xF

        # Reciprocal (only meaningful for AVG pooling, but always written)
        recip_w = self._recip_fixed(kernel)
        recip_h = self._recip_fixed(kernel)

        # Data format register
        fmt_enc = _DATA_FMT[data_fmt]

        # Memory layout — atom-aligned data
        # In NVDLA nv_small, each pixel occupies one 8-byte atom in memory.
        # Line stride  = W * pixel_bytes   (distance between consecutive rows)
        # Surface stride = line_stride * H  (distance between channel surfaces)
        src_line_stride    = in_w * pixel_bytes
        src_surface_stride = src_line_stride * in_h

        dst_line_stride    = out_w * pixel_bytes
        dst_surface_stride = dst_line_stride * out_h

        # Source / destination base addresses — split into low and high 32-bit parts
        src_addr_low, src_addr_high = self._split_addr(src_addr)
        dst_addr_low, dst_addr_high = self._split_addr(dst_addr)

        # Partial widths (single split)
        partial_in  = self._partial_width_reg(in_w + pad_l + pad_r)
        partial_out = self._partial_width_reg(out_w)

        # RDMA partial width
        rdma_partial_in = self._partial_width_reg(in_w + pad_l + pad_r)

        # Padding fill value — sign-extend to 19 bits for the register
        pad_val_19 = pad_val & 0x7FFFF

        # ========================= BUILD REGISTER LIST =========================
        return [
            # -------- PDP CORE: Status and Control --------
            (PDP_REG.S_STATUS, 0x00000000),
            (PDP_REG.S_POINTER, 0x00000000),

            # -------- PDP CORE: Input Data Cube (value-1 encoding) --------
            (PDP_REG.D_DATA_CUBE_IN_WIDTH,   in_w - 1),
            (PDP_REG.D_DATA_CUBE_IN_HEIGHT,  in_h - 1),
            (PDP_REG.D_DATA_CUBE_IN_CHANNEL, channels - 1),

            # -------- PDP CORE: Output Data Cube (value-1 encoding) --------
            (PDP_REG.D_DATA_CUBE_OUT_WIDTH,   out_w - 1),
            (PDP_REG.D_DATA_CUBE_OUT_HEIGHT,  out_h - 1),
            (PDP_REG.D_DATA_CUBE_OUT_CHANNEL, channels - 1),

            # -------- PDP CORE: Operation Mode --------
            (PDP_REG.D_OPERATION_MODE_CFG, op_mode_cfg),
            (PDP_REG.D_NAN_FLUSH_TO_ZERO,  0x00000000),

            # -------- PDP CORE: Partial Width --------
            (PDP_REG.D_PARTIAL_WIDTH_IN,  partial_in),
            (PDP_REG.D_PARTIAL_WIDTH_OUT, partial_out),

            # -------- PDP CORE: Pooling Kernel --------
            (PDP_REG.D_POOLING_KERNEL_CFG, pdp_kernel_cfg),
            (PDP_REG.D_RECIP_KERNEL_HEIGHT, recip_h),
            (PDP_REG.D_RECIP_KERNEL_WIDTH,  recip_w),

            # -------- PDP CORE: Pooling Padding --------
            (PDP_REG.D_POOLING_PADDING_CFG,           pdp_pad_cfg),
            (PDP_REG.D_POOLING_PADDING_VALUE_1_CFG,   pad_val_19),
            (PDP_REG.D_POOLING_PADDING_VALUE_2_CFG,   2*pad_val_19),
            (PDP_REG.D_POOLING_PADDING_VALUE_3_CFG,   3*pad_val_19),
            (PDP_REG.D_POOLING_PADDING_VALUE_4_CFG,   4*pad_val_19),
            (PDP_REG.D_POOLING_PADDING_VALUE_5_CFG,   5*pad_val_19),
            (PDP_REG.D_POOLING_PADDING_VALUE_6_CFG,   6*pad_val_19),
            (PDP_REG.D_POOLING_PADDING_VALUE_7_CFG,   7*pad_val_19),

            # -------- PDP CORE: Source Memory --------
            (PDP_REG.D_SRC_BASE_ADDR_LOW,  src_addr_low),
            (PDP_REG.D_SRC_BASE_ADDR_HIGH, src_addr_high),
            (PDP_REG.D_SRC_LINE_STRIDE,    src_line_stride),
            (PDP_REG.D_SRC_SURFACE_STRIDE, src_surface_stride),

            # -------- PDP CORE: Destination Memory --------
            (PDP_REG.D_DST_BASE_ADDR_LOW,    dst_addr_low),
            (PDP_REG.D_DST_BASE_ADDR_HIGH,   dst_addr_high),
            (PDP_REG.D_DST_LINE_STRIDE,      dst_line_stride),
            (PDP_REG.D_DST_SURFACE_STRIDE,   dst_surface_stride),
            (PDP_REG.D_DST_RAM_CFG,          0x00000001),   # MC (external memory)

            # -------- PDP CORE: Data Format --------
            (PDP_REG.D_DATA_FORMAT, fmt_enc),

            # -------- PDP CORE: Statistics (read-only, write zeros to clear) --------
            (PDP_REG.D_INF_INPUT_NUM,  0x00000000),
            (PDP_REG.D_NAN_INPUT_NUM,  0x00000000),
            (PDP_REG.D_NAN_OUTPUT_NUM, 0x00000000),

            # -------- PDP CORE: Performance --------
            (PDP_REG.D_PERF_ENABLE,      0x00000000),
            (PDP_REG.D_PERF_WRITE_STALL, 0x00000000),

            # -------- PDP CORE: Debug --------
            (PDP_REG.D_CYA, 0x00000000),

            # ======== PDP RDMA: Status and Control ========
            (PDP_RDMA_REG.S_STATUS,  0x00000000),
            (PDP_RDMA_REG.S_POINTER, 0x00000000),

            # ======== PDP RDMA: Input Data Cube (value-1 encoding) ========
            (PDP_RDMA_REG.D_DATA_CUBE_IN_WIDTH,   in_w - 1),
            (PDP_RDMA_REG.D_DATA_CUBE_IN_HEIGHT,  in_h - 1),
            (PDP_RDMA_REG.D_DATA_CUBE_IN_CHANNEL, channels - 1),

            # ======== PDP RDMA: Flying Mode ========
            (PDP_RDMA_REG.D_FLYING_MODE, 0x00000001),  # OFF_FLYING

            # ======== PDP RDMA: Source Memory ========
            (PDP_RDMA_REG.D_SRC_BASE_ADDR_LOW,  src_addr_low),
            (PDP_RDMA_REG.D_SRC_BASE_ADDR_HIGH, src_addr_high),
            (PDP_RDMA_REG.D_SRC_LINE_STRIDE,    src_line_stride),
            (PDP_RDMA_REG.D_SRC_SURFACE_STRIDE, src_surface_stride),
            (PDP_RDMA_REG.D_SRC_RAM_CFG,        0x00000001),  # MC

            # ======== PDP RDMA: Data Format and Operation ========
            (PDP_RDMA_REG.D_DATA_FORMAT,          fmt_enc),
            (PDP_RDMA_REG.D_OPERATION_MODE_CFG,   split_num),  # split_num only

            # ======== PDP RDMA: Pooling Configuration ========
            (PDP_RDMA_REG.D_POOLING_KERNEL_CFG,  rdma_kernel_cfg),
            (PDP_RDMA_REG.D_POOLING_PADDING_CFG, rdma_pad_cfg),
            (PDP_RDMA_REG.D_PARTIAL_WIDTH_IN,    rdma_partial_in),

            # ======== PDP RDMA: Performance ========
            (PDP_RDMA_REG.D_PERF_ENABLE,     0x00000000),
            (PDP_RDMA_REG.D_PERF_READ_STALL, 0x00000000),

            # ======== PDP RDMA: Debug ========
            (PDP_RDMA_REG.D_CYA, 0x00000000),

            # ======== ENABLE OPERATIONS (Must be last! RDMA first) ========
            (PDP_RDMA_REG.D_OP_ENABLE, 0x00000001),
            (PDP_REG.D_OP_ENABLE,      0x00000001),
        ]

    def conv_configs(self, layer_configs, input_addr=0x0, weight_addr=0x100, output_addr=0x200):
        """
        Build the full convolution pipeline register list (CDMA→CSC→CMAC→CACC)
        plus SDP in transparent passthrough mode so the raw convolution result
        is written to DRAM.

        The CACC has NO direct path to DRAM — output MUST flow through SDP.
        SDP is configured with flying_mode=1 and all BS/BN/EW sub-units
        bypassed, giving an effectively "convolution-only" result.

        Enable order (bottom-up): SDP → CACC → CMAC_A → CMAC_B → CSC → CDMA

        Args:
            layer_configs: dict — see convolution YAML for expected keys
            input_addr:  DRAM base address of input feature data
            weight_addr: DRAM base address of formatted weight data
            output_addr: DRAM base address for output result
        """
        # ── extract config ────────────────────────────────────────────
        kernel_h     = layer_configs.get('kernel_h', 1)
        kernel_w     = layer_configs.get('kernel_w', 1)
        num_channels = layer_configs['num_channels']
        num_kernels  = layer_configs['num_kernels']
        stride_h     = layer_configs.get('stride_h', 1)
        stride_w     = layer_configs.get('stride_w', 1)
        pad_l        = layer_configs.get('padding_left', 0)
        pad_t        = layer_configs.get('padding_top', 0)
        pad_value    = layer_configs.get('padding_value', 0)
        clip_truncate = layer_configs.get('clip_truncate', 0)
        dilation_x   = layer_configs.get('dilation_x', 1)
        dilation_y   = layer_configs.get('dilation_y', 1)

        input_shape  = layer_configs['input_shape']
        in_h = input_shape[0]
        in_w = input_shape[1]

        # nv_small constants
        ATOM_C = 8
        ATOM_K = 8
        BPE    = 1  # INT8

        # ── derived dimensions ────────────────────────────────────────
        eff_kh = (kernel_h - 1) * dilation_y + 1
        eff_kw = (kernel_w - 1) * dilation_x + 1
        out_h  = (in_h + pad_t * 2 - eff_kh) // stride_h + 1   # symmetric pad
        out_w  = (in_w + pad_l * 2 - eff_kw) // stride_w + 1

        # ── weight byte calculations ──────────────────────────────────
        bytes_per_kernel   = kernel_h * kernel_w * num_channels * BPE
        num_kernel_groups  = (num_kernels  + ATOM_K - 1) // ATOM_K
        num_channel_groups = (num_channels + ATOM_C - 1) // ATOM_C
        weight_raw = num_kernel_groups * kernel_h * kernel_w * num_channel_groups * ATOM_K * ATOM_C
        weight_bytes = max(128, ((weight_raw + 127) // 128) * 128)

        # ── input memory layout ───────────────────────────────────────
        # NVDLA CDMA addresses data per-surface (each surface = 1 atom = 8B).
        # line_stride = distance between adjacent rows *within one surface*.
        # surf_stride = distance between the start of surface N and surface N+1.
        atoms_per_pixel = max(1, (num_channels * BPE + ATOM_C - 1) // ATOM_C)
        pixel_bytes     = atoms_per_pixel * ATOM_C   # total bytes per pixel (all surfaces)
        input_line_stride = in_w * ATOM_C             # per-surface: W pixels × 1 atom each
        input_surf_stride = input_line_stride * in_h  # contiguous surfaces

        # ── CBUF slice ────────────────────────────────────────────────
        entries_per_slice = in_w * num_channel_groups  # CBUF entries per line
        entries_per_slice_val = entries_per_slice - 1   # value-1 encoding

        # ── register field encodings ──────────────────────────────────
        # D_MISC_CFG: [0]=DC, [13:12]=INT8, [28]=skip_weight_rls
        misc_cfg       = 0x10000000

        datain_size_0  = ((in_h - 1) << 16) | (in_w - 1)
        datain_size_1  = num_channels - 1
        conv_stride    = ((stride_h - 1) << 16) | (stride_w - 1)
        zero_padding   = (pad_l & 0x1F) | ((pad_t & 0x1F) << 16)
        bank_cfg       = (7 << 16) | 6  # weight_bank=7, data_bank=6

        weight_size_0  = bytes_per_kernel - 1
        weight_size_1  = num_kernels - 1

        # CSC specific
        datain_size_ext_0 = datain_size_0   # mirrors for DC mode
        datain_size_ext_1 = num_channels - 1
        weight_size_ext_0 = ((kernel_h - 1) << 16) | (kernel_w - 1)
        weight_size_ext_1 = ((num_kernels - 1) << 16) | (num_channels - 1)
        dataout_size_0    = ((out_h - 1) << 16) | (out_w - 1)
        dataout_size_1    = num_kernels - 1
        atomics_val       = out_w * out_h - 1  # renamed to avoid collision
        dilation_ext      = ((dilation_y - 1) << 16) | (dilation_x - 1)

        # CACC output (32-bit per element internally)
        cacc_line_stride  = out_w * ATOM_K * 4
        cacc_surf_stride  = cacc_line_stride * out_h

        # SDP output (INT8)
        # SDP also uses per-surface addressing (each surface = 1 atom = 8B).
        out_atoms = max(1, (num_kernels * BPE + ATOM_C - 1) // ATOM_C)
        out_pixel_bytes = out_atoms * ATOM_C          # total bytes per output pixel
        sdp_line_stride = out_w * ATOM_C              # per-surface output stride
        sdp_surf_stride = sdp_line_stride * out_h     # contiguous surfaces

        # address splits (32-bit addressing)
        in_lo  = input_addr  & 0xFFFFFFFF
        in_hi  = (input_addr  >> 32) & 0xFFFFFFFF
        wt_lo  = weight_addr & 0xFFFFFFFF
        wt_hi  = (weight_addr >> 32) & 0xFFFFFFFF
        out_lo = output_addr & 0xFFFFFFFF
        out_hi = (output_addr >> 32) & 0xFFFFFFFF

        # ═════════════ BUILD REGISTER LIST ═════════════════════════════
        return [
            # ── GLB: interrupt config ─────────────────────────────────
            (GLB_REG.S_INTR_MASK,   0x003F03FC),  # unmask SDP_done only
            (GLB_REG.S_INTR_STATUS, 0x00000000),

            # ── SDP: transparent passthrough ──────────────────────────
            (SDP_REG.S_POINTER,            0x00000000),
            (SDP_REG.D_DATA_CUBE_WIDTH,    out_w - 1),
            (SDP_REG.D_DATA_CUBE_HEIGHT,   out_h - 1),
            (SDP_REG.D_DATA_CUBE_CHANNEL,  num_kernels - 1),
            (SDP_REG.D_DST_BASE_ADDR_LOW,  out_lo),
            (SDP_REG.D_DST_BASE_ADDR_HIGH, out_hi),
            (SDP_REG.D_DST_LINE_STRIDE,    sdp_line_stride),
            (SDP_REG.D_DST_SURFACE_STRIDE, sdp_surf_stride),
            (SDP_REG.D_DP_BS_CFG,          0x00000001),   # bypass BS
            (SDP_REG.D_DP_BN_CFG,          0x00000001),   # bypass BN
            (SDP_REG.D_DP_EW_CFG,          0x00000001),   # bypass EW
            (SDP_REG.D_FEATURE_MODE_CFG,   0x00000001),   # flying_mode=1
            (SDP_REG.D_DST_DMA_CFG,        0x00000001),   # MC
            (SDP_REG.D_DATA_FORMAT,         0x00000000),   # INT8 → INT8
            (SDP_REG.D_CVT_OFFSET,          0x00000000),
            (SDP_REG.D_CVT_SCALE,           0x00000001),   # identity: multiply by 1
            (SDP_REG.D_CVT_SHIFT,           0x00000000),

            # ── CDMA: convolution DMA ─────────────────────────────────
            (CDMA_REG.S_POINTER,          0x00000000),
            (CDMA_REG.D_MISC_CFG,         misc_cfg),
            (CDMA_REG.D_DATAIN_FORMAT,    0x00000000),     # FEATURE mode
            (CDMA_REG.D_DATAIN_SIZE_0,    datain_size_0),
            (CDMA_REG.D_DATAIN_SIZE_1,    datain_size_1),
            (CDMA_REG.D_DATAIN_SIZE_EXT_0, datain_size_0), # DC: same as SIZE_0
            (CDMA_REG.D_DAIN_RAM_TYPE,    0x00000001),     # MC
            (CDMA_REG.D_DAIN_ADDR_HIGH_0, in_hi),
            (CDMA_REG.D_DAIN_ADDR_LOW_0,  in_lo),
            (CDMA_REG.D_LINE_STRIDE,      input_line_stride),
            (CDMA_REG.D_SURF_STRIDE,      input_surf_stride),
            (CDMA_REG.D_DAIN_MAP,         0x00000000),
            (CDMA_REG.D_BATCH_NUMBER,     0x00000000),     # 1 batch
            (CDMA_REG.D_BATCH_STRIDE,     0x00000000),
            (CDMA_REG.D_ENTRY_PER_SLICE,  entries_per_slice_val),
            (CDMA_REG.D_FETCH_GRAIN,      0x00000000),
            (CDMA_REG.D_WEIGHT_FORMAT,    0x00000000),     # uncompressed
            (CDMA_REG.D_WEIGHT_SIZE_0,    weight_size_0),
            (CDMA_REG.D_WEIGHT_SIZE_1,    weight_size_1),
            (CDMA_REG.D_WEIGHT_RAM_TYPE,  0x00000001),     # MC
            (CDMA_REG.D_WEIGHT_ADDR_HIGH, wt_hi),
            (CDMA_REG.D_WEIGHT_ADDR_LOW,  wt_lo),
            (CDMA_REG.D_WEIGHT_BYTES,     weight_bytes),
            (CDMA_REG.D_CONV_STRIDE,      conv_stride),
            (CDMA_REG.D_ZERO_PADDING,     zero_padding),
            (CDMA_REG.D_ZERO_PADDING_VALUE, pad_value & 0xFFFF),
            (CDMA_REG.D_BANK,             bank_cfg),
            (CDMA_REG.D_NAN_FLUSH_TO_ZERO, 0x00000001),

            # ── CSC: sequence controller ──────────────────────────────
            (CSC_REG.S_POINTER,           0x00000000),
            (CSC_REG.D_MISC_CFG,          misc_cfg),
            (CSC_REG.D_DATAIN_FORMAT,     0x00000000),     # FEATURE
            (CSC_REG.D_DATAIN_SIZE_EXT_0, datain_size_ext_0),
            (CSC_REG.D_DATAIN_SIZE_EXT_1, datain_size_ext_1),
            (CSC_REG.D_BATCH_NUMBER,      0x00000000),
            (CSC_REG.D_POST_Y_EXTENSION,  0x00000000),
            (CSC_REG.D_ENTRY_PER_SLICE,   entries_per_slice_val),
            (CSC_REG.D_WEIGHT_FORMAT,     0x00000000),
            (CSC_REG.D_WEIGHT_SIZE_EXT_0, weight_size_ext_0),
            (CSC_REG.D_WEIGHT_SIZE_EXT_1, weight_size_ext_1),
            (CSC_REG.D_WEIGHT_BYTES,      weight_bytes),
            (CSC_REG.D_WMB_BYTES,         0x00000000),
            (CSC_REG.D_DATAOUT_SIZE_0,    dataout_size_0),
            (CSC_REG.D_DATAOUT_SIZE_1,    dataout_size_1),
            (CSC_REG.D_ATOMICS,           atomics_val),
            (CSC_REG.D_RELEASE,           0x00000000),
            (CSC_REG.D_CONV_STRIDE_EXT,   conv_stride),
            (CSC_REG.D_DILATION_EXT,      dilation_ext),
            (CSC_REG.D_ZERO_PADDING,      zero_padding),
            (CSC_REG.D_ZERO_PADDING_VALUE, pad_value & 0xFFFF),
            (CSC_REG.D_BANK,              bank_cfg),
            (CSC_REG.D_PRA_CFG,           0x00000001),

            # ── CMAC_A ────────────────────────────────────────────────
            (CMAC_A_REG.S_POINTER,   0x00000000),
            (CMAC_A_REG.D_MISC_CFG,  0x00000000),  # DC, INT8

            # ── CMAC_B ────────────────────────────────────────────────
            (CMAC_B_REG.S_POINTER,   0x00000000),
            (CMAC_B_REG.D_MISC_CFG,  0x00000000),  # DC, INT8

            # ── CACC: accumulator ─────────────────────────────────────
            (CACC_REG.S_POINTER,        0x00000000),
            (CACC_REG.D_MISC_CFG,       0x00000000),         # DC, INT8
            (CACC_REG.D_DATAOUT_SIZE_0, dataout_size_0),
            (CACC_REG.D_DATAOUT_SIZE_1, dataout_size_1),
            (CACC_REG.D_DATAOUT_ADDR,   0x00000000),         # internal routing
            (CACC_REG.D_BATCH_NUMBER,   0x00000000),
            (CACC_REG.D_LINE_STRIDE,    cacc_line_stride),
            (CACC_REG.D_SURF_STRIDE,    cacc_surf_stride),
            (CACC_REG.D_DATAOUT_MAP,    0x00010001),         # line+surf packed
            (CACC_REG.D_CLIP_CFG,       clip_truncate),

            # ── POLL: wait for CBUF flush before enabling ─────────────
            (CDMA_REG.S_CBUF_FLUSH_STATUS, 0x00000001, 'poll'),

            # ── ENABLE: bottom-up order ───────────────────────────────
            (SDP_REG.D_OP_ENABLE,    0x00000001),
            (CACC_REG.D_OP_ENABLE,   0x00000001),
            (CMAC_A_REG.D_OP_ENABLE, 0x00000001),
            (CMAC_B_REG.D_OP_ENABLE, 0x00000001),
            (CSC_REG.D_OP_ENABLE,    0x00000001),
            (CDMA_REG.D_OP_ENABLE,   0x00000001),
        ]

    def fullyConnected_configs(self, fc_config, input_addr=0x0, weight_addr=0x100, output_addr=0x200):
        """
        Build register list for a fully-connected layer by mapping FC params
        to convolution params and delegating to conv_configs.

        FC → Conv mapping:
            input_features  → num_channels (C), input_shape = [1, 1, C]
            output_features → num_kernels  (K)
            kernel = 1×1, stride = 1, padding = 0

        Args:
            fc_config:   dict with 'input_features', 'output_features', etc.
            input_addr:  DRAM base address of input data
            weight_addr: DRAM base address of weight data
            output_addr: DRAM base address for output result
        """
        from strategy.fc_strategy import FullyConnectedStrategy
        conv_config = FullyConnectedStrategy.fc_to_conv_config(fc_config)
        return self.conv_configs(conv_config, input_addr, weight_addr, output_addr)

    def activation_configs(self, layer_configs, src_addr=0x0, dst_addr=0x100):
        """
        Build SDP + SDP_RDMA register list for standalone activation.

        SDP operates in non-flying mode: data is read from DRAM via SDP_RDMA,
        processed through the SDP pipeline, and written back to DRAM via WDMA.

        Supported activation_type values:
            'relu'    : BS stage ReLU only
            'prelu'   : BS stage MUL in PReLU mode
            'sigmoid' : EW stage LUT (requires LUT table programming)
            'tanh'    : EW stage LUT (requires LUT table programming)
            'clamp'   : BS ALU=MAX(min) + BN ALU=MIN(max)

        Args:
            layer_configs: dict from YAML with activation parameters
            src_addr: DRAM base address of input data
            dst_addr: DRAM base address for output data
        """
        act_type = layer_configs['activation_type']

        # ── extract dimensions ────────────────────────────────────────
        input_shape = layer_configs['input_shape']
        in_h = input_shape[0]
        in_w = input_shape[1]
        channels = input_shape[2] if len(input_shape) == 3 else 1

        data_fmt = layer_configs.get('data_format', 'INT8')
        bpe = _BYTES_PER_EL.get(data_fmt, 1)
        fmt_enc = _DATA_FMT.get(data_fmt, 0)

        ATOM = 8
        atoms_per_pixel = max(1, (channels * bpe + ATOM - 1) // ATOM)
        pixel_bytes = atoms_per_pixel * ATOM

        # For activation, output shape == input shape
        out_h = in_h
        out_w = in_w

        # Memory strides
        src_line_stride    = in_w * pixel_bytes
        src_surface_stride = src_line_stride * in_h
        dst_line_stride    = out_w * pixel_bytes
        dst_surface_stride = dst_line_stride * out_h

        src_lo, src_hi = self._split_addr(src_addr)
        dst_lo, dst_hi = self._split_addr(dst_addr)

        # ── build stage config based on activation type ───────────────
        if act_type == 'relu':
            bs_cfg = 0x00000012    # alu bypass, mul bypass, relu ON
            bn_cfg = 0x00000001    # bypass
            ew_cfg = 0x00000001    # bypass
            bs_alu_cfg = 0x0
            bs_alu_val = 0x0
            bs_mul_cfg = 0x0
            bs_mul_val = 0x0
            bn_alu_cfg = 0x0
            bn_alu_val = 0x0
            cvt_offset = 0
            cvt_scale  = 1
            cvt_shift  = 0

        elif act_type == 'prelu':
            leak_float = layer_configs['leak_factor']
            mul_shift  = layer_configs.get('prelu_shift', 8)
            from strategy.activation_strategy import ActivationStrategy
            leak_quant, _ = ActivationStrategy.quantize_leak_factor(
                leak_float, mul_shift
            )
            # BS: alu bypass, mul ON, prelu=1, relu bypass
            # bit[0]=0(active), bit[1]=1(alu_bypass), bit[4]=0(mul_on),
            # bit[5]=1(prelu), bit[6]=1(relu bypass)
            bs_cfg = 0x00000062
            bn_cfg = 0x00000001
            ew_cfg = 0x00000001
            bs_alu_cfg = 0x0
            bs_alu_val = 0x0
            bs_mul_cfg = (mul_shift & 0xFF) << 8   # src=REG, shift
            bs_mul_val = leak_quant & 0xFFFF
            bn_alu_cfg = 0x0
            bn_alu_val = 0x0
            cvt_offset = 0
            cvt_scale  = 1
            cvt_shift  = 0

        elif act_type == 'clamp':
            clamp_min = layer_configs['clamp_min']
            clamp_max = layer_configs['clamp_max']
            # BS: alu=MAX(algo=0), alu_bypass=0, mul_bypass=1, relu_bypass=1
            # bit[0]=0, bit[1]=0(alu ON), bit[3:2]=00(MAX), bit[4]=1(mul byp),
            # bit[6]=1(relu byp) → 0x50
            bs_cfg = 0x00000050
            # BN: alu=MIN(algo=1), alu_bypass=0, mul_bypass=1, relu_bypass=1
            # bit[0]=0, bit[1]=0, bit[3:2]=01(MIN), bit[4]=1(mul byp),
            # bit[6]=1(relu byp) → 0x54
            bn_cfg = 0x00000054
            ew_cfg = 0x00000001
            bs_alu_cfg = 0x0   # src=REG
            bs_alu_val = clamp_min & 0xFFFF
            bs_mul_cfg = 0x0
            bs_mul_val = 0x0
            bn_alu_cfg = 0x0   # src=REG
            bn_alu_val = clamp_max & 0xFFFF
            cvt_offset = 0
            cvt_scale  = 1
            cvt_shift  = 0

        elif act_type in ('sigmoid', 'tanh'):
            # LUT-based: all stages bypass except EW with LUT enabled
            bs_cfg = 0x00000001
            bn_cfg = 0x00000001
            # EW: active, alu bypass, mul bypass, LUT ON
            ew_cfg = 0x00000012
            bs_alu_cfg = 0x0
            bs_alu_val = 0x0
            bs_mul_cfg = 0x0
            bs_mul_val = 0x0
            bn_alu_cfg = 0x0
            bn_alu_val = 0x0
            cvt_offset = layer_configs.get('cvt_offset', 0)
            cvt_scale  = layer_configs.get('cvt_scale', 1)
            cvt_shift  = layer_configs.get('cvt_shift', 0)
        else:
            raise ValueError(f"Unsupported activation: {act_type}")

        # ── build LUT register writes (for sigmoid/tanh) ─────────────
        lut_regs = []
        if act_type in ('sigmoid', 'tanh'):
            from strategy.activation_strategy import ActivationStrategy
            frac_bits = layer_configs.get('lut_frac_bits', 15)

            if act_type == 'sigmoid':
                x_min = layer_configs.get('lut_range_min', -8.0)
                x_max = layer_configs.get('lut_range_max', 8.0)
                lo_table = ActivationStrategy.generate_lo_lut_sigmoid(
                    frac_bits, x_min, x_max)
                le_table = ActivationStrategy.generate_le_lut_sigmoid(
                    frac_bits, x_min, x_max)
            else:  # tanh
                x_min = layer_configs.get('lut_range_min', -4.0)
                x_max = layer_configs.get('lut_range_max', 4.0)
                lo_table = ActivationStrategy.generate_lo_lut_tanh(
                    frac_bits, x_min, x_max)
                le_table = ActivationStrategy.generate_le_lut_tanh(
                    frac_bits, x_min, x_max)

            # LUT range in RAW INTEGER units matching the input data format.
            # The hardware LUT compares the 32-bit sign-extended INT8 input
            # directly against these start/end values.  They must NOT be
            # Q-scaled — they are plain integers in the same domain as the
            # input pixels (e.g. -128 .. 128 for full INT8 coverage).
            import math
            le_start = int(x_min) & 0xFFFFFFFF   # signed → unsigned 32-bit
            le_end   = int(x_max) & 0xFFFFFFFF
            lo_start = le_start
            lo_end   = le_end

            # LO index select: step = 2^lo_idx_sel, 256 intervals
            # lo_idx_sel = floor(log2(range / 256)), minimum 0
            lo_range = int(x_max) - int(x_min)            # e.g. 256
            lo_idx_sel = max(0, int(math.log2(max(1, lo_range / 256))))

            # LE index select: step = 2^le_idx_sel, 64 intervals
            le_idx_sel = max(0, int(math.log2(max(1, lo_range / 64))))

            # Program LE table (65 entries, table_id=0)
            for i in range(65):
                cfg_val = (i & 0x3FF) | (0 << 16) | (1 << 17)  # LE, WRITE
                lut_regs.append((SDP_REG.S_LUT_ACCESS_CFG, cfg_val))
                lut_regs.append((SDP_REG.S_LUT_ACCESS_DATA,
                                 int(le_table[i]) & 0xFFFF))

            # Program LO table (257 entries, table_id=1)
            for i in range(257):
                cfg_val = (i & 0x3FF) | (1 << 16) | (1 << 17)  # LO, WRITE
                lut_regs.append((SDP_REG.S_LUT_ACCESS_CFG, cfg_val))
                lut_regs.append((SDP_REG.S_LUT_ACCESS_DATA,
                                 int(lo_table[i]) & 0xFFFF))

            # LUT configuration registers
            # S_LUT_CFG: le_function=1(LINEAR), priorities=LO for all
            lut_regs.append((SDP_REG.S_LUT_CFG,
                             0x00000001 | (1 << 4) | (1 << 5) | (1 << 6)))
            # S_LUT_INFO: le_index_offset[7:0], le_index_select[15:8],
            #             lo_index_select[23:16]
            lut_regs.append((SDP_REG.S_LUT_INFO,
                             (0 & 0xFF) |
                             ((le_idx_sel & 0xFF) << 8) |
                             ((lo_idx_sel & 0xFF) << 16)))
            # Ranges
            lut_regs.append((SDP_REG.S_LUT_LE_START, le_start))
            lut_regs.append((SDP_REG.S_LUT_LE_END,   le_end))
            lut_regs.append((SDP_REG.S_LUT_LO_START, lo_start))
            lut_regs.append((SDP_REG.S_LUT_LO_END,   lo_end))
            # Slopes: 0 for saturating functions (sigmoid/tanh)
            lut_regs.append((SDP_REG.S_LUT_LE_SLOPE_SCALE, 0x00000000))
            lut_regs.append((SDP_REG.S_LUT_LE_SLOPE_SHIFT, 0x00000000))
            lut_regs.append((SDP_REG.S_LUT_LO_SLOPE_SCALE, 0x00000000))
            lut_regs.append((SDP_REG.S_LUT_LO_SLOPE_SHIFT, 0x00000000))

        # ── SDP_RDMA feature mode ────────────────────────────────────
        # flying_mode=0 (OFF — standalone, read from DRAM)
        # in_precision=INT8, proc_precision=INT8, out_precision=INT8
        rdma_feature_cfg = 0x00000000  # flying=0, INT8 all

        # ═══════════ BUILD REGISTER LIST ═══════════════════════════════
        regs = [
            # ── GLB: interrupt config ─────────────────────────────────
            (GLB_REG.S_INTR_MASK,   0x003F03FC),  # unmask SDP_done
            (GLB_REG.S_INTR_STATUS, 0x00000000),

            # ── SDP: pointer ──────────────────────────────────────────
            (SDP_REG.S_POINTER, 0x00000000),
        ]

        # ── LUT tables and config (if sigmoid/tanh) ──────────────────
        regs.extend(lut_regs)

        # ── SDP: data cube dimensions ─────────────────────────────────
        regs.extend([
            (SDP_REG.D_DATA_CUBE_WIDTH,    out_w - 1),
            (SDP_REG.D_DATA_CUBE_HEIGHT,   out_h - 1),
            (SDP_REG.D_DATA_CUBE_CHANNEL,  channels - 1),
            # ── SDP: destination memory ────────────────────────────────
            (SDP_REG.D_DST_BASE_ADDR_LOW,  dst_lo),
            (SDP_REG.D_DST_BASE_ADDR_HIGH, dst_hi),
            (SDP_REG.D_DST_LINE_STRIDE,    dst_line_stride),
            (SDP_REG.D_DST_SURFACE_STRIDE, dst_surface_stride),
            # ── SDP: BS sub-processor ─────────────────────────────────
            (SDP_REG.D_DP_BS_CFG,          bs_cfg),
            (SDP_REG.D_DP_BS_ALU_CFG,      bs_alu_cfg),
            (SDP_REG.D_DP_BS_ALU_SRC_VALUE, bs_alu_val),
            (SDP_REG.D_DP_BS_MUL_CFG,      bs_mul_cfg),
            (SDP_REG.D_DP_BS_MUL_SRC_VALUE, bs_mul_val),
            # ── SDP: BN sub-processor ─────────────────────────────────
            (SDP_REG.D_DP_BN_CFG,          bn_cfg),
            (SDP_REG.D_DP_BN_ALU_CFG,      bn_alu_cfg),
            (SDP_REG.D_DP_BN_ALU_SRC_VALUE, bn_alu_val),
            (SDP_REG.D_DP_BN_MUL_CFG,      0x00000000),
            (SDP_REG.D_DP_BN_MUL_SRC_VALUE, 0x00000000),
            # ── SDP: EW sub-processor ─────────────────────────────────
            (SDP_REG.D_DP_EW_CFG,          ew_cfg),
            (SDP_REG.D_DP_EW_ALU_CFG,      0x00000002),   # cvt bypass
            (SDP_REG.D_DP_EW_ALU_SRC_VALUE, 0x00000000),
            (SDP_REG.D_DP_EW_ALU_CVT_OFFSET,  0x00000000),
            (SDP_REG.D_DP_EW_ALU_CVT_SCALE,   0x00000001),
            (SDP_REG.D_DP_EW_ALU_CVT_TRUNCATE, 0x00000000),
            (SDP_REG.D_DP_EW_MUL_CFG,      0x00000002),   # cvt bypass
            (SDP_REG.D_DP_EW_MUL_SRC_VALUE, 0x00000001),
            (SDP_REG.D_DP_EW_MUL_CVT_OFFSET,  0x00000000),
            (SDP_REG.D_DP_EW_MUL_CVT_SCALE,   0x00000001),
            (SDP_REG.D_DP_EW_MUL_CVT_TRUNCATE, 0x00000000),
            (SDP_REG.D_DP_EW_TRUNCATE_VALUE, 0x00000000),
            # ── SDP: feature mode (non-flying) ────────────────────────
            (SDP_REG.D_FEATURE_MODE_CFG, 0x00000000),  # flying=0 (standalone)
            (SDP_REG.D_DST_DMA_CFG,     0x00000001),   # MC DRAM
            (SDP_REG.D_DATA_FORMAT,      fmt_enc),      # INT8→INT8
            # ── SDP: output converter ─────────────────────────────────
            (SDP_REG.D_CVT_OFFSET, cvt_offset & 0xFFFFFFFF),
            (SDP_REG.D_CVT_SCALE,  cvt_scale  & 0xFFFF),
            (SDP_REG.D_CVT_SHIFT,  cvt_shift  & 0x3F),
        ])

        # ── SDP RDMA ──────────────────────────────────────────────────
        regs.extend([
            (SDP_RDMA_REG.S_POINTER,            0x00000000),
            (SDP_RDMA_REG.D_DATA_CUBE_WIDTH,    in_w - 1),
            (SDP_RDMA_REG.D_DATA_CUBE_HEIGHT,   in_h - 1),
            (SDP_RDMA_REG.D_DATA_CUBE_CHANNEL,  channels - 1),
            (SDP_RDMA_REG.D_SRC_BASE_ADDR_LOW,  src_lo),
            (SDP_RDMA_REG.D_SRC_BASE_ADDR_HIGH, src_hi),
            (SDP_RDMA_REG.D_SRC_LINE_STRIDE,    src_line_stride),
            (SDP_RDMA_REG.D_SRC_SURFACE_STRIDE, src_surface_stride),
            # Disable all sub-unit RDMAs (we use register operands)
            (SDP_RDMA_REG.D_BRDMA_CFG,  0x00000001),   # disabled
            (SDP_RDMA_REG.D_NRDMA_CFG,  0x00000001),   # disabled
            (SDP_RDMA_REG.D_ERDMA_CFG,  0x00000001),   # disabled
            # Feature mode: flying=0 (standalone), INT8
            (SDP_RDMA_REG.D_FEATURE_MODE_CFG, rdma_feature_cfg),
            (SDP_RDMA_REG.D_SRC_DMA_CFG,      0x00000001),  # MC DRAM
        ])

        # ── ENABLE: SDP_RDMA first, then SDP ──────────────────────────
        regs.extend([
            (SDP_RDMA_REG.D_OP_ENABLE, 0x00000001),
            (SDP_REG.D_OP_ENABLE,      0x00000001),
        ])

        return regs

    def normalization_configs(self, layer_configs, src_addr=0x0, dst_addr=0x100):
        """
        Build CDP + CDP_RDMA register list from a YAML layer config dict.

        CDP performs Local Response Normalization (LRN).  Output dimensions
        are identical to input dimensions (no spatial / channel reduction).

        The LUT is programmed first (LE: 65 entries, LO: 257 entries), then
        the RDMA and CDP core D_ registers.  Enable order: RDMA first, CDP last.

        Expected keys in layer_configs:
            input_shape       : [H, W, C] or [H, W]
            normalz_len       : int (3, 5, 7, or 9)
            data_format       : str  ('INT8')                   default 'INT8'
            datin_offset      : int  (signed)                   default 0
            datin_scale       : int  (signed 16-bit)            default 1
            datin_shifter     : int  (unsigned 0-31)            default 0
            datout_offset     : int  (signed 32-bit)            default 0
            datout_scale      : int  (signed 16-bit)            default 1
            datout_shifter    : int  (unsigned 0-63)            default 0
            sqsum_bypass      : bool                            default False
            mul_bypass        : bool                            default False
            lut_le_table      : list[int]  65 signed-16 values  default identity
            lut_lo_table      : list[int] 257 signed-16 values  default identity
            lut_le_start / lut_le_end                           default 0 / 0xFFFF
            lut_lo_start / lut_lo_end                           default 0 / 0xFFFF
            lut_le_function   : 'EXPONENT' | 'LINEAR'          default 'LINEAR'
            lut_le_index_offset / lut_le_index_select / lut_lo_index_select
            lut_priority      : 'LE' | 'LO'                    default 'LE'
            lut_*_slope_*_scale / shift                         default 0

        Args:
            layer_configs : dict with keys listed above
            src_addr      : source DRAM base address (input data)
            dst_addr      : destination DRAM base address (output data)
        """
        # ---------- extract config values ----------
        input_shape = layer_configs['input_shape']
        in_h = input_shape[0]
        in_w = input_shape[1]
        channels = (input_shape[2] if len(input_shape) == 3
                    else layer_configs.get('num_channels', 1))

        normalz_len = layer_configs['normalz_len']
        data_fmt    = layer_configs.get('data_format', 'INT8')

        # Normalization-length encoding: 3→0, 5→1, 7→2, 9→3
        _NORMALZ_ENC = {3: 0, 5: 1, 7: 2, 9: 3}
        normalz_enc = _NORMALZ_ENC[normalz_len]

        # Data conversion params
        datin_offset   = layer_configs.get('datin_offset', 0)
        datin_scale    = layer_configs.get('datin_scale', 1)
        datin_shifter  = layer_configs.get('datin_shifter', 0)
        datout_offset  = layer_configs.get('datout_offset', 0)
        datout_scale   = layer_configs.get('datout_scale', 1)
        datout_shifter = layer_configs.get('datout_shifter', 0)

        # Bypass flags
        sqsum_bypass = 1 if layer_configs.get('sqsum_bypass', False) else 0
        mul_bypass   = 1 if layer_configs.get('mul_bypass', False) else 0
        func_bypass  = sqsum_bypass | (mul_bypass << 1)

        # ---------- LUT tables (default: identity = 256 → 1.0 in Q8.8) ----------
        lut_le_table = layer_configs.get('lut_le_table', [256] * 65)
        lut_lo_table = layer_configs.get('lut_lo_table', [256] * 257)

        lut_le_start = layer_configs.get('lut_le_start', 0)
        lut_le_end   = layer_configs.get('lut_le_end', 0xFFFF)
        lut_lo_start = layer_configs.get('lut_lo_start', 0)
        lut_lo_end   = layer_configs.get('lut_lo_end', 0xFFFF)

        # LUT configuration register
        lut_le_function = layer_configs.get('lut_le_function', 'LINEAR')
        le_func_enc  = 1 if lut_le_function == 'LINEAR' else 0
        lut_priority = layer_configs.get('lut_priority', 'LE')
        pri_lo       = 1 if lut_priority == 'LO' else 0
        lut_cfg = (le_func_enc
                   | (pri_lo << 4)      # uflow priority
                   | (pri_lo << 5)      # oflow priority
                   | (pri_lo << 6))     # hybrid priority

        # LUT info register — index_select controls the segment size (2^select)
        # for LINEAR mode index computation: index = (x - start) >> select
        le_idx_offset = layer_configs.get('lut_le_index_offset', 0) & 0xFF
        le_idx_select = layer_configs.get('lut_le_index_select', 0) & 0xFF
        lo_idx_select = layer_configs.get('lut_lo_index_select', 0) & 0xFF
        lut_info = le_idx_offset | (le_idx_select << 8) | (lo_idx_select << 16)

        # LUT slope registers
        le_uflow_scale = layer_configs.get('lut_le_slope_uflow_scale', 0) & 0xFFFF
        le_oflow_scale = layer_configs.get('lut_le_slope_oflow_scale', 0) & 0xFFFF
        le_slope_scale = le_uflow_scale | (le_oflow_scale << 16)
        le_uflow_shift = layer_configs.get('lut_le_slope_uflow_shift', 0) & 0x1F
        le_oflow_shift = layer_configs.get('lut_le_slope_oflow_shift', 0) & 0x1F
        le_slope_shift = le_uflow_shift | (le_oflow_shift << 5)

        lo_uflow_scale = layer_configs.get('lut_lo_slope_uflow_scale', 0) & 0xFFFF
        lo_oflow_scale = layer_configs.get('lut_lo_slope_oflow_scale', 0) & 0xFFFF
        lo_slope_scale = lo_uflow_scale | (lo_oflow_scale << 16)
        lo_uflow_shift = layer_configs.get('lut_lo_slope_uflow_shift', 0) & 0x1F
        lo_oflow_shift = layer_configs.get('lut_lo_slope_oflow_shift', 0) & 0x1F
        lo_slope_shift = lo_uflow_shift | (lo_oflow_shift << 5)

        # ---------- memory layout ----------
        bpe  = _BYTES_PER_EL.get(data_fmt, 1)
        ATOM = 8

        # Per-pixel: one atom holds atomic_m channels (INT8: 8 ch per atom)
        atoms_per_pixel = max(1, (channels * bpe + ATOM - 1) // ATOM)
        pixel_bytes     = atoms_per_pixel * ATOM

        # CDP output dimensions == input dimensions
        line_stride    = in_w * pixel_bytes
        surface_stride = line_stride * in_h

        # Data format encoding
        fmt_enc = _DATA_FMT[data_fmt]

        # Source / destination base addresses
        src_lo, src_hi = self._split_addr(src_addr)
        dst_lo, dst_hi = self._split_addr(dst_addr)

        # Register-width masking for data conversion fields
        datin_offset_reg   = datin_offset  & 0xFFFF
        datin_scale_reg    = datin_scale   & 0xFFFF
        datin_shifter_reg  = datin_shifter & 0x1F
        datout_offset_reg  = datout_offset & 0xFFFFFFFF
        datout_scale_reg   = datout_scale  & 0xFFFF
        datout_shifter_reg = datout_shifter & 0x3F

        # ========================= BUILD REGISTER LIST =========================
        regs = []

        # -------- CDP CORE: Status / Pointer --------
        regs.append((CDP_REG.S_STATUS,  0x00000000))
        regs.append((CDP_REG.S_POINTER, 0x00000000))   # Group 0

        # -------- CDP CORE: LUT Programming (S_ regs, NOT dual-grouped) --------
        # Write LE table (65 entries): table=LE(0), type=WRITE(1), addr=0
        regs.append((CDP_REG.S_LUT_ACCESS_CFG, 0x20000))   # (1<<17)|(0<<16)|0
        for entry in lut_le_table:
            regs.append((CDP_REG.S_LUT_ACCESS_DATA, entry & 0xFFFF))

        # Write LO table (257 entries): table=LO(1), type=WRITE(1), addr=0
        regs.append((CDP_REG.S_LUT_ACCESS_CFG, 0x30000))   # (1<<17)|(1<<16)|0
        for entry in lut_lo_table:
            regs.append((CDP_REG.S_LUT_ACCESS_DATA, entry & 0xFFFF))

        # -------- CDP CORE: LUT Configuration --------
        regs.append((CDP_REG.S_LUT_CFG,  lut_cfg))
        regs.append((CDP_REG.S_LUT_INFO, lut_info))

        # LE table range (38-bit signed, split low/high)
        regs.append((CDP_REG.S_LUT_LE_START_LOW,  lut_le_start & 0xFFFFFFFF))
        regs.append((CDP_REG.S_LUT_LE_START_HIGH, (lut_le_start >> 32) & 0x3F))
        regs.append((CDP_REG.S_LUT_LE_END_LOW,    lut_le_end   & 0xFFFFFFFF))
        regs.append((CDP_REG.S_LUT_LE_END_HIGH,   (lut_le_end  >> 32) & 0x3F))

        # LO table range
        regs.append((CDP_REG.S_LUT_LO_START_LOW,  lut_lo_start & 0xFFFFFFFF))
        regs.append((CDP_REG.S_LUT_LO_START_HIGH, (lut_lo_start >> 32) & 0x3F))
        regs.append((CDP_REG.S_LUT_LO_END_LOW,    lut_lo_end   & 0xFFFFFFFF))
        regs.append((CDP_REG.S_LUT_LO_END_HIGH,   (lut_lo_end  >> 32) & 0x3F))

        # Slope registers
        regs.append((CDP_REG.S_LUT_LE_SLOPE_SCALE, le_slope_scale))
        regs.append((CDP_REG.S_LUT_LE_SLOPE_SHIFT, le_slope_shift))
        regs.append((CDP_REG.S_LUT_LO_SLOPE_SCALE, lo_slope_scale))
        regs.append((CDP_REG.S_LUT_LO_SLOPE_SHIFT, lo_slope_shift))

        # -------- CDP CORE: D_ registers --------
        regs.append((CDP_REG.D_FUNC_BYPASS, func_bypass))

        # Destination memory
        regs.append((CDP_REG.D_DST_BASE_ADDR_LOW,  dst_lo))
        regs.append((CDP_REG.D_DST_BASE_ADDR_HIGH, dst_hi))
        regs.append((CDP_REG.D_DST_LINE_STRIDE,    line_stride))
        regs.append((CDP_REG.D_DST_SURFACE_STRIDE, surface_stride))
        regs.append((CDP_REG.D_DST_DMA_CFG,        0x00000001))   # MC

        # Data format (MUST match RDMA — C5) and NaN handling
        regs.append((CDP_REG.D_DATA_FORMAT,        fmt_enc))
        regs.append((CDP_REG.D_NAN_FLUSH_TO_ZERO,  0x00000000))

        # LRN normalization length
        regs.append((CDP_REG.D_LRN_CFG, normalz_enc))

        # Input data conversion
        regs.append((CDP_REG.D_DATIN_OFFSET,  datin_offset_reg))
        regs.append((CDP_REG.D_DATIN_SCALE,   datin_scale_reg))
        regs.append((CDP_REG.D_DATIN_SHIFTER, datin_shifter_reg))

        # Output data conversion
        regs.append((CDP_REG.D_DATOUT_OFFSET,  datout_offset_reg))
        regs.append((CDP_REG.D_DATOUT_SCALE,   datout_scale_reg))
        regs.append((CDP_REG.D_DATOUT_SHIFTER, datout_shifter_reg))

        # Performance / Debug
        regs.append((CDP_REG.D_PERF_ENABLE, 0x00000000))
        regs.append((CDP_REG.D_CYA,         0x00000000))

        # ======== CDP RDMA: Status / Pointer ========
        regs.append((CDP_RDMA_REG.S_STATUS,  0x00000000))
        regs.append((CDP_RDMA_REG.S_POINTER, 0x00000000))   # Group 0

        # ======== CDP RDMA: Data Cube Dimensions (value-1 encoding) ========
        regs.append((CDP_RDMA_REG.D_DATA_CUBE_WIDTH,   in_w     - 1))
        regs.append((CDP_RDMA_REG.D_DATA_CUBE_HEIGHT,  in_h     - 1))
        regs.append((CDP_RDMA_REG.D_DATA_CUBE_CHANNEL, channels - 1))

        # ======== CDP RDMA: Source Memory ========
        regs.append((CDP_RDMA_REG.D_SRC_BASE_ADDR_LOW,  src_lo))
        regs.append((CDP_RDMA_REG.D_SRC_BASE_ADDR_HIGH, src_hi))
        regs.append((CDP_RDMA_REG.D_SRC_LINE_STRIDE,    line_stride))
        regs.append((CDP_RDMA_REG.D_SRC_SURFACE_STRIDE, surface_stride))
        regs.append((CDP_RDMA_REG.D_SRC_DMA_CFG,        0x00000001))   # MC

        # ======== CDP RDMA: Data Format ========
        regs.append((CDP_RDMA_REG.D_DATA_FORMAT, fmt_enc))

        # ======== CDP RDMA: Performance / Debug ========
        regs.append((CDP_RDMA_REG.D_PERF_ENABLE, 0x00000000))
        regs.append((CDP_RDMA_REG.D_CYA,         0x00000000))

        # ======== ENABLE (RDMA first, then CDP — C14) ========
        regs.append((CDP_RDMA_REG.D_OP_ENABLE, 0x00000001))
        regs.append((CDP_REG.D_OP_ENABLE,      0x00000001))

        return regs

    def regularization_configs(self):
        pass