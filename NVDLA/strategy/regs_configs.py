import math
from strategy.Layers_regs_addresses import PDP_REG
from strategy.Layers_regs_addresses import PDP_RDMA_REG
from strategy.Layers_regs_addresses import (
    GLB_REG, CDMA_REG, CSC_REG, CMAC_A_REG, CMAC_B_REG, CACC_REG, SDP_REG
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
            (PDP_REG.D_POOLING_PADDING_VALUE_2_CFG,   pad_val_19),
            (PDP_REG.D_POOLING_PADDING_VALUE_3_CFG,   pad_val_19),
            (PDP_REG.D_POOLING_PADDING_VALUE_4_CFG,   pad_val_19),
            (PDP_REG.D_POOLING_PADDING_VALUE_5_CFG,   pad_val_19),
            (PDP_REG.D_POOLING_PADDING_VALUE_6_CFG,   pad_val_19),
            (PDP_REG.D_POOLING_PADDING_VALUE_7_CFG,   pad_val_19),

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

    def activation_configs(self):
        pass

    def normalization_configs(self):
        pass

    def regularization_configs(self):
        pass