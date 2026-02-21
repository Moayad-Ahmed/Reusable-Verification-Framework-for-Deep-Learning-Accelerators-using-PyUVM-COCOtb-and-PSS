from pyuvm import *
from pyuvm_components.seq_item import DataTransaction, CsbTransaction
import numpy as np
import os
import yaml
from strategy.regs_configs import RegistrationConfigs
from strategy.Layer_Factory import LayerFactory
from utils.nvdla_utils import NvdlaBFM


# ══════════════════════════════════════════════════════════════════════
#  VIRTUAL SEQUENCER
# ══════════════════════════════════════════════════════════════════════

class NVDLAVirtualSequencer(uvm_sequencer):

    def __init__(self, name, parent):
        super().__init__(name, parent)
        self.data_sqr = None
        self.csb_sqr  = None


# ══════════════════════════════════════════════════════════════════════
#  SHARED BASE UTILITIES
# ══════════════════════════════════════════════════════════════════════

class NVDLASequenceBase(uvm_sequence):
    """Common helpers shared by PdpTestSequence and ConvTestSequence."""

    ATOM = 8

    @staticmethod
    def align_to_256(value):
        """Align value to 256-byte boundary, minimum 0x100."""
        return max(0x100, ((value + 255) // 256) * 256)

    def write_hex_file(self, path, byte_list):
        """Write a flat list of unsigned byte values as two-digit hex lines."""
        with open(path, 'w') as f:
            for b in byte_list:
                f.write(f"{b & 0xFF:02x}\n")

    def load_yaml_config(self):
        """Load and return the first layer config from the YAML test suite."""
        if not self.config_file or not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        with open(self.config_file, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        test_suite = yaml_data.get('test_suite', [])
        if not test_suite:
            raise ValueError("No test suite found in YAML config")
        layers = test_suite[0].get('layers', [])
        if not layers:
            raise ValueError("No layers found in test configuration")
        return layers[0].get('config', {})

    @staticmethod
    def chw_to_hwc_flat(data, channels):
        """Convert CHW numpy array to flattened HWC format."""
        if channels > 1 and data.ndim == 3:
            return np.transpose(data, (1, 2, 0)).flatten()
        return data.flatten()

# ══════════════════════════════════════════════════════════════════════
#  CONV SUB-SEQUENCES
# ══════════════════════════════════════════════════════════════════════

class DataSubSequence(uvm_sequence):
    """Sends one DataTransaction to the DataAgent sequencer."""

    def __init__(self, name, data_tx):
        super().__init__(name)
        self.data_tx = data_tx

    async def body(self):
        await self.start_item(self.data_tx)
        await self.finish_item(self.data_tx)


class CsbSubSequence(uvm_sequence):
    """Sends one CsbTransaction to the CsbAgent sequencer."""

    def __init__(self, name, csb_tx):
        super().__init__(name)
        self.csb_tx = csb_tx

    async def body(self):
        await self.start_item(self.csb_tx)
        await self.finish_item(self.csb_tx)

# ══════════════════════════════════════════════════════════════════════
#  PDP VIRTUAL SEQUENCE
# ══════════════════════════════════════════════════════════════════════

class PdpTestSequence(NVDLASequenceBase):
    """
    Virtual sequence for NVDLA PDP (pooling) tests.

    Each iteration:
        1. Generates random input and computes golden output
        2. Writes atom-padded HWC hex file to disk
        3. Sends DataTransaction → DataAgent  (loads DRAM)
        4. Sends CsbTransaction  → CsbAgent   (programs hardware)
        5. Waits for scoreboard to signal iteration complete
    """

    BPE_MAP    = {'INT8': 1, 'INT16': 2, 'FP16': 2}
    ITERATIONS = 2

    def __init__(self, name, input_file=None, config_file=None):
        super().__init__(name)
        self.input_file  = input_file
        self.config_file = config_file

    # ── helpers ───────────────────────────────────────────────────────

    @classmethod
    def compute_pixel_layout(cls, config):
        """Return (bpe, pixel_bytes) for the configured data format."""
        data_format = config.get('data_format', 'INT8')
        bpe         = cls.BPE_MAP.get(data_format, 1)
        channels    = config.get('channels', 1)
        atoms       = max(1, (channels * bpe + cls.ATOM - 1) // cls.ATOM)
        return bpe, atoms * cls.ATOM

    @classmethod
    def get_numpy_dtype(cls, config):
        fmt = config.get('data_format', 'INT8')
        if fmt == 'INT16': return np.int16
        if fmt == 'FP16':  return np.float16
        return np.int8

    def write_input_data_to_file(self, input_bytes, config):
        """Write input data in atom-padded HWC hex format."""
        bpe, pixel_bytes = self.compute_pixel_layout(config)
        input_shape = config['input_shape']
        h, w = input_shape[0], input_shape[1]
        c = input_shape[2] if len(input_shape) == 3 else 1

        if not self.input_file:
            return

        with open(self.input_file, 'w') as f:
            data_array = (
                np.array(input_bytes).reshape((h, w))
                if c == 1
                else np.array(input_bytes).reshape((c, h, w))
            )
            for row in range(h):
                for col in range(w):
                    pixel_data = (
                        [data_array[row, col]]
                        if c == 1
                        else [data_array[ch, row, col] for ch in range(c)]
                    )
                    for val in pixel_data:
                        vb = int(val).to_bytes(bpe, byteorder='little', signed=True)
                        for b in vb:
                            f.write(f"{b:02x}\n")
                    for _ in range(pixel_bytes - c * bpe):
                        f.write("00\n")

    # ── body ──────────────────────────────────────────────────────────

    async def body(self):
        strategy = LayerFactory.create_strategy('pooling')
        reg_cfg  = RegistrationConfigs()
        config   = self.load_yaml_config()
        bfm      = NvdlaBFM()
        np_dtype = self.get_numpy_dtype(config)
        _, pixel_bytes = self.compute_pixel_layout(config)

        input_shape        = config['input_shape']
        channels           = input_shape[2] if len(input_shape) == 3 else 1
        input_total_bytes  = input_shape[0] * input_shape[1] * pixel_bytes

        for i in range(self.ITERATIONS):
            # ---- Generate stimulus ----
            input_data      = strategy.generate_input_data(config)
            expected_output = strategy.nvdla_pool_adjust(input_data, config)
            input_bytes     = input_data.flatten().astype(np_dtype).tolist()
            expected_bytes  = (
                self.chw_to_hwc_flat(expected_output, channels)
                .astype(np_dtype)
                .tolist()
            )

            self.write_input_data_to_file(input_bytes, config)

            output_base = self.align_to_256(input_total_bytes)
            bpe, _      = self.compute_pixel_layout(config)

            if expected_output.ndim == 2:
                out_h, out_w = expected_output.shape
            else:
                _, out_h, out_w = expected_output.shape

            # ---- Build DataTransaction ----
            data_tx = DataTransaction(f"pdp_data_tx_{i}", strategy)
            data_tx.input_file      = self.input_file
            data_tx.input_base_addr = 0
            # weight_file stays None (pooling has no weights)

            # ---- Build CsbTransaction ----
            csb_tx = CsbTransaction(f"pdp_csb_tx_{i}", strategy)
            csb_tx.reg_configs = reg_cfg.pooling_configs(
                config, src_addr=0, dst_addr=output_base
            )
            csb_tx.output_base_addr            = output_base
            csb_tx.output_num_pixels           = out_h * out_w
            csb_tx.output_pixel_bytes          = pixel_bytes
            csb_tx.output_data_bytes_per_pixel = channels * bpe
            csb_tx.expected_output_data        = expected_bytes

            # ---- Dispatch: data first, then CSB ----
            data_sub = DataSubSequence(f"pdp_data_sub_{i}", data_tx)
            await data_sub.start(self.sequencer.data_sqr)

            csb_sub = CsbSubSequence(f"pdp_csb_sub_{i}", csb_tx)
            await csb_sub.start(self.sequencer.csb_sqr)

            # ---- Wait for scoreboard to confirm check complete ----
            await bfm.iteration_done_queue.get()
            print(f"PDP iteration {i} checked by scoreboard.")





# ══════════════════════════════════════════════════════════════════════
#  CONV VIRTUAL SEQUENCE
# ══════════════════════════════════════════════════════════════════════

class ConvTestSequence(NVDLASequenceBase):
    """
    Virtual sequence for NVDLA convolution (DC, INT8, SDP passthrough) tests.

    Each iteration:
        1. Generates random input + weights and computes golden output
        2. Writes atom-padded HWC hex files to disk (input and weights)
        3. Sends DataTransaction → DataAgent  (loads input + weights into DRAM)
        4. Sends CsbTransaction  → CsbAgent   (programs hardware registers)
        5. Waits for scoreboard to signal iteration complete
    """

    BPE        = 1   # INT8 only
    ITERATIONS = 2

    def __init__(self, name, input_file=None, weight_file=None, config_file=None):
        super().__init__(name)
        self.input_file  = input_file
        self.weight_file = weight_file
        self.config_file = config_file

    def _input_to_hwc_bytes(self, input_data):
        """
        Convert CHW int8 array to NVDLA feature-cube (surface-planar) byte list.

        NVDLA DC feature format stores data surface-by-surface:
          Surface 0 (channels 0-7):  all H×W pixels, each 8 bytes (1 atom)
          Surface 1 (channels 8-15): all H×W pixels, each 8 bytes
          ...
        Within each surface, data is row-major (W varies fastest).
        """
        C, H, W     = input_data.shape
        num_surfaces = max(1, (C * self.BPE + self.ATOM - 1) // self.ATOM)
        buf = []
        for s in range(num_surfaces):
            c_start = s * self.ATOM
            c_end   = min(c_start + self.ATOM, C)
            for h in range(H):
                for w in range(W):
                    for c in range(c_start, c_end):
                        buf.append(int(input_data[c, h, w]) & 0xFF)
                    # Pad to full atom (8 bytes)
                    buf.extend([0] * (self.ATOM - (c_end - c_start)))
        pixel_bytes = self.ATOM  # per-surface pixel size
        return buf, pixel_bytes

    async def body(self):
        strategy = LayerFactory.create_strategy('convolution')
        reg_cfg  = RegistrationConfigs()
        config   = self.load_yaml_config()
        bfm      = NvdlaBFM()

        # ---- Extract dimensions ----
        input_shape = config['input_shape']
        if len(input_shape) == 3:
            in_h, in_w, num_ch = input_shape
        else:
            in_h, in_w = input_shape
            num_ch = config.get('num_channels', 1)
        config.setdefault('num_channels', num_ch)

        num_kernels = config['num_kernels']
        kh      = config.get('kernel_h', 1)
        kw      = config.get('kernel_w', 1)
        stride_h = config.get('stride_h', 1)
        stride_w = config.get('stride_w', 1)
        pad_l   = config.get('padding_left', 0)
        pad_t   = config.get('padding_top', 0)
        dil_x   = config.get('dilation_x', 1)
        dil_y   = config.get('dilation_y', 1)

        eff_kh = (kh - 1) * dil_y + 1
        eff_kw = (kw - 1) * dil_x + 1
        out_h  = (in_h + pad_t * 2 - eff_kh) // stride_h + 1
        out_w  = (in_w + pad_l * 2 - eff_kw) // stride_w + 1

        # ---- Memory layout ----
        in_atoms  = max(1, (num_ch      * self.BPE + self.ATOM - 1) // self.ATOM)
        in_pixel  = in_atoms * self.ATOM
        in_total  = in_h * in_w * in_pixel

        out_atoms = max(1, (num_kernels * self.BPE + self.ATOM - 1) // self.ATOM)
        out_pixel_total = out_atoms * self.ATOM  # total output bytes per pixel

        # ---- DRAM address plan ----
        input_base  = 0
        weight_base = self.align_to_256(in_total)
        nkg     = (num_kernels + 7) // 8
        ncg     = (num_ch + 7) // 8
        wt_raw  = nkg * kh * kw * ncg * 8 * 8
        wt_size = max(128, ((wt_raw + 127) // 128) * 128)
        output_base = self.align_to_256(weight_base + wt_size)

        for i in range(self.ITERATIONS):
            # ---- Generate stimulus ----
            input_data  = strategy.generate_input_data(config)
            weight_data = strategy.generate_weight_data(config)
            expected    = strategy.compute_golden(input_data, config, weight_data)

            # ---- Write hex files ----
            input_bytes, _ = self._input_to_hwc_bytes(input_data)
            self.write_hex_file(self.input_file, input_bytes)

            weight_bytes = strategy.format_weights_for_nvdla(weight_data)
            self.write_hex_file(self.weight_file, weight_bytes)

            # ---- Expected output in flat byte order ----
            # For 1×1 spatial (or single output surface), surfaces are contiguous
            # in DRAM. Read pixel-by-pixel, only the valid K data bytes.
            K, OH, OW = expected.shape
            expected_list = []
            for oh in range(OH):
                for ow in range(OW):
                    for k in range(K):
                        expected_list.append(int(expected[k, oh, ow]) & 0xFF)

            # ---- Build DataTransaction ----
            data_tx = DataTransaction(f"conv_data_tx_{i}", strategy)
            data_tx.input_file       = self.input_file
            data_tx.input_base_addr  = input_base
            data_tx.weight_file      = self.weight_file
            data_tx.weight_base_addr = weight_base

            # ---- Build CsbTransaction ----
            csb_tx = CsbTransaction(f"conv_csb_tx_{i}", strategy)
            csb_tx.reg_configs = reg_cfg.conv_configs(
                config,
                input_addr  = input_base,
                weight_addr = weight_base,
                output_addr = output_base,
            )
            csb_tx.output_base_addr            = output_base
            csb_tx.output_num_pixels           = out_h * out_w
            csb_tx.output_pixel_bytes          = out_pixel_total
            csb_tx.output_data_bytes_per_pixel = num_kernels * self.BPE
            csb_tx.expected_output_data        = expected_list

            # ---- Dispatch: data first, then CSB ----
            data_sub = DataSubSequence(f"conv_data_sub_{i}", data_tx)
            await data_sub.start(self.sequencer.data_sqr)

            csb_sub = CsbSubSequence(f"conv_csb_sub_{i}", csb_tx)
            await csb_sub.start(self.sequencer.csb_sqr)

            # ---- Wait for scoreboard to confirm check complete ----
            await bfm.iteration_done_queue.get()
            print(f"Conv iteration {i} checked by scoreboard.")


# ══════════════════════════════════════════════════════════════════════
#  FC (FULLY-CONNECTED) VIRTUAL SEQUENCE
# ══════════════════════════════════════════════════════════════════════

class FcTestSequence(NVDLASequenceBase):
    """
    Virtual sequence for NVDLA fully-connected (dense) layer tests.

    FC layers are mapped to the convolution engine as 1×1 direct convolutions:
        input_features  → C channels, 1×1 spatial
        output_features → K kernels,  1×1 kernel

    Hardware pipeline: CDMA → CSC → CMAC_A/B → CACC → SDP (passthrough)

    Each iteration:
        1. Generates random input vector + weight matrix, computes golden output
        2. Writes atom-padded hex files to disk (input and weights)
        3. Sends DataTransaction → DataAgent  (loads DRAM)
        4. Sends CsbTransaction  → CsbAgent   (programs hardware)
        5. Waits for scoreboard to signal iteration complete
    """

    BPE        = 1   # INT8 only
    ITERATIONS = 2

    def __init__(self, name, input_file=None, weight_file=None, config_file=None):
        super().__init__(name)
        self.input_file  = input_file
        self.weight_file = weight_file
        self.config_file = config_file

    def _input_to_hwc_bytes(self, input_data):
        """
        Convert CHW int8 array to NVDLA feature-cube (surface-planar) byte list.
        Identical to ConvTestSequence's method.
        """
        C, H, W     = input_data.shape
        num_surfaces = max(1, (C * self.BPE + self.ATOM - 1) // self.ATOM)
        buf = []
        for s in range(num_surfaces):
            c_start = s * self.ATOM
            c_end   = min(c_start + self.ATOM, C)
            for h in range(H):
                for w in range(W):
                    for c in range(c_start, c_end):
                        buf.append(int(input_data[c, h, w]) & 0xFF)
                    buf.extend([0] * (self.ATOM - (c_end - c_start)))
        pixel_bytes = self.ATOM
        return buf, pixel_bytes

    async def body(self):
        from strategy.fc_strategy import FullyConnectedStrategy

        strategy = LayerFactory.create_strategy('fully_connected')
        reg_cfg  = RegistrationConfigs()
        fc_config = self.load_yaml_config()
        bfm      = NvdlaBFM()

        # ---- Translate FC → Conv dimensions ----
        conv_config     = FullyConnectedStrategy.fc_to_conv_config(fc_config)
        input_features  = fc_config['input_features']
        output_features = fc_config['output_features']

        # Spatial is always 1×1 for FC
        in_h, in_w = 1, 1
        out_h, out_w = 1, 1
        num_ch      = input_features
        num_kernels = output_features

        # ---- Memory layout ----
        in_atoms  = max(1, (num_ch      * self.BPE + self.ATOM - 1) // self.ATOM)
        in_pixel  = in_atoms * self.ATOM
        in_total  = in_h * in_w * in_pixel

        out_atoms = max(1, (num_kernels * self.BPE + self.ATOM - 1) // self.ATOM)
        out_pixel_total = out_atoms * self.ATOM

        # ---- DRAM address plan ----
        input_base  = 0
        weight_base = self.align_to_256(in_total)
        nkg     = (num_kernels + 7) // 8
        ncg     = (num_ch + 7) // 8
        wt_raw  = nkg * 1 * 1 * ncg * 8 * 8   # 1×1 kernel
        wt_size = max(128, ((wt_raw + 127) // 128) * 128)
        output_base = self.align_to_256(weight_base + wt_size)

        for i in range(self.ITERATIONS):
            # ---- Generate stimulus ----
            input_data  = strategy.generate_input_data(fc_config)
            weight_data = strategy.generate_weight_data(fc_config)
            expected    = strategy.compute_golden(input_data, fc_config, weight_data)

            # ---- Write hex files ----
            input_bytes, _ = self._input_to_hwc_bytes(input_data)
            self.write_hex_file(self.input_file, input_bytes)

            weight_bytes = strategy.format_weights_for_nvdla(weight_data)
            self.write_hex_file(self.weight_file, weight_bytes)

            # ---- Expected output in flat byte order ----
            K, OH, OW = expected.shape
            expected_list = []
            for oh in range(OH):
                for ow in range(OW):
                    for k in range(K):
                        expected_list.append(int(expected[k, oh, ow]) & 0xFF)

            # ---- Build DataTransaction ----
            data_tx = DataTransaction(f"fc_data_tx_{i}", strategy)
            data_tx.input_file       = self.input_file
            data_tx.input_base_addr  = input_base
            data_tx.weight_file      = self.weight_file
            data_tx.weight_base_addr = weight_base

            # ---- Build CsbTransaction (uses conv registers via FC mapping) ----
            csb_tx = CsbTransaction(f"fc_csb_tx_{i}", strategy)
            csb_tx.reg_configs = reg_cfg.fullyConnected_configs(
                fc_config,
                input_addr  = input_base,
                weight_addr = weight_base,
                output_addr = output_base,
            )
            csb_tx.output_base_addr            = output_base
            csb_tx.output_num_pixels           = out_h * out_w
            csb_tx.output_pixel_bytes          = out_pixel_total
            csb_tx.output_data_bytes_per_pixel = num_kernels * self.BPE
            csb_tx.expected_output_data        = expected_list

            # ---- Dispatch: data first, then CSB ----
            data_sub = DataSubSequence(f"fc_data_sub_{i}", data_tx)
            await data_sub.start(self.sequencer.data_sqr)

            csb_sub = CsbSubSequence(f"fc_csb_sub_{i}", csb_tx)
            await csb_sub.start(self.sequencer.csb_sqr)

            # ---- Wait for scoreboard to confirm check complete ----
            await bfm.iteration_done_queue.get()
            print(f"FC iteration {i} checked by scoreboard.")

# ══════════════════════════════════════════════════════════════════════
#  CDP (NORMALIZATION / LRN) VIRTUAL SEQUENCE
# ══════════════════════════════════════════════════════════════════════

class CdpTestSequence(NVDLASequenceBase):
    """
    Virtual sequence for NVDLA CDP (Channel Data Processor / LRN) tests.

    CDP performs Local Response Normalization across channels using a
    LUT-based pipeline:  RDMA → CvtIn → SqSum → LUT → Mul → CvtOut → WDMA

    Output dimensions are identical to input dimensions (LRN does not
    change spatial or channel shapes).

    Each iteration:
        1. Generates random input and computes golden output via NVDLA
           CDP pipeline model (compute_golden_nvdla)
        2. Generates LRN LUT tables from (k, alpha, beta) parameters
        3. Writes atom-padded surface-planar hex file to disk
        4. Sends DataTransaction → DataAgent  (loads DRAM)
        5. Sends CsbTransaction  → CsbAgent   (programs hardware + LUT)
        6. Waits for scoreboard to signal iteration complete
    """

    BPE        = 1   # INT8 only
    ITERATIONS = 2

    def __init__(self, name, input_file=None, config_file=None):
        super().__init__(name)
        self.input_file  = input_file
        self.config_file = config_file

    def _input_to_hwc_bytes(self, input_data):
        """
        Convert CHW int8 array to NVDLA surface-planar byte list.

        Within each surface (group of ATOM channels), data is stored
        pixel-by-pixel, row-major.  Each pixel occupies one ATOM of
        bytes with partial channels zero-padded.
        """
        C, H, W = input_data.shape
        num_surfaces = max(1, (C * self.BPE + self.ATOM - 1) // self.ATOM)
        buf = []
        for s in range(num_surfaces):
            c_start = s * self.ATOM
            c_end   = min(c_start + self.ATOM, C)
            for h in range(H):
                for w in range(W):
                    for c in range(c_start, c_end):
                        buf.append(int(input_data[c, h, w]) & 0xFF)
                    # Pad to full atom
                    buf.extend([0] * (self.ATOM - (c_end - c_start)))
        return buf

    async def body(self):
        from strategy.normalization_strategy import NormalizationStrategy

        strategy = LayerFactory.create_strategy('normalization')
        reg_cfg  = RegistrationConfigs()
        config   = self.load_yaml_config()
        bfm      = NvdlaBFM()

        # ---- Extract dimensions ----
        input_shape = config['input_shape']
        if len(input_shape) == 3:
            in_h, in_w, channels = input_shape
        else:
            in_h, in_w = input_shape
            channels = config.get('num_channels', 1)
        config.setdefault('num_channels', channels)

        # ---- Memory layout (surface-planar) ----
        # Each surface holds one ATOM of channels for all (H,W) pixels.
        # line_stride / surface_stride are per-surface, not per-pixel-all-channels.
        num_surfaces    = max(1, (channels * self.BPE + self.ATOM - 1) // self.ATOM)
        line_stride     = in_w * self.ATOM         # per-surface row
        surface_stride  = line_stride * in_h       # size of one surface
        input_total     = num_surfaces * surface_stride

        # ---- Generate LUT tables from LRN formula parameters ----
        k     = config.get('k', 1.0)
        alpha = config.get('alpha', 1e-4)
        beta  = config.get('beta', 0.75)
        normalz_len = config['normalz_len']

        le_table, le_start, le_end, le_idx_sel = NormalizationStrategy.generate_lrn_lut(
            k, alpha, beta, normalz_len, num_entries=65
        )
        lo_table, lo_start, lo_end, lo_idx_sel = NormalizationStrategy.generate_lrn_lut(
            k, alpha, beta, normalz_len, num_entries=257
        )

        # Enrich config with LUT data (used by golden model AND reg config)
        config['lut_le_table'] = le_table
        config['lut_lo_table'] = lo_table
        config['lut_le_start'] = le_start
        config['lut_le_end']   = le_end
        config['lut_lo_start'] = lo_start
        config['lut_lo_end']   = lo_end
        config['lut_le_index_select'] = le_idx_sel
        config['lut_lo_index_select'] = lo_idx_sel

        output_base = self.align_to_256(input_total)

        for i in range(self.ITERATIONS):
            # ---- Generate stimulus ----
            input_data      = strategy.generate_input_data(config)
            expected_output = strategy.compute_golden_nvdla(input_data, config)

            # ---- Write input hex file (surface-planar, atom-padded) ----
            input_bytes = self._input_to_hwc_bytes(input_data)
            self.write_hex_file(self.input_file, input_bytes)

            # ---- Build expected byte list (surface-planar order) ----
            # Matches NVDLA DRAM layout: surface 0 (ch 0-7), surface 1 (ch 8-15), ...
            C, H, W = expected_output.shape
            expected_list = []
            for s in range(num_surfaces):
                c_start = s * self.ATOM
                c_end   = min(c_start + self.ATOM, C)
                for h in range(H):
                    for w in range(W):
                        for c in range(c_start, c_end):
                            expected_list.append(int(expected_output[c, h, w]) & 0xFF)
                        # Pad to full ATOM (for non-multiple-of-8 channels)
                        expected_list.extend([0] * (self.ATOM - (c_end - c_start)))

            # ---- Build DataTransaction (no weights for CDP) ----
            data_tx = DataTransaction(f"cdp_data_tx_{i}", strategy)
            data_tx.input_file      = self.input_file
            data_tx.input_base_addr = 0

            # ---- Build CsbTransaction ----
            csb_tx = CsbTransaction(f"cdp_csb_tx_{i}", strategy)
            csb_tx.reg_configs = reg_cfg.normalization_configs(
                config, src_addr=0, dst_addr=output_base
            )
            csb_tx.output_base_addr            = output_base
            csb_tx.output_num_pixels           = in_h * in_w * num_surfaces
            csb_tx.output_pixel_bytes          = self.ATOM
            csb_tx.output_data_bytes_per_pixel = self.ATOM
            csb_tx.expected_output_data        = expected_list

            # ---- Dispatch: data first, then CSB ----
            data_sub = DataSubSequence(f"cdp_data_sub_{i}", data_tx)
            await data_sub.start(self.sequencer.data_sqr)

            csb_sub = CsbSubSequence(f"cdp_csb_sub_{i}", csb_tx)
            await csb_sub.start(self.sequencer.csb_sqr)

            # ---- Wait for scoreboard to confirm check complete ----
            await bfm.iteration_done_queue.get()
            print(f"CDP iteration {i} checked by scoreboard.")


# ══════════════════════════════════════════════════════════════════════
#  SDP ACTIVATION SUB-SEQUENCES
# ══════════════════════════════════════════════════════════════════════

class SdpDataSubSequence(uvm_sequence):
    """Sends one DataTransaction to the DataAgent sequencer."""

    def __init__(self, name, data_tx):
        super().__init__(name)
        self.data_tx = data_tx

    async def body(self):
        await self.start_item(self.data_tx)
        await self.finish_item(self.data_tx)


class SdpCsbSubSequence(uvm_sequence):
    """Sends one CsbTransaction to the CsbAgent sequencer."""

    def __init__(self, name, csb_tx):
        super().__init__(name)
        self.csb_tx = csb_tx

    async def body(self):
        await self.start_item(self.csb_tx)
        await self.finish_item(self.csb_tx)


# ══════════════════════════════════════════════════════════════════════
#  SDP ACTIVATION VIRTUAL SEQUENCE
# ══════════════════════════════════════════════════════════════════════

class SdpTestSequence(NVDLASequenceBase):
    """
    Virtual sequence for NVDLA SDP activation (standalone, non-flying) tests.

    Pipeline: SDP_RDMA → SDP[activation] → WDMA → DRAM

    Each iteration:
        1. Generates random input and computes golden activation output
        2. Writes atom-padded HWC hex file to disk
        3. Sends DataTransaction → DataAgent  (loads DRAM)
        4. Sends CsbTransaction  → CsbAgent   (programs SDP_RDMA + SDP regs)
        5. Waits for scoreboard to signal iteration complete
    """

    BPE        = 1     # INT8 only
    ITERATIONS = 2

    def __init__(self, name, input_file=None, config_file=None):
        super().__init__(name)
        self.input_file  = input_file
        self.config_file = config_file

    def _input_to_hwc_bytes(self, input_data, channels):
        """
        Convert input array to flat HWC byte list, atom-padded per pixel.

        Args:
            input_data: np.int8 (H, W) or (C, H, W)
            channels: number of channels

        Returns:
            (byte_list, pixel_bytes) tuple
        """
        atoms       = max(1, (channels * self.BPE + self.ATOM - 1) // self.ATOM)
        pixel_bytes = atoms * self.ATOM

        if input_data.ndim == 2:
            h, w = input_data.shape
            buf = []
            for r in range(h):
                for c in range(w):
                    buf.append(int(input_data[r, c]) & 0xFF)
                    buf.extend([0] * (pixel_bytes - self.BPE))
        else:
            ch, h, w = input_data.shape
            buf = []
            for r in range(h):
                for c_idx in range(w):
                    for ch_idx in range(ch):
                        buf.append(int(input_data[ch_idx, r, c_idx]) & 0xFF)
                    buf.extend([0] * (pixel_bytes - ch * self.BPE))

        return buf, pixel_bytes

    def _expected_to_hwc_bytes(self, output_data, channels):
        """Convert output array to flat HWC unsigned byte list."""
        if output_data.ndim == 2:
            return [int(v) & 0xFF for v in output_data.flatten()]
        else:
            # (C, H, W) → HWC flat
            _, h, w = output_data.shape
            result = []
            for r in range(h):
                for c_idx in range(w):
                    for ch_idx in range(channels):
                        result.append(int(output_data[ch_idx, r, c_idx]) & 0xFF)
            return result

    async def body(self):
        strategy = LayerFactory.create_strategy('activation')
        reg_cfg  = RegistrationConfigs()
        config   = self.load_yaml_config()
        bfm      = NvdlaBFM()

        # ---- Extract dimensions ----
        input_shape = config['input_shape']
        if len(input_shape) == 3:
            in_h, in_w, channels = input_shape
        else:
            in_h, in_w = input_shape
            channels = 1

        # ---- Memory layout ----
        atoms_per_pixel = max(1, (channels * self.BPE + self.ATOM - 1) // self.ATOM)
        pixel_bytes     = atoms_per_pixel * self.ATOM
        input_total     = in_h * in_w * pixel_bytes
        output_base     = self.align_to_256(input_total)

        # Output shape == input shape for activation
        out_h, out_w = in_h, in_w

        for i in range(self.ITERATIONS):
            # ---- Generate stimulus & golden ----
            input_data = strategy.generate_input_data(config)
            expected   = strategy.compute_golden(input_data, config)

            # ---- Write input hex file ----
            input_bytes, _ = self._input_to_hwc_bytes(input_data, channels)
            self.write_hex_file(self.input_file, input_bytes)

            # ---- Expected output in HWC flat unsigned bytes ----
            expected_list = self._expected_to_hwc_bytes(expected, channels)

            # ---- Build DataTransaction ----
            data_tx = DataTransaction(f"sdp_data_tx_{i}", strategy)
            data_tx.input_file      = self.input_file
            data_tx.input_base_addr = 0
            # No weight file for activation

            # ---- Build CsbTransaction ----
            csb_tx = CsbTransaction(f"sdp_csb_tx_{i}", strategy)
            csb_tx.reg_configs = reg_cfg.activation_configs(
                config, src_addr=0, dst_addr=output_base
            )
            csb_tx.output_base_addr            = output_base
            csb_tx.output_num_pixels           = out_h * out_w
            csb_tx.output_pixel_bytes          = pixel_bytes
            csb_tx.output_data_bytes_per_pixel = channels * self.BPE
            csb_tx.expected_output_data        = expected_list

            # ---- Dispatch: data first, then CSB ----
            data_sub = SdpDataSubSequence(f"sdp_data_sub_{i}", data_tx)
            await data_sub.start(self.sequencer.data_sqr)

            csb_sub = SdpCsbSubSequence(f"sdp_csb_sub_{i}", csb_tx)
            await csb_sub.start(self.sequencer.csb_sqr)

            # ---- Wait for scoreboard to confirm check complete ----
            await bfm.iteration_done_queue.get()
            print(f"SDP activation iteration {i} checked by scoreboard.")