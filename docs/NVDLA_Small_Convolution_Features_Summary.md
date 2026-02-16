# NVDLA Small (nv_small) — Convolution Pipeline Feature Summary

> **Generated from RTL analysis** of the actual nv_small RTL source code and cross-referenced
> with the official NVDLA v1 documentation at https://nvdla.org/hw/v1/ias/unit_description.html

---

## 1. Convolution Pipeline Architecture

The convolution pipeline consists of **5 stages**, each with its own CSB register interface:

| Stage | Module | Base Address | Role |
|-------|--------|-------------|------|
| **CDMA** | Convolution DMA | `0x3000` | Fetches input data & weights from DRAM into CBUF |
| **CBUF** | Convolution Buffer | *(no register interface)* | 128 KB SRAM buffer for data + weights |
| **CSC** | Convolution Sequence Controller | `0x4000` (shifted: `0x6000`) | Reads CBUF, schedules and feeds MAC operations |
| **CMAC** | Convolution MAC (A + B) | `0x5000` / `0x6000` (shifted: `0x7000`/`0x8000`) | Multiply-accumulate engine |
| **CACC** | Convolution Accumulator | `0x7000` (shifted: `0x9000`) | Accumulates partial sums, truncates, sends to SDP |

The output of CACC flows into **SDP** (Single Data Processor) which handles bias addition,
activation, batch normalization, and writes the final result to DRAM. For convolution,
SDP operates in **on-flying (passthrough) mode**.

---

## 2. Hardware Configuration (nv_small Specific)

These values are baked into the generated RTL and confirmed via **CFGROM** registers:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| **atomC** | **8** | Atomic channel size (elements processed per MAC cell per cycle) |
| **atomK** | **8** | Atomic kernel size (output channels per MAC cell group) |
| **atomM** | **8** | Atomic memory size |
| **CBUF banks** | **32** | Number of SRAM banks in convolution buffer |
| **CBUF entry width** | **8 bytes** (64 bits) | Width of each SRAM entry |
| **CBUF entries/bank** | **512** | Entries per bank |
| **Total CBUF size** | **128 KB** | 32 × 512 × 8 bytes |
| **MAC cells per CMAC** | **4** | (atomC / 2 = 8 / 2 = 4 cells) |
| **Total CMACs** | **2** | CMAC_A + CMAC_B |
| **INT8 MACs/cycle** | **64** | 2 CMACs × 4 cells × 8 elements |
| **DBBIF bus width** | **64-bit** | External memory data bus |
| **AXI burst length** | **4-bit** | AXI3-style, max burst 16 |
| **SRAMIF** | **Not present** | Only MCIF (DRAM) interface available |
| **DesignWare IPs** | **Disabled** | Uses simple `$signed()` multipliers |
| **CC_ATOMC_DIV_ATOMK** | **1** | atomC/atomK ratio (equal sizes) |

### Comparison with nv_large/nv_full

| Feature | **nv_small (this RTL)** | **nv_large/nv_full** |
|---------|------------------------|---------------------|
| atomC | 8 | 64 |
| atomK | 8 | 16–32 |
| MAC cells/CMAC | 4 | 16 |
| Total INT8 MACs/cycle | 64 | 2048+ |
| Precision | INT8 only | INT8 + INT16 + FP16 |
| CBUF entry width | 64-bit | 512–1024 bit |
| CBUF total | 128 KB | 256–512 KB |
| DBBIF bus | 64-bit | 256–512 bit |
| SRAMIF | Not present | Present |

---

## 3. Supported Precision: **INT8 ONLY**

| Evidence | Finding |
|----------|---------|
| CACC calculator | Only `NV_NVDLA_CACC_CALC_int8.v` exists — no FP16 or INT16 calculator |
| CMAC MAC cells | 8-bit × 8-bit multiplication → 18-bit result, sum → 19-bit |
| CFGROM FEATURE_TYPES | `0x10` (INT8 encoding) |
| CFGROM WEIGHT_TYPES | `0x10` (INT8 encoding) |

The register fields `in_precision` and `proc_precision` accept INT8/INT16/FP16 encodings,
but **only INT8 has actual datapath hardware in this build**. Always set both to `0x0` (INT8).

---

## 4. Supported Convolution Modes

### 4.1 Direct Convolution (DC Mode) ✅ SUPPORTED

- **Primary mode** for feature-to-feature convolution layers
- `D_MISC_CFG.CONV_MODE = 0` (DIRECT)
- `D_DATAIN_FORMAT = 0` (FEATURE)
- Supports arbitrary kernel sizes, strides, padding, and dilation
- This is the mode to **use first for simple convolution tests**

**DC Operation Hierarchy:**
1. **Atomic Operation** — 1 cycle: each MAC cell processes a 1×1×8 data cube × 1×1×8 weight cube
2. **Stripe Operation** — 16–32 atomic ops scanning across W/H dimensions
3. **Block Operation** — Multiple stripes covering R×S kernel dimensions
4. **Channel Operation** — ⌈C/8⌉ block operations across input channels
5. **Group Operation** — Multiple channel ops covering all output spatial points

### 4.2 Winograd Mode ✅ RTL PRESENT

- All `NV_NVDLA_CDMA_wg.v` and related Winograd modules are present in RTL
- `D_MISC_CFG.CONV_MODE = 1` (WINOGRAD)
- Only for **3×3 kernels** with stride 1
- 2.25× performance improvement over DC for qualifying layers
- PRA (Pre-Addition) in CSC transforms input data
- POA (Post-Addition) in CMAC transforms output
- **Constraint**: Output width and height must be divisible by 4
- ⚠️ More complex to configure; **start with DC mode first**

### 4.3 Image Input Mode ✅ RTL PRESENT

- `D_DATAIN_FORMAT = 1` (PIXEL)
- For the **first layer** of a network where input is raw image pixels
- Supports many pixel formats (ARGB, RGB, YUV, etc.) — see `PIXEL_FORMAT` field
- Channel pre-extension and post-extension for MAC utilization
- **Cannot be combined** with Winograd or multi-batch
- Pixel data mapped differently in CBUF than feature data

### 4.4 Multi-Batch Mode ✅ SUPPORTED (DC only)

- `D_BATCH_NUMBER` register: 5-bit field, supports 1–32 batches
- Improves MAC efficiency for FC-like layers where weights are reused
- Each batch has its own input data at `base_addr + batch_index × batch_stride`
- Not available for Winograd or Image Input modes

### 4.5 Deconvolution ✅ SUPPORTED (SW feature)

- Not a separate HW mode — implemented by software as:
  1. Multiple convolution layers with reversed kernels
  2. RUBIK contract operation to reorder output
- Fully supported since the conv pipeline + RUBIK are both present

---

## 5. Supported Convolution Features

### 5.1 Stride ✅
- **Register**: CDMA `D_CONV_STRIDE` and CSC `D_CONV_STRIDE_EXT`
- **Range**: 1–8 in both X and Y (3-bit field, value-1 encoding)
- `D_CONV_STRIDE[2:0]` = X stride - 1, `D_CONV_STRIDE[18:16]` = Y stride - 1

### 5.2 Zero Padding ✅
- **Register**: CDMA `D_ZERO_PADDING` and CSC `D_ZERO_PADDING`
- **Range**: 0–31 for left/top (5-bit fields)
- CDMA also has right/bottom padding (auto-computed from input size + padding + kernel)
- **Pad value**: Configurable via `D_ZERO_PADDING_VALUE` (16-bit signed)

### 5.3 Dilation ✅
- **Register**: CSC `D_DILATION_EXT`
- **Range**: 1–32 in both X and Y (5-bit fields)
- `D_DILATION_EXT[4:0]` = X dilation - 1, `D_DILATION_EXT[20:16]` = Y dilation - 1
- **Only for DC mode** — not available for Winograd or Image Input

### 5.4 Weight Compression ✅
- **Register**: CDMA `D_WEIGHT_FORMAT`
- Supports both **uncompressed** (dense) and **compressed** (sparse) weights
- Compressed weights require WMB (Weight Mask Bits) and WGS (Weight Group Status) data
- **Start with uncompressed for simplicity**

### 5.5 Data/Weight Reuse ✅
- **Register**: `D_MISC_CFG` bits 16 (DATA_REUSE) and 20 (WEIGHT_REUSE)
- When enabled, reuses data/weights already in CBUF from previous operation
- Must also set SKIP_DATA_RLS / SKIP_WEIGHT_RLS in the previous layer

### 5.6 Mean Subtraction ✅
- **Register**: CDMA `D_MEAN_FORMAT`, `D_MEAN_GLOBAL_0/1`
- Per-channel mean subtraction in CDMA (for preprocessing)
- Typically used with Image Input mode for first layer

### 5.7 CVT (Convert) Pipeline ✅
- **Register**: CDMA `D_CVT_CFG`, `D_CVT_OFFSET`, `D_CVT_SCALE`
- Input data conversion: `output = (input + offset) * scale >> truncate`
- Used for input normalization/quantization

### 5.8 Channel Post-Extension ✅ (Image Input only)
- **Register**: CSC `D_POST_Y_EXTENSION`
- Improves MAC utilization when input channels < 8
- Supports 2-line or 4-line extension

### 5.9 CACC Clip/Truncate ✅
- **Register**: CACC `D_CLIP_CFG`
- 5-bit truncate value: right-shift accumulator result before sending to SDP
- Critical for INT8 output precision

---

## 6. Register Programming Flow (for a Simple DC Convolution)

The 5-stage pipeline requires **synchronized configuration**. Many fields must match
across modules. The programming order (from reference test) is:

```
1. Initialize DRAM with input data and weights
2. Configure GLB interrupt mask
3. Configure SDP (passthrough mode for simple conv)
4. Configure CDMA (data source, weight source, dimensions, etc.)
5. Configure CSC (must match CDMA settings + add dilation/weight kernel info)
6. Configure CMAC_A and CMAC_B (just conv_mode and precision)
7. Configure CACC (output dimensions, clip truncate, output address)
8. Poll CDMA.S_CBUF_FLUSH_STATUS for ready
9. Enable operations in order: SDP → CACC → CMAC_A → CMAC_B → CSC → CDMA
10. Wait for SDP interrupt (completion)
```

### Fields That MUST Match Across Modules

| Field | CDMA | CSC | CMAC_A | CMAC_B | CACC |
|-------|------|-----|--------|--------|------|
| `conv_mode` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `proc_precision` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `in_precision` | ✅ | ✅ | — | — | — |
| `weight_format` | ✅ | ✅ | — | — | — |
| `datain_format` | ✅ | ✅ | — | — | — |
| `batch_number` | ✅ | ✅ | — | — | ✅ |
| `entry_per_slice` | ✅ | ✅ | — | — | — |
| `bank allocation` | ✅ | ✅ | — | — | — |
| `weight_bytes` | ✅ | ✅ | — | — | — |
| `wmb_bytes` | ✅ | ✅ | — | — | — |

---

## 7. Memory Layout for nv_small

### Input Feature Data (DC Mode)
- Stored as **W × H × C** with channel-last (surface) layout
- Each "surface" holds up to **atomC=8** channels
- **Surface stride** = W × H × bytes_per_element (for 1 surface of 8 channels)
- **Line stride** = W × bytes_per_element × ceil(C / 8) (depends on packing)
- For INT8 with C ≤ 8: each pixel = 8 bytes (1 atom)

### Weight Data (DC Mode, Uncompressed)
- Stored as **K kernels**, each of size **R × S × C** elements
- Byte per kernel = R × S × C × 1 (INT8)
- **Must be padded** to atomC boundary: each kernel padded to next 8-byte boundary
- Total weight bytes = num_kernels × padded_byte_per_kernel

### Output Data
- Written through SDP to DRAM
- Same surface layout as feature data
- Surface stride and line stride configured in CACC

---

## 8. What's Needed for a Simple First Convolution Test

### Simplest Test Case: 1×1 Convolution (like the reference dc_1x1x8 test)
- **Input**: 1×1×8 (1 pixel, 8 channels, fits in 1 atom)
- **Kernel**: 1×1, 8 input channels, 1 output kernel
- **Output**: 1×1×1
- **No padding, no stride, no dilation**

### Recommended Next Test: Small 3×3 Convolution
- **Input**: 4×4×8 or 8×8×8 feature map
- **Kernel**: 3×3, 8 input channels, 1–8 output kernels
- **Stride**: 1
- **Padding**: 0 or 1
- **Output**: 2×2×K or 6×6×K (depending on padding)

### Key Configuration Steps for Simple DC INT8 Conv:
1. **CDMA**: Set `conv_mode=DC`, `in_precision=INT8`, `proc_precision=INT8`, `datain_format=FEATURE`
2. **CDMA**: Configure input dimensions (W-1, H-1, C-1), data address, line/surface stride
3. **CDMA**: Configure weight address, bytes_per_kernel, num_kernels, weight_bytes
4. **CDMA**: Set padding=0, stride=1, bank allocation
5. **CSC**: Mirror CDMA settings + set weight kernel W/H, dilation=0, output dimensions
6. **CMAC_A/B**: Set `conv_mode=DC`, `proc_precision=INT8`
7. **CACC**: Set output dimensions, clip_truncate, output destination
8. **SDP**: Configure for passthrough (write result directly to DRAM)
9. **Enable pipeline** bottom-up: SDP → CACC → CMAC_A → CMAC_B → CSC → CDMA

---

## 9. Integration Steps for Framework

To add convolution support to the existing PyUVM/cocotb framework:

### New Files Needed:
1. **`strategy/convolution_strategy.py`** — `ConvolutionStrategy(LayerStrategy)`:
   - `get_layer_type()` → `"convolution"`
   - `generate_input_data()` — random INT8 input feature data + weights
   - `compute_golden()` — NumPy/PyTorch convolution for golden reference

2. **Update `strategy/Layers_regs_addresses.py`** — Add register address classes:
   - `CDMA_REG` (base `0x0C00`, 39 registers)
   - `CSC_REG` (base `0x1000`, 26 registers)  (Note: shifted addresses used in the CSB interface, not raw 0x3000/0x4000 etc.)
   - `CMAC_A_REG` (base `0x1400`)
   - `CMAC_B_REG` (base `0x1800`)
   - `CACC_REG` (base `0x1C00`)
   - `SDP_REG` (for passthrough configuration)
   - `GLB_REG` (for interrupt mask)

3. **Update `strategy/regs_configs.py`** — Implement `conv_configs()`:
   - Takes YAML config → produces list of `(addr, value)` register writes
   - Handles all CDMA/CSC/CMAC/CACC/SDP register programming

4. **Update `strategy/Layer_Factory.py`** — Register `'convolution': ConvolutionStrategy`

5. **YAML config files** — Create convolution test configs under `yaml/`

### Notes on Address Mapping:
The NVDLA CSB interface uses **shifted addresses** (divided by 4). The register files
in RTL use 12-bit offsets. The mapping from raw addresses to CSB offsets is:
- Raw `0x3014` → CSB write address = `0x3014 >> 2` = `0x0C05`
- This is the value used in `reg_offset_wr` in the RTL
- The existing framework likely uses these shifted addresses (matching the PDP pattern)

---

## 10. Known Limitations & Caveats

1. **INT8 only** — Do not configure INT16 or FP16; hardware won't produce correct results
2. **No SRAMIF** — All data must go through MCIF (DRAM). Set all RAM_TYPE fields to MCIF (0x1)
3. **Small CBUF** — 128KB buffer limits the size of feature maps that can be buffered
4. **64-bit bus** — Lower bandwidth than full NVDLA; large layers will be DMA-bound
5. **Winograd complexity** — While RTL is present, the configuration is complex; start with DC
6. **SDP required** — Conv output always passes through SDP before reaching DRAM
7. **Register coherency** — Many fields must match across pipeline stages exactly
8. **Enable order** — Pipeline stages must be enabled bottom-up (SDP first, CDMA last)
9. **CBUF flush** — Must poll `CDMA.S_CBUF_FLUSH_STATUS` before enabling pipeline
