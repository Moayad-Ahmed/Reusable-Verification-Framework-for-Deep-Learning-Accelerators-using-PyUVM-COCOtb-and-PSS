# NVDLA SDP — Complete Register Reference & Activation Configuration Guide

> **Scope**: nv_small configuration, INT8 precision, throughput k=1  
> **RTL source**: `rtl/vmod/nvdla/sdp/`, `rtl/sdp_temp/`  
> **Verification**: `strategy/Layers_regs_addresses.py` (`SDP_REG` class, base `0x2400`)

---

## Table of Contents

1. [SDP Architecture Overview](#1-sdp-architecture-overview)
2. [Register Map — SDP Single (Shared)](#2-register-map--sdp-single-shared)
3. [Register Map — SDP Dual (Double-Buffered)](#3-register-map--sdp-dual-double-buffered)
4. [Register Map — SDP RDMA](#4-register-map--sdp-rdma)
5. [Sub-Processor Pipelines (BS / BN / EW)](#5-sub-processor-pipelines-bs--bn--ew)
6. [LUT System — Deep Dive](#6-lut-system--deep-dive)
7. [Output Converter (C)](#7-output-converter-c)
8. [Hardware Limitations & Constraints](#8-hardware-limitations--constraints)
9. [Activation Function Configurations](#9-activation-function-configurations)
10. [Register Programming Recipes](#10-register-programming-recipes)
11. [Verification Testbench Integration](#11-verification-testbench-integration)

---

## 1. SDP Architecture Overview

The SDP sits between CACC (Convolution Accumulator) and WDMA/PDP. It applies post-convolution element-wise operations: bias, batch normalization, activation functions, and output format conversion.

```
                      ┌────────────────────────────────────────────┐
                      │              SDP_core                       │
   CACC ─flying──►    │                                             │
                 CMUX─►│ X1/BS ──► X2/BN ──► Y/EW ──► Output CVT ──►─── WDMA (→DRAM)
   MRDMA─memory──►    │   │          │         │                    │      │
                      │  BRDMA     NRDMA     ERDMA                 │      └── PDP
                      └────────────────────────────────────────────┘
```

**Three sub-processors** execute in series:

| Stage | Name | Purpose | Operand Width | Pipeline Inside |
|-------|------|---------|---------------|-----------------|
| **X1** | BS (Bias/Scale) | Bias addition + scale multiplication | 16-bit | ALU → MUL → TRT → ReLU |
| **X2** | BN (Batch Norm) | BatchNorm offset + scale | 16-bit | ALU → MUL → TRT → ReLU |
| **Y** | EW (Element-Wise) | Activation functions, LUT, element-wise ops | 32-bit | MUL → ALU → LUT(idx→lookup→interp) |

Each stage can be **independently bypassed**. If all three are bypassed, data flows straight from CMUX to the output converter.

---

## 2. Register Map — SDP Single (Shared)

Single registers are NOT double-buffered. They control the LUT tables and status.

> **CSB address** = `SDP_REG.<name>` (word address from `Layers_regs_addresses.py`)

### 2.1 Status & Control

| Register | Addr | Field | Bits | R/W | Description |
|----------|------|-------|------|-----|-------------|
| **S_STATUS** | `0x2400` | `status_0` | [1:0] | RO | Group 0: 0=IDLE, 1=RUNNING, 2=PENDING |
| | | `status_1` | [17:16] | RO | Group 1 status |
| **S_POINTER** | `0x2401` | `producer` | [0] | RW | Producer group select: 0=GROUP_0, 1=GROUP_1 |
| | | `consumer` | [16] | RO | Consumer group select |

### 2.2 LUT Access

| Register | Addr | Field | Bits | R/W | Description |
|----------|------|-------|------|-----|-------------|
| **S_LUT_ACCESS_CFG** | `0x2402` | `lut_addr` | [9:0] | RW | Entry address (0–1023) |
| | | `lut_table_id` | [16] | RW | **0** = LE table, **1** = LO table |
| | | `lut_access_type` | [17] | RW | **0** = READ, **1** = WRITE |
| **S_LUT_ACCESS_DATA** | `0x2403` | `lut_data` | [15:0] | RW | 16-bit LUT entry value (write triggers storage) |

### 2.3 LUT Configuration

| Register | Addr | Field | Bits | R/W | Description |
|----------|------|-------|------|-----|-------------|
| **S_LUT_CFG** | `0x2404` | `lut_le_function` | [0] | RW | LE indexing: **0** = Exponential (log₂), **1** = Linear |
| | | `lut_uflow_priority` | [4] | RW | Underflow winner: **0** = LE, **1** = LO |
| | | `lut_oflow_priority` | [5] | RW | Overflow winner: **0** = LE, **1** = LO |
| | | `lut_hybrid_priority` | [6] | RW | Both-hit winner: **0** = LE, **1** = LO |
| **S_LUT_INFO** | `0x2405` | `lut_le_index_offset` | [7:0] | RW | LE index offset (signed, for exp table start exponent) |
| | | `lut_le_index_select` | [15:8] | RW | LE index select (bit position for linear mode) |
| | | `lut_lo_index_select` | [23:16] | RW | LO index select (right-shift amount) |

### 2.4 LUT Range

| Register | Addr | Field | Bits | R/W | Description |
|----------|------|-------|------|-----|-------------|
| **S_LUT_LE_START** | `0x2406` | `lut_le_start` | [31:0] | RW | LE input range start (signed 32-bit) |
| **S_LUT_LE_END** | `0x2407` | `lut_le_end` | [31:0] | RW | LE input range end (signed 32-bit) |
| **S_LUT_LO_START** | `0x2408` | `lut_lo_start` | [31:0] | RW | LO input range start (signed 32-bit) |
| **S_LUT_LO_END** | `0x2409` | `lut_lo_end` | [31:0] | RW | LO input range end (signed 32-bit) |

### 2.5 LUT Slope (Extrapolation Outside Table Range)

| Register | Addr | Field | Bits | R/W | Description |
|----------|------|-------|------|-----|-------------|
| **S_LUT_LE_SLOPE_SCALE** | `0x240A` | `le_slope_uflow_scale` | [15:0] | RW | LE underflow slope scale |
| | | `le_slope_oflow_scale` | [31:16] | RW | LE overflow slope scale |
| **S_LUT_LE_SLOPE_SHIFT** | `0x240B` | `le_slope_uflow_shift` | [4:0] | RW | LE underflow slope shift |
| | | `le_slope_oflow_shift` | [9:5] | RW | LE overflow slope shift |
| **S_LUT_LO_SLOPE_SCALE** | `0x240C` | `lo_slope_uflow_scale` | [15:0] | RW | LO underflow slope scale |
| | | `lo_slope_oflow_scale` | [31:16] | RW | LO overflow slope scale |
| **S_LUT_LO_SLOPE_SHIFT** | `0x240D` | `lo_slope_uflow_shift` | [4:0] | RW | LO underflow slope shift |
| | | `lo_slope_oflow_shift` | [9:5] | RW | LO overflow slope shift |

---

## 3. Register Map — SDP Dual (Double-Buffered)

Dual registers use the producer/consumer pointer for double-buffering so the next layer's config can be written while the current layer executes.

### 3.1 Operation & Data Cube

| Register | Addr | Field | Bits | R/W | Description |
|----------|------|-------|------|-----|-------------|
| **D_OP_ENABLE** | `0x240E` | `op_en` | [0] | RW | **1** = start operation |
| **D_DATA_CUBE_WIDTH** | `0x240F` | `width` | [12:0] | RW | Output width − 1 (0-based) |
| **D_DATA_CUBE_HEIGHT** | `0x2410` | `height` | [12:0] | RW | Output height − 1 |
| **D_DATA_CUBE_CHANNEL** | `0x2411` | `channel` | [12:0] | RW | Output channel − 1 |

### 3.2 Destination DMA

| Register | Addr | Field | Bits | R/W | Description |
|----------|------|-------|------|-----|-------------|
| **D_DST_BASE_ADDR_LOW** | `0x2412` | `addr_low` | [31:0] | RW | Output DRAM base address (low 32) |
| **D_DST_BASE_ADDR_HIGH** | `0x2413` | `addr_high` | [31:0] | RW | Output DRAM base address (high 32) |
| **D_DST_LINE_STRIDE** | `0x2414` | `line_stride` | [31:0] | RW | Bytes between consecutive lines |
| **D_DST_SURFACE_STRIDE** | `0x2415` | `surf_stride` | [31:0] | RW | Bytes between consecutive surfaces |

### 3.3 BS (Bias/Scale) Sub-Processor — X1 Stage

| Register | Addr | Field | Bits | R/W | Description |
|----------|------|-------|------|-----|-------------|
| **D_DP_BS_CFG** | `0x2416` | `bs_bypass` | [0] | RW | **1** = bypass entire BS stage |
| | | `bs_alu_bypass` | [1] | RW | **1** = bypass ALU within BS |
| | | `bs_alu_algo` | [3:2] | RW | **0**=MAX, **1**=MIN, **2**=SUM |
| | | `bs_mul_bypass` | [4] | RW | **1** = bypass MUL within BS |
| | | `bs_mul_prelu` | [5] | RW | **1** = enable PReLU mode in MUL |
| | | `bs_relu_bypass` | [6] | RW | **1** = bypass ReLU at end of BS |
| **D_DP_BS_ALU_CFG** | `0x2417` | `bs_alu_src` | [0] | RW | **0** = register operand, **1** = memory (BRDMA) |
| | | `bs_alu_shift_value` | [13:8] | RW | Left-shift for ALU operand (0–63) |
| **D_DP_BS_ALU_SRC_VALUE** | `0x2418` | `bs_alu_operand` | [15:0] | RW | ALU operand value (16-bit signed) |
| **D_DP_BS_MUL_CFG** | `0x2419` | `bs_mul_src` | [0] | RW | **0** = register operand, **1** = memory |
| | | `bs_mul_shift_value` | [15:8] | RW | Right-shift for truncation after multiply (0–63) |
| **D_DP_BS_MUL_SRC_VALUE** | `0x241A` | `bs_mul_operand` | [15:0] | RW | MUL operand value (16-bit signed) |

### 3.4 BN (Batch Normalization) Sub-Processor — X2 Stage

| Register | Addr | Field | Bits | R/W | Description |
|----------|------|-------|------|-----|-------------|
| **D_DP_BN_CFG** | `0x241B` | `bn_bypass` | [0] | RW | **1** = bypass entire BN stage |
| | | `bn_alu_bypass` | [1] | RW | **1** = bypass ALU within BN |
| | | `bn_alu_algo` | [3:2] | RW | **0**=MAX, **1**=MIN, **2**=SUM |
| | | `bn_mul_bypass` | [4] | RW | **1** = bypass MUL within BN |
| | | `bn_mul_prelu` | [5] | RW | **1** = enable PReLU mode in MUL |
| | | `bn_relu_bypass` | [6] | RW | **1** = bypass ReLU at end of BN |
| **D_DP_BN_ALU_CFG** | `0x241C` | `bn_alu_src` | [0] | RW | **0** = register, **1** = memory (NRDMA) |
| | | `bn_alu_shift_value` | [13:8] | RW | Left-shift for ALU operand (0–63) |
| **D_DP_BN_ALU_SRC_VALUE** | `0x241D` | `bn_alu_operand` | [15:0] | RW | ALU operand (16-bit signed) |
| **D_DP_BN_MUL_CFG** | `0x241E` | `bn_mul_src` | [0] | RW | **0** = register, **1** = memory |
| | | `bn_mul_shift_value` | [15:8] | RW | Right-shift after multiply (0–63) |
| **D_DP_BN_MUL_SRC_VALUE** | `0x241F` | `bn_mul_operand` | [15:0] | RW | MUL operand (16-bit signed) |

### 3.5 EW (Element-Wise) Sub-Processor — Y Stage

| Register | Addr | Field | Bits | R/W | Description |
|----------|------|-------|------|-----|-------------|
| **D_DP_EW_CFG** | `0x2420` | `ew_bypass` | [0] | RW | **1** = bypass entire EW stage |
| | | `ew_alu_bypass` | [1] | RW | **1** = bypass ALU within EW |
| | | `ew_alu_algo` | [3:2] | RW | **0**=MAX, **1**=MIN, **2**=SUM, **3**=EQL |
| | | `ew_mul_bypass` | [4] | RW | **1** = bypass MUL within EW |
| | | `ew_mul_prelu` | [5] | RW | **1** = enable PReLU in MUL |
| | | `ew_lut_bypass` | [6] | RW | **1** = bypass LUT (no activation table) |
| **D_DP_EW_ALU_CFG** | `0x2421` | `ew_alu_src` | [0] | RW | **0** = register, **1** = memory (ERDMA) |
| | | `ew_alu_cvt_bypass` | [1] | RW | **1** = bypass ALU input cvt |
| **D_DP_EW_ALU_SRC_VALUE** | `0x2422` | `ew_alu_operand` | [31:0] | RW | EW ALU operand (**32-bit**, wider than BS/BN) |
| **D_DP_EW_ALU_CVT_OFFSET** | `0x2423` | `ew_alu_cvt_offset` | [31:0] | RW | ALU input converter offset |
| **D_DP_EW_ALU_CVT_SCALE** | `0x2424` | `ew_alu_cvt_scale` | [15:0] | RW | ALU input converter scale |
| **D_DP_EW_ALU_CVT_TRUNCATE** | `0x2425` | `ew_alu_cvt_truncate` | [5:0] | RW | ALU input converter truncate |
| **D_DP_EW_MUL_CFG** | `0x2426` | `ew_mul_src` | [0] | RW | **0** = register, **1** = memory |
| | | `ew_mul_cvt_bypass` | [1] | RW | **1** = bypass MUL input cvt |
| **D_DP_EW_MUL_SRC_VALUE** | `0x2427` | `ew_mul_operand` | [31:0] | RW | EW MUL operand (32-bit) |
| **D_DP_EW_MUL_CVT_OFFSET** | `0x2428` | `ew_mul_cvt_offset` | [31:0] | RW | MUL input converter offset |
| **D_DP_EW_MUL_CVT_SCALE** | `0x2429` | `ew_mul_cvt_scale` | [15:0] | RW | MUL input converter scale |
| **D_DP_EW_MUL_CVT_TRUNCATE** | `0x242A` | `ew_mul_cvt_truncate` | [5:0] | RW | MUL input converter truncate |
| **D_DP_EW_TRUNCATE_VALUE** | `0x242B` | `ew_truncate` | [9:0] | RW | Y-stage output truncate (10-bit, for MUL result) |

### 3.6 Feature Mode & Destination

| Register | Addr | Field | Bits | R/W | Description |
|----------|------|-------|------|-----|-------------|
| **D_FEATURE_MODE_CFG** | `0x242C` | `flying_mode` | [0] | RW | **0** = OFF (RDMA→DRAM), **1** = ON (from CACC) |
| | | `output_dst` | [1] | RW | **0** = MEM (WDMA), **1** = PDP |
| | | `winograd` | [2] | RW | **0** = OFF, **1** = ON |
| | | `nan_to_zero` | [3] | RW | **1** = convert NaN to zero |
| | | `batch_number` | [12:8] | RW | Batch number (0-based) |
| **D_DST_DMA_CFG** | `0x242D` | `dst_ram_type` | [0] | RW | **0** = CV SRAM, **1** = MC DRAM |
| **D_DST_BATCH_STRIDE** | `0x242E` | `dst_batch_stride` | [31:0] | RW | Batch stride in bytes |

### 3.7 Data Format & Output Converter

| Register | Addr | Field | Bits | R/W | Description |
|----------|------|-------|------|-----|-------------|
| **D_DATA_FORMAT** | `0x242F` | `proc_precision` | [1:0] | RW | Processing: **0**=INT8, **1**=INT16, **2**=FP16 |
| | | `out_precision` | [3:2] | RW | Output: **0**=INT8, **1**=INT16, **2**=FP16 |
| **D_CVT_OFFSET** | `0x2430` | `cvt_offset` | [31:0] | RW | Output converter offset (signed) |
| **D_CVT_SCALE** | `0x2431` | `cvt_scale` | [15:0] | RW | Output converter scale multiplier |
| **D_CVT_SHIFT** | `0x2432` | `cvt_shift` | [5:0] | RW | Output converter right-shift |

**Output Converter math**: `out = saturate( (data − cvt_offset) × cvt_scale >> cvt_shift )`

### 3.8 Status & Performance (Read-Only)

| Register | Addr | Field | Bits | R/W | Description |
|----------|------|-------|------|-----|-------------|
| **D_STATUS** | `0x2433` | `status_unequal` | [0] | RO | EQL mode inequality flag |
| **D_STATUS_NAN_INPUT_NUM** | `0x2434` | | [31:0] | RO | NaN input count |
| **D_STATUS_INF_INPUT_NUM** | `0x2435` | | [31:0] | RO | Inf input count |
| **D_STATUS_NAN_OUTPUT_NUM** | `0x2436` | | [31:0] | RO | NaN output count |
| **D_PERF_ENABLE** | `0x2437` | `perf_dma_en` | [0] | RW | Enable DMA stall counter |
| | | `perf_lut_en` | [1] | RW | Enable LUT hit counters |
| | | `perf_sat_en` | [2] | RW | Enable saturation counter |
| | | `perf_nan_inf_count_en` | [3] | RW | Enable NaN/Inf counters |
| **D_PERF_WDMA_WRITE_STALL** | `0x2438` | | [31:0] | RO | WDMA write stall cycles |
| **D_PERF_LUT_UFLOW** | `0x2439` | | [31:0] | RO | LUT underflow count |
| **D_PERF_LUT_OFLOW** | `0x243A` | | [31:0] | RO | LUT overflow count |
| **D_PERF_OUT_SATURATION** | `0x243B` | | [31:0] | RO | Output saturation count |
| **D_PERF_LUT_HYBRID** | `0x243C` | | [31:0] | RO | LUT hybrid hit count (both LE+LO hit) |
| **D_PERF_LUT_LE_HIT** | `0x243D` | | [31:0] | RO | LUT LE-only hit count |
| **D_PERF_LUT_LO_HIT** | `0x243E` | | [31:0] | RO | LUT LO-only hit count |

---

## 4. Register Map — SDP RDMA

The SDP RDMA reads operand data from DRAM for BS/BN/EW sub-processors. The RDMA register space is at byte addresses `0x8000–0x8090`.

> **Note**: The current verification framework (`Layers_regs_addresses.py`) does NOT define `SDP_RDMA_REG`. For SDP activation tests using memory operands, an `SDP_RDMA_REG` class must be added.

### Key RDMA Registers

| Register | Byte Addr | Key Fields | Description |
|----------|-----------|------------|-------------|
| **D_OP_ENABLE** | `0x8008` | `op_en[0]` | Enable RDMA |
| **D_DATA_CUBE_WIDTH/HEIGHT/CHANNEL** | `0x800C–0x8014` | `[12:0]` | Input cube dimensions (N−1) |
| **D_SRC_BASE_ADDR_LOW/HIGH** | `0x8018–0x801C` | `[31:0]` | Main RDMA source address |
| **D_SRC_LINE/SURFACE_STRIDE** | `0x8020–0x8024` | `[31:0]` | Source strides |
| **D_BRDMA_CFG** | `0x8028` | `disable[0]`, `data_use[2:1]`, `data_size[3]`, `data_mode[4]` | BS operand DMA config |
| **D_BS_BASE_ADDR_LOW/HIGH** | `0x802C–0x8030` | `[31:0]` | BS operand DRAM address |
| **D_NRDMA_CFG** | `0x8040` | Same fields as BRDMA | BN operand DMA config |
| **D_BN_BASE_ADDR_LOW/HIGH** | `0x8044–0x8048` | `[31:0]` | BN operand DRAM address |
| **D_ERDMA_CFG** | `0x8058` | Same fields as BRDMA | EW operand DMA config |
| **D_EW_BASE_ADDR_LOW/HIGH** | `0x805C–0x8060` | `[31:0]` | EW operand DRAM address |
| **D_FEATURE_MODE_CFG** | `0x8070` | `flying_mode[0]`, `in/proc/out_precision`, `batch_number` | Input source and precision |

### RDMA DMA Config Bit Fields (applies to BRDMA/NRDMA/ERDMA)

| Bit | Field | Values |
|-----|-------|--------|
| [0] | `disable` | **0** = enabled, **1** = disabled (default) |
| [2:1] | `data_use` | **0** = MUL only, **1** = ALU only, **2** = BOTH (ALU+MUL) |
| [3] | `data_size` | **0** = ONE_BYTE (INT8), **1** = TWO_BYTE (INT16) |
| [4] | `data_mode` | **0** = PER_KERNEL (broadcast same value), **1** = PER_ELEMENT |
| [5] | `ram_type` | **0** = CV SRAM, **1** = MC DRAM |

---

## 5. Sub-Processor Pipelines (BS / BN / EW)

### 5.1 X1 / X2 Pipeline (BS and BN — Identical Structure)

```
data_in[31:0] ──► ALU ──► MUL ──► TRT ──► ReLU ──► data_out[31:0]
                   │        │       │        │
                 16-bit   16-bit  shift    clamp≥0
                 operand  operand  right
```

**ALU stage** (33-bit internal):
- Selects operand from register (`cfg_alu_src=0`) or RDMA channel (`cfg_alu_src=1`)
- Left-shifts 16-bit operand by `cfg_alu_shift_value` (0–63) to create 32-bit effective operand
- Operation: MAX / MIN / SUM based on `cfg_alu_algo`
- Output: 33 bits (sign-extended)

**MUL stage** (49-bit internal):
- `cfg_mul_prelu=0`: normal signed multiply → 33×16 = 49-bit result
- `cfg_mul_prelu=1`: PReLU mode — positive inputs pass through unchanged; negative inputs get multiplied by operand (leak factor)

**TRT (Truncation)** stage:
- Right-shifts 49-bit product by `cfg_mul_shift_value` with rounding and saturation → 32 bits
- Bypassed when PReLU detected positive input

**ReLU stage**:
- `cfg_relu_bypass=0`: clamps negative values to 0
- `cfg_relu_bypass=1`: pass-through

### 5.2 Y Pipeline (EW — More Complex)

```
data_in[31:0] ──► MUL ──► ALU ──► [LUT bypass?] ──►┬──► data_out[31:0]
                   │        │                        │
                 32-bit   32-bit                   LUT path:
                 operand  operand                  IDX → LUT → INP
                 (cvt)    (cvt)                    (interp)
```

**Key differences from X1/X2:**
- **MUL comes BEFORE ALU** (reversed order!)
- **Operands are 32-bit** (vs 16-bit in X1/X2)
- Each operand has its own **input converter** (offset/scale/truncate) for ERDMA data
- **10-bit truncate** for MUL result (vs 6-bit in X1/X2)
- **LUT path** follows for programmable activation functions
- **EQL mode** (`ew_alu_algo=3`): special equality comparison, bypasses LUT and output CVT

### 5.3 Complete Bit-Field Reference for D_DP_BS_CFG / D_DP_BN_CFG / D_DP_EW_CFG

```
Bit [6]  [5]        [4]        [3:2]       [1]         [0]
     │    │          │           │           │           │
     │    │          │           │           │           └─ stage_bypass
     │    │          │           │           └───────────── alu_bypass
     │    │          │           └───────────────────────── alu_algo (MAX/MIN/SUM)
     │    │          └───────────────────────────────────── mul_bypass
     │    └──────────────────────────────────────────────── mul_prelu (BS/BN) or mul_prelu (EW)
     └───────────────────────────────────────────────────── relu_bypass (BS/BN) or lut_bypass (EW)
```

---

## 6. LUT System — Deep Dive

The Y-channel LUT enables **arbitrary programmable activation functions** (sigmoid, tanh, custom).

### 6.1 Dual-Table Architecture

| Table | Entries | Entry Width | Indexing Mode | Purpose |
|-------|---------|-------------|---------------|---------|
| **LE** (Linear/Exponential) | 65 (nv_small) | 16-bit signed | Exponential (log₂) or Linear | Covers large dynamic range with exponential spacing |
| **LO** (Linear Only) | 257 | 16-bit signed | Linear | Fine-grained linear coverage of a sub-range |

Both tables are looked up **simultaneously**. Priority logic decides which result to use based on hit/miss/overflow/underflow conditions.

### 6.2 LUT Programming Sequence

To program a LUT table via CSB:

```python
# 1. Select table and set to WRITE mode
#    For LE table: table_id=0, for LO table: table_id=1
for i, value in enumerate(table_entries):
    cfg = (i & 0x3FF) | (table_id << 16) | (1 << 17)  # addr | id | write
    write_reg(S_LUT_ACCESS_CFG, cfg)
    write_reg(S_LUT_ACCESS_DATA, value & 0xFFFF)
```

**Programming order**: Write `S_LUT_ACCESS_CFG` first (sets address, table, write mode), then write `S_LUT_ACCESS_DATA` (triggers the actual store).

### 6.3 Index Computation

**Exponential indexing** (LE with `lut_le_function=0`):
- Uses leading-sign-detector (LSD) for log₂ approximation
- `index = log2(input − lut_le_start) − lut_le_index_offset`
- Produces fractional part for interpolation
- Good for sigmoid/tanh where the interesting region spans many orders of magnitude

**Linear indexing** (LE with `lut_le_function=1`, or LO always):
- `index = (input − lut_start) >> lut_index_select`
- `fraction = (input − lut_start) & ((1 << lut_index_select) − 1)`
- Uniform spacing across the table range

### 6.4 Interpolation

After table lookup, linear interpolation between adjacent entries:

```
result = table[index] + fraction × (table[index+1] − table[index]) / (1 << fraction_bits)
```

The `NV_NVDLA_SDP_HLS_Y_inp_top` module performs this interpolation.

### 6.5 Overflow/Underflow Handling

When input falls outside the LUT range:
- **Underflow** (input < lut_start): slope extrapolation using `slope_uflow_scale` and `slope_uflow_shift`
- **Overflow** (input > lut_end): slope extrapolation using `slope_oflow_scale` and `slope_oflow_shift`

Extrapolation formula:
```
result = offset + (input − offset) × slope_scale >> slope_shift
```

Where `offset` = `lut_le_start` or `lut_le_end` depending on flow direction. For exponential LE tables, the bias calculation also factors in `lut_le_index_offset`.

### 6.6 Priority Resolution

When both LE and LO tables produce results:

| Condition | Priority Register | Winner |
|-----------|-------------------|--------|
| Both underflow | `lut_uflow_priority` | 0→LE, 1→LO |
| Both overflow | `lut_oflow_priority` | 0→LE, 1→LO |
| LE hit + LO hit (hybrid) | `lut_hybrid_priority` | 0→LE, 1→LO |
| Only one hit | — | The one that hit |

---

## 7. Output Converter (C)

The final stage converts internal 32-bit signed values to the output precision.

**Math**: `output = saturate( (data − CVT_OFFSET) × CVT_SCALE >> CVT_SHIFT )`

| Step | Width | Operation |
|------|-------|-----------|
| Subtract | 32→33 bit | `data − cvt_offset` (signed) |
| Multiply | 33×16→49 bit | `× cvt_scale` (signed) |
| Truncate | 49→17 bit | `>> cvt_shift` with rounding + saturation |
| Saturate | 17→8 or 16 bit | Clamp to INT8 [−128,127] or INT16 [−32768,32767] |

**Identity passthrough**: Set `cvt_offset=0`, `cvt_scale=1`, `cvt_shift=0`.

**EQL mode bypass**: When `ew_alu_algo=3` and EW is not bypassed, the output converter is skipped entirely.

---

## 8. Hardware Limitations & Constraints

### 8.1 Data Width Constraints

| Parameter | nv_small Value | Impact |
|-----------|---------------|--------|
| Throughput (k) | **1** element/cycle | Single element processed per clock |
| Internal precision | **32-bit** signed | All intermediate values are 32-bit |
| BS/BN operand width | **16-bit** signed | Max operand value ±32767 |
| EW operand width | **32-bit** signed | Full 32-bit operand range |
| LUT entry width | **16-bit** signed | Table values limited to ±32767 |
| LUT LE entries | **65** (nv_small) | Coarser than full config (257) |
| LUT LO entries | **257** | Full linear table |
| Output CVT scale | **16-bit** signed | Limits output scaling precision |
| Output CVT shift | **6-bit** (0–63) | Max right-shift = 63 |

### 8.2 Processing Limitations

| Limitation | Details |
|------------|---------|
| **No FP16 in nv_small** | Only INT8 and INT16 supported |
| **X1/X2 multiply overflow** | 33×16=49 bits, truncated to 32. If `shift_value` is too small, saturation occurs |
| **Y multiply overflow** | 32×32=64 bits, truncated by 10-bit `ew_truncate`. Must be carefully managed |
| **LUT precision** | 16-bit entries limit activation function precision. Interpolation helps but can't overcome quantization |
| **LUT range** | Input outside [LE_START, LE_END] / [LO_START, LO_END] uses linear slope extrapolation only |
| **No division** | All operations are add/subtract/multiply/shift. Division must be pre-converted to multiply+shift |
| **Serial pipeline** | X1→X2→Y are strictly serial. Cannot run X1 and Y in parallel on different data |
| **ALU shift direction** | ALU operand shift is LEFT only (amplifies operand). Truncation shift is RIGHT only |
| **PReLU in Y** | Y-stage has PReLU in MUL but no standalone ReLU block (unlike X1/X2) |
| **RDMA disabled by default** | All three RDMAs (BRDMA/NRDMA/ERDMA) default to disabled. Must explicitly enable for memory operands |

### 8.3 Address Alignment

| Parameter | Alignment |
|-----------|-----------|
| Source/Destination base address | Must be 32-byte aligned (ATOM_C = 32 bytes for nv_small INT8) |
| Line stride | Must be a multiple of 32 bytes |
| Surface stride | Must be line_stride × height |

---

## 9. Activation Function Configurations

### 9.1 ReLU — Using BS Stage

The simplest activation. Uses the built-in ReLU gate at the end of X1 (BS stage).

```
Pipeline: data ──► [BS: ALU bypass, MUL bypass, ReLU ENABLED] ──► [BN bypass] ──► [EW bypass] ──► CVT
```

**Register values:**
| Register | Value | Explanation |
|----------|-------|-------------|
| `D_DP_BS_CFG` | `0x00000040` | bs_bypass=0, alu_bypass=1(bit1), mul_bypass=1(bit4), relu_bypass=**0**(bit6=0), prelu=0 → BUT WAIT: bypass=0 means stage active → we need `0b_0_0_1_00_1_0` = bypass=0, alu_bypass=1, algo=0, mul_bypass=1, prelu=0, relu_bypass=0 |

Let me provide the exact bit calculations:

**D_DP_BS_CFG for ReLU:**
```
bit[0] = 0  (bs_bypass = NO, stage is active)
bit[1] = 1  (bs_alu_bypass = YES, skip ALU)
bit[3:2] = 00 (alu_algo = don't care)
bit[4] = 1  (bs_mul_bypass = YES, skip MUL)
bit[5] = 0  (bs_mul_prelu = NO)
bit[6] = 0  (bs_relu_bypass = NO, ReLU ACTIVE)
→ Value = 0b_0_0_1_00_1_0 = 0x12
```

| Register | Value | Notes |
|----------|-------|-------|
| `D_DP_BS_CFG` | `0x00000012` | enable BS stage, bypass ALU+MUL, enable ReLU |
| `D_DP_BN_CFG` | `0x00000001` | bypass BN entirely |
| `D_DP_EW_CFG` | `0x00000001` | bypass EW entirely |
| `D_DATA_FORMAT` | `0x00000000` | INT8 proc + INT8 out |
| `D_CVT_OFFSET` | `0x00000000` | no offset |
| `D_CVT_SCALE` | `0x00000001` | identity scale |
| `D_CVT_SHIFT` | `0x00000000` | no shift |

### 9.2 Leaky ReLU / PReLU — Using BS Stage

PReLU: positive values pass through; negative values are multiplied by a **leak factor**.

```
Pipeline: data ──► [BS: ALU bypass, MUL with PReLU, TRT, ReLU bypass] ──► CVT
```

**D_DP_BS_CFG for PReLU:**
```
bit[0] = 0  (stage active)
bit[1] = 1  (alu_bypass)
bit[3:2] = 00 (don't care)
bit[4] = 0  (mul NOT bypassed — need the multiply)
bit[5] = 1  (mul_prelu = YES)
bit[6] = 1  (relu_bypass = YES — PReLU handles it)
→ Value = 0b_1_1_0_00_1_0 = 0x62
```

| Register | Value | Notes |
|----------|-------|-------|
| `D_DP_BS_CFG` | `0x00000062` | enable BS, bypass ALU, MUL active with PReLU, bypass ReLU |
| `D_DP_BS_MUL_CFG` | `0x00000000` | src=REG, shift=0 (adjust shift as needed) |
| `D_DP_BS_MUL_SRC_VALUE` | leak factor | e.g., `0x0019` for α≈0.1 if scale requires it |
| `D_DP_BS_MUL_CFG[15:8]` | shift_value | Right-shift to re-normalize after multiply |
| `D_DP_BN_CFG` | `0x00000001` | bypass BN |
| `D_DP_EW_CFG` | `0x00000001` | bypass EW |

**Leak factor encoding**: For Leaky ReLU with α = 0.01:
- With `shift_value = 14`: operand = round(0.01 × 2¹⁴) = 164 → `0x00A4`
- `D_DP_BS_MUL_CFG` = `(14 << 8) | 0` = `0x0E00`
- `D_DP_BS_MUL_SRC_VALUE` = `0x00A4`

### 9.3 Bias + Scale — Using BS Stage

Add a constant bias then multiply by a scale factor.

```
Pipeline: data ──► [BS: ALU adds bias, MUL scales, TRT, ReLU optional] ──► CVT
```

**D_DP_BS_CFG for Bias+Scale+ReLU:**
```
bit[0] = 0  (stage active)
bit[1] = 0  (alu NOT bypassed)
bit[3:2] = 10 (alu_algo = SUM)
bit[4] = 0  (mul NOT bypassed)
bit[5] = 0  (prelu = NO)
bit[6] = 0  (relu = ACTIVE)
→ Value = 0b_0_0_0_10_0_0 = 0x08
```

| Register | Value | Notes |
|----------|-------|-------|
| `D_DP_BS_CFG` | `0x00000008` | active, ALU=SUM, MUL active, ReLU active |
| `D_DP_BS_ALU_CFG` | `shift << 8` | shift the 16-bit bias to match data scale |
| `D_DP_BS_ALU_SRC_VALUE` | bias value | 16-bit signed bias |
| `D_DP_BS_MUL_CFG` | `shift << 8` | truncation shift |
| `D_DP_BS_MUL_SRC_VALUE` | scale value | 16-bit signed scale |

### 9.4 Sigmoid — Using EW Stage + LUT

Sigmoid requires the programmable LUT.

```
Pipeline: data ──► [BS bypass] ──► [BN bypass] ──► [EW: MUL bypass, ALU bypass, LUT ENABLED] ──► CVT
```

#### Step 1: Pre-compute Sigmoid LUT Table

For INT8 input range [−128, 127], sigmoid(x) = 1/(1+e^(−x)):

```python
import numpy as np

# For LO table (257 entries, linear indexing)
# Map input range to table index  
x_min, x_max = -8.0, 8.0  # scaled range after fixed-point conversion
x_values = np.linspace(x_min, x_max, 257)
sigmoid_values = 1.0 / (1.0 + np.exp(-x_values))
# Quantize to INT16
lut_lo = np.clip(np.round(sigmoid_values * 32767), -32768, 32767).astype(np.int16)
```

#### Step 2: Program LUT Tables

```python
# Program LO table (257 entries)
for i in range(257):
    write_reg(SDP_REG.S_LUT_ACCESS_CFG, (i & 0x3FF) | (1 << 16) | (1 << 17))  # LO, WRITE
    write_reg(SDP_REG.S_LUT_ACCESS_DATA, int(lut_lo[i]) & 0xFFFF)

# Program LE table (65 entries) — exponential spacing for wider coverage
for i in range(65):
    write_reg(SDP_REG.S_LUT_ACCESS_CFG, (i & 0x3FF) | (0 << 16) | (1 << 17))  # LE, WRITE
    write_reg(SDP_REG.S_LUT_ACCESS_DATA, int(lut_le[i]) & 0xFFFF)
```

#### Step 3: Configure LUT Parameters

| Register | Value | Explanation |
|----------|-------|-------------|
| `S_LUT_CFG` | `0x00000001` | LE function = LINEAR (for simpler start) |
| `S_LUT_INFO` | `(lo_sel << 16) \| (le_sel << 8) \| le_offset` | Index scaling |
| `S_LUT_LE_START` | Input range start (signed 32-bit) | e.g., `0xFFFFFFF8` for −8 |
| `S_LUT_LE_END` | Input range end | e.g., `0x00000008` for +8 |
| `S_LUT_LO_START` | Same or different range for LO | |
| `S_LUT_LO_END` | | |
| `S_LUT_LE_SLOPE_SCALE` | `0x00000000` | 0 slope outside range (sigmoid saturates) |
| `S_LUT_LE_SLOPE_SHIFT` | `0x00000000` | |
| `S_LUT_LO_SLOPE_SCALE` | `0x00000000` | 0 slope (sigmoid = 0 or 1 at extremes) |
| `S_LUT_LO_SLOPE_SHIFT` | `0x00000000` | |

#### Step 4: Configure EW Stage

**D_DP_EW_CFG for LUT activation:**
```
bit[0] = 0  (ew_bypass = NO, stage active)
bit[1] = 1  (ew_alu_bypass = YES)
bit[3:2] = 00 (don't care)
bit[4] = 1  (ew_mul_bypass = YES)
bit[5] = 0  (prelu = NO)
bit[6] = 0  (ew_lut_bypass = NO, LUT ACTIVE)
→ Value = 0b_0_0_1_00_1_0 = 0x12
```

| Register | Value | Notes |
|----------|-------|-------|
| `D_DP_BS_CFG` | `0x00000001` | bypass BS |
| `D_DP_BN_CFG` | `0x00000001` | bypass BN |
| `D_DP_EW_CFG` | `0x00000012` | enable EW, bypass ALU+MUL, enable LUT |
| `D_DATA_FORMAT` | `0x00000000` | INT8 proc + INT8 out |
| `D_CVT_OFFSET` | `0x00000000` | adjust for sigmoid output scaling |
| `D_CVT_SCALE` | scale factor | re-quantize sigmoid output back to INT8 |
| `D_CVT_SHIFT` | shift value | right-shift after scale multiply |

### 9.5 Tanh — Using EW Stage + LUT

Identical structure to Sigmoid, but with tanh table values:

```python
x_values = np.linspace(-4.0, 4.0, 257)
tanh_values = np.tanh(x_values)
lut_lo = np.clip(np.round(tanh_values * 32767), -32768, 32767).astype(np.int16)
```

The register configuration is the same as Sigmoid; only the LUT table contents and range parameters change.

### 9.6 Clamp (Min/Max) — Using BS ALU

Clamp output to a range [min_val, max_val] using two stages:

```
Pipeline: data ──► [BS: ALU=MAX with min_val, then MUL bypass] ──► [BN: ALU=MIN with max_val] ──► CVT
```

| Register | Value | Notes |
|----------|-------|-------|
| `D_DP_BS_CFG` | `0x00000052` | active, ALU=MAX(algo=0), MUL bypass, ReLU bypass |
| `D_DP_BS_ALU_SRC_VALUE` | min_val | Lower clamp bound |
| `D_DP_BN_CFG` | `0x00000056` | active, ALU=MIN(algo=1→bits=01), MUL bypass, ReLU bypass |
| `D_DP_BN_ALU_SRC_VALUE` | max_val | Upper clamp bound |
| `D_DP_EW_CFG` | `0x00000001` | bypass EW |

### 9.7 Batch Normalization — Using BN Stage

For `output = γ × (input − μ) / σ + β`:
- **ALU**: adds `−μ` (encoded as bias with shift)
- **MUL**: multiplies by `γ/σ` (pre-computed as fixed-point)
- After MUL, a **second pass** or the output CVT adds β

| Register | Value | Notes |
|----------|-------|-------|
| `D_DP_BS_CFG` | `0x00000001` | bypass BS |
| `D_DP_BN_CFG` | `0x00000008` | active, ALU=SUM, MUL active, ReLU bypass(bit6=0 or 1) |
| `D_DP_BN_ALU_CFG` | `0x00000000` | src=REG |
| `D_DP_BN_ALU_SRC_VALUE` | −μ (quantized) | Mean subtraction |
| `D_DP_BN_MUL_CFG` | `shift << 8` | src=REG + truncation shift |
| `D_DP_BN_MUL_SRC_VALUE` | γ/σ (quantized) | Scale factor |
| `D_CVT_OFFSET` | −β (if adding bias via CVT) | Or use BS stage for bias |

### 9.8 Conv + ReLU (Combined, Flying Mode)

The most common use case: convolution output passes through SDP with ReLU.

```
CACC ──flying──► SDP [BS: ReLU only] ──► WDMA ──► DRAM
```

| Register | Value | Notes |
|----------|-------|-------|
| `D_DP_BS_CFG` | `0x00000012` | active, ALU bypass, MUL bypass, ReLU ON |
| `D_DP_BN_CFG` | `0x00000001` | bypass |
| `D_DP_EW_CFG` | `0x00000001` | bypass |
| `D_FEATURE_MODE_CFG` | `0x00000001` | flying_mode=1 (from CACC) |
| `D_DST_DMA_CFG` | `0x00000001` | MC DRAM |
| `D_DATA_FORMAT` | `0x00000000` | INT8 → INT8 |
| `D_CVT_OFFSET` | `0x00000000` | identity |
| `D_CVT_SCALE` | `0x00000001` | identity |
| `D_CVT_SHIFT` | `0x00000000` | identity |

---

## 10. Register Programming Recipes

### 10.1 Programming Order (Critical!)

The SDP must be programmed **bottom-up** (SDP before upstream blocks) and enable order matters:

```
1. Program S_POINTER (select register group)
2. Write all S_LUT_* registers (if using LUT)
3. Write all D_* configuration registers
4. Write D_OP_ENABLE = 1  ← MUST BE LAST within SDP
5. Then enable upstream: CACC → CMAC_A → CMAC_B → CSC → CDMA
```

### 10.2 SDP RDMA Programming (When Using Memory Operands)

When using per-element operands from DRAM (not register constants):

```
1. Program SDP_RDMA S_POINTER
2. Configure data cube dimensions (must match SDP core dims)
3. Configure xRDMA_CFG (enable, data_use, size, mode)
4. Set base address and strides for the operand data
5. Set feature mode (flying_mode, precision)
6. Write SDP_RDMA D_OP_ENABLE = 1
7. THEN program SDP core registers
```

### 10.3 LUT Programming Protocol

```python
def program_lut(table_id, entries):
    """
    table_id: 0 = LE, 1 = LO
    entries: list of int16 values
    """
    for addr, value in enumerate(entries):
        cfg = (addr & 0x3FF) | (table_id << 16) | (1 << 17)
        write_reg(SDP_REG.S_LUT_ACCESS_CFG, cfg)
        write_reg(SDP_REG.S_LUT_ACCESS_DATA, value & 0xFFFF)
```

---

## 11. Verification Testbench Integration

### 11.1 Current State

The existing testbench (`pyuvm_components/`) supports SDP in **passthrough mode only** via `conv_configs()` in `regs_configs.py`. Key observations:

- `SDP_REG` class in `Layers_regs_addresses.py`: **Complete** — all 63 registers defined
- `conv_configs()`: Sets `D_DP_BS_CFG=1`, `D_DP_BN_CFG=1`, `D_DP_EW_CFG=1` (all bypass)
- **No LUT programming** exists anywhere in the codebase
- **No `SDP_RDMA_REG` class** defined (needed for memory operands)

### 11.2 What Needs to Be Built for Activation Tests

| Component | File | Status | Effort |
|-----------|------|--------|--------|
| `SDP_RDMA_REG` address class | `Layers_regs_addresses.py` | **Missing** | Low |
| SDP activation `regs_configs` method | `regs_configs.py` | **Missing** | Medium |
| LUT table generator (sigmoid/tanh/etc.) | New: `strategy/sdp_lut_tables.py` | **Missing** | Medium |
| SDP activation golden model | New: `strategy/sdp_activation_strategy.py` | **Missing** | High |
| `LayerFactory` registration | `Layer_Factory.py` | 1 line addition | Trivial |
| YAML configs for activation tests | `yaml/` | **Missing** | Low |
| Test classes | `pyuvm_components/test.py` | **Missing** | Low |
| CsbDriver / Scoreboard | Existing | **No changes needed** | None |

### 11.3 Golden Model Requirements

The golden model must replicate NVDLA's **exact fixed-point LUT interpolation**, not floating-point approximations:

1. **Input quantization**: INT8 input → 32-bit signed internal
2. **LUT indexing**: exact log₂ (for exponential) or linear shift computation
3. **Table lookup**: return `table[index]` and `table[index+1]` (both 16-bit)
4. **Interpolation**: `result = y0 + fraction × (y1 − y0)` with correct fractional bit width
5. **Overflow/underflow**: slope extrapolation with scale/shift
6. **Output CVT**: `saturate((result − offset) × scale >> shift)` to INT8

Mismatches in any of these steps will cause scoreboard failures.

### 11.4 Suggested YAML Extension

```yaml
test_suite:
  - name: "conv_1x1x8_relu"
    layers:
      - type: "convolution"
        config:
          input_shape: [1, 1, 8]
          num_kernels: 1
          kernel_h: 1
          kernel_w: 1
          # ... standard conv config ...
          
          # SDP activation config
          sdp_activation: "relu"        # relu | prelu | sigmoid | tanh | none

      - type: "convolution"
        config:
          # ... conv config ...
          sdp_activation: "sigmoid"
          sdp_lut_range: [-8, 8]        # input range for LUT tables
          sdp_lut_precision: 15         # fractional bits for LUT quantization
```

---

## Quick Reference: SDP Register Summary

| Address Range | Block | Count | Purpose |
|---------------|-------|-------|---------|
| `0x2400–0x2401` | SDP Single Status | 2 | Status, producer/consumer pointer |
| `0x2402–0x240D` | SDP Single LUT | 12 | LUT access, config, range, slopes |
| `0x240E–0x2415` | SDP Dual Core | 8 | Op enable, cube dims, dest addr |
| `0x2416–0x241A` | SDP Dual BS | 5 | BS (X1) ALU/MUL/ReLU config |
| `0x241B–0x241F` | SDP Dual BN | 5 | BN (X2) ALU/MUL/ReLU config |
| `0x2420–0x242B` | SDP Dual EW | 12 | EW (Y) ALU/MUL/LUT/CVT config |
| `0x242C–0x2432` | SDP Dual Feature/CVT | 7 | Mode, DMA, format, output CVT |
| `0x2433–0x243E` | SDP Dual Status/Perf | 12 | Status (RO), perf counters |
| **Total** | | **63** | |
