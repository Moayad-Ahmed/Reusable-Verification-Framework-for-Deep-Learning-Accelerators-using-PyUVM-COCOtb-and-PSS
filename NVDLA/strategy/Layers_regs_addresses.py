# ═══════════════════════════════════════════════════════════════════════
#  CONVOLUTION PIPELINE REGISTERS
#  Pipeline: CDMA → CBUF → CSC → CMAC_A/B → CACC → (SDP passthrough)
# ═══════════════════════════════════════════════════════════════════════

# GLB Registers (Base: 0x0400 = byte 0x1000)
class GLB_REG:
    """Global Control Registers — interrupt mask/status"""
    BASE = 0x0400
    S_NVDLA_HW_VERSION = BASE + 0x00   # 0x0400
    S_INTR_MASK         = BASE + 0x01   # 0x0401
    S_INTR_SET          = BASE + 0x02   # 0x0402
    S_INTR_STATUS       = BASE + 0x03   # 0x0403


# CDMA Registers (Base: 0x0C00 = byte 0x3000)
class CDMA_REG:
    """Convolution DMA — fetches input data & weights from DRAM into CBUF"""
    BASE = 0x0C00
    # Status / Control
    S_STATUS              = BASE + 0x00   # 0x0C00
    S_POINTER             = BASE + 0x01   # 0x0C01
    S_ARBITER             = BASE + 0x02   # 0x0C02
    S_CBUF_FLUSH_STATUS   = BASE + 0x03   # 0x0C03
    D_OP_ENABLE           = BASE + 0x04   # 0x0C04
    # Configuration
    D_MISC_CFG            = BASE + 0x05   # 0x0C05
    D_DATAIN_FORMAT       = BASE + 0x06   # 0x0C06
    D_DATAIN_SIZE_0       = BASE + 0x07   # 0x0C07  [12:0]=W-1 [28:16]=H-1
    D_DATAIN_SIZE_1       = BASE + 0x08   # 0x0C08  [12:0]=C-1
    D_DATAIN_SIZE_EXT_0   = BASE + 0x09   # 0x0C09  Winograd ext
    D_PIXEL_OFFSET        = BASE + 0x0A   # 0x0C0A
    D_DAIN_RAM_TYPE       = BASE + 0x0B   # 0x0C0B
    D_DAIN_ADDR_HIGH_0    = BASE + 0x0C   # 0x0C0C
    D_DAIN_ADDR_LOW_0     = BASE + 0x0D   # 0x0C0D
    D_DAIN_ADDR_HIGH_1    = BASE + 0x0E   # 0x0C0E  UV plane
    D_DAIN_ADDR_LOW_1     = BASE + 0x0F   # 0x0C0F  UV plane
    D_LINE_STRIDE         = BASE + 0x10   # 0x0C10
    D_LINE_UV_STRIDE      = BASE + 0x11   # 0x0C11
    D_SURF_STRIDE         = BASE + 0x12   # 0x0C12
    D_DAIN_MAP            = BASE + 0x13   # 0x0C13
    D_RESERVED_X_CFG      = BASE + 0x14   # 0x0C14
    D_RESERVED_Y_CFG      = BASE + 0x15   # 0x0C15
    D_BATCH_NUMBER        = BASE + 0x16   # 0x0C16
    D_BATCH_STRIDE        = BASE + 0x17   # 0x0C17
    D_ENTRY_PER_SLICE     = BASE + 0x18   # 0x0C18
    D_FETCH_GRAIN         = BASE + 0x19   # 0x0C19
    # Weight
    D_WEIGHT_FORMAT       = BASE + 0x1A   # 0x0C1A
    D_WEIGHT_SIZE_0       = BASE + 0x1B   # 0x0C1B  bytes_per_kernel-1
    D_WEIGHT_SIZE_1       = BASE + 0x1C   # 0x0C1C  num_kernels-1
    D_WEIGHT_RAM_TYPE     = BASE + 0x1D   # 0x0C1D
    D_WEIGHT_ADDR_HIGH    = BASE + 0x1E   # 0x0C1E
    D_WEIGHT_ADDR_LOW     = BASE + 0x1F   # 0x0C1F
    D_WEIGHT_BYTES        = BASE + 0x20   # 0x0C20
    D_WGS_ADDR_HIGH       = BASE + 0x21   # 0x0C21
    D_WGS_ADDR_LOW        = BASE + 0x22   # 0x0C22
    D_WMB_ADDR_HIGH       = BASE + 0x23   # 0x0C23
    D_WMB_ADDR_LOW        = BASE + 0x24   # 0x0C24
    D_WMB_BYTES           = BASE + 0x25   # 0x0C25
    # Pre-processing
    D_MEAN_FORMAT         = BASE + 0x26   # 0x0C26
    D_MEAN_GLOBAL_0       = BASE + 0x27   # 0x0C27
    D_MEAN_GLOBAL_1       = BASE + 0x28   # 0x0C28
    D_CVT_CFG             = BASE + 0x29   # 0x0C29
    D_CVT_OFFSET          = BASE + 0x2A   # 0x0C2A
    D_CVT_SCALE           = BASE + 0x2B   # 0x0C2B
    # Convolution geometry
    D_CONV_STRIDE         = BASE + 0x2C   # 0x0C2C  [2:0]=xs-1 [18:16]=ys-1
    D_ZERO_PADDING        = BASE + 0x2D   # 0x0C2D  [4:0]=left [20:16]=top
    D_ZERO_PADDING_VALUE  = BASE + 0x2E   # 0x0C2E
    D_BANK                = BASE + 0x2F   # 0x0C2F  [4:0]=data [20:16]=wt
    # Misc
    D_NAN_FLUSH_TO_ZERO   = BASE + 0x30   # 0x0C30
    D_NAN_INPUT_DATA_NUM  = BASE + 0x31   # 0x0C31  (RO)
    D_NAN_INPUT_WEIGHT_NUM = BASE + 0x32  # 0x0C32  (RO)
    D_INF_INPUT_DATA_NUM  = BASE + 0x33   # 0x0C33  (RO)
    D_INF_INPUT_WEIGHT_NUM = BASE + 0x34  # 0x0C34  (RO)
    D_PERF_ENABLE         = BASE + 0x35   # 0x0C35
    D_PERF_DAT_READ_STALL = BASE + 0x36   # 0x0C36  (RO)
    D_PERF_WT_READ_STALL  = BASE + 0x37   # 0x0C37  (RO)
    D_PERF_DAT_READ_LATENCY = BASE + 0x38 # 0x0C38  (RO)
    D_PERF_WT_READ_LATENCY = BASE + 0x39  # 0x0C39  (RO)
    D_CYA                 = BASE + 0x3A   # 0x0C3A


# CSC Registers (Base: 0x1000 = byte 0x4000)
class CSC_REG:
    """Convolution Sequence Controller — reads CBUF, schedules MACs"""
    BASE = 0x1000
    S_STATUS              = BASE + 0x00   # 0x1000
    S_POINTER             = BASE + 0x01   # 0x1001
    D_OP_ENABLE           = BASE + 0x02   # 0x1002
    D_MISC_CFG            = BASE + 0x03   # 0x1003
    D_DATAIN_FORMAT       = BASE + 0x04   # 0x1004
    D_DATAIN_SIZE_EXT_0   = BASE + 0x05   # 0x1005  [12:0]=W-1 [28:16]=H-1
    D_DATAIN_SIZE_EXT_1   = BASE + 0x06   # 0x1006  [12:0]=C-1
    D_BATCH_NUMBER        = BASE + 0x07   # 0x1007
    D_POST_Y_EXTENSION    = BASE + 0x08   # 0x1008
    D_ENTRY_PER_SLICE     = BASE + 0x09   # 0x1009
    D_WEIGHT_FORMAT       = BASE + 0x0A   # 0x100A
    D_WEIGHT_SIZE_EXT_0   = BASE + 0x0B   # 0x100B  [4:0]=kw-1 [20:16]=kh-1
    D_WEIGHT_SIZE_EXT_1   = BASE + 0x0C   # 0x100C  [12:0]=C-1 [28:16]=K-1
    D_WEIGHT_BYTES        = BASE + 0x0D   # 0x100D
    D_WMB_BYTES           = BASE + 0x0E   # 0x100E
    D_DATAOUT_SIZE_0      = BASE + 0x0F   # 0x100F  [12:0]=W-1 [28:16]=H-1
    D_DATAOUT_SIZE_1      = BASE + 0x10   # 0x1010  [12:0]=K-1
    D_ATOMICS             = BASE + 0x11   # 0x1011  W*H-1
    D_RELEASE             = BASE + 0x12   # 0x1012
    D_CONV_STRIDE_EXT     = BASE + 0x13   # 0x1013  mirrors CDMA stride
    D_DILATION_EXT        = BASE + 0x14   # 0x1014  [4:0]=dx-1 [20:16]=dy-1
    D_ZERO_PADDING        = BASE + 0x15   # 0x1015  [4:0]=left [20:16]=top
    D_ZERO_PADDING_VALUE  = BASE + 0x16   # 0x1016
    D_BANK                = BASE + 0x17   # 0x1017
    D_PRA_CFG             = BASE + 0x18   # 0x1018
    D_CYA                 = BASE + 0x19   # 0x1019


# CMAC_A Registers (Base: 0x1400 = byte 0x5000)
class CMAC_A_REG:
    """Convolution MAC Array A"""
    BASE = 0x1400
    S_STATUS     = BASE + 0x00   # 0x1400
    S_POINTER    = BASE + 0x01   # 0x1401
    D_OP_ENABLE  = BASE + 0x02   # 0x1402
    D_MISC_CFG   = BASE + 0x03   # 0x1403


# CMAC_B Registers (Base: 0x1800 = byte 0x6000)
class CMAC_B_REG:
    """Convolution MAC Array B"""
    BASE = 0x1800
    S_STATUS     = BASE + 0x00   # 0x1800
    S_POINTER    = BASE + 0x01   # 0x1801
    D_OP_ENABLE  = BASE + 0x02   # 0x1802
    D_MISC_CFG   = BASE + 0x03   # 0x1803


# CACC Registers (Base: 0x1C00 = byte 0x7000)
class CACC_REG:
    """Convolution Accumulator — partial-sum accumulation, truncation"""
    BASE = 0x1C00
    S_STATUS          = BASE + 0x00   # 0x1C00
    S_POINTER         = BASE + 0x01   # 0x1C01
    D_OP_ENABLE       = BASE + 0x02   # 0x1C02
    D_MISC_CFG        = BASE + 0x03   # 0x1C03
    D_DATAOUT_SIZE_0  = BASE + 0x04   # 0x1C04  [12:0]=W-1 [28:16]=H-1
    D_DATAOUT_SIZE_1  = BASE + 0x05   # 0x1C05  [12:0]=K-1
    D_DATAOUT_ADDR    = BASE + 0x06   # 0x1C06  internal routing
    D_BATCH_NUMBER    = BASE + 0x07   # 0x1C07
    D_LINE_STRIDE     = BASE + 0x08   # 0x1C08  W*atomK*4
    D_SURF_STRIDE     = BASE + 0x09   # 0x1C09
    D_DATAOUT_MAP     = BASE + 0x0A   # 0x1C0A  [0]=line_pk [16]=surf_pk
    D_CLIP_CFG        = BASE + 0x0B   # 0x1C0B  [4:0]=truncate bits
    D_OUT_SATURATION  = BASE + 0x0C   # 0x1C0C  (RO)
    D_CYA             = BASE + 0x0D   # 0x1C0D


# SDP Registers (Base: 0x2400 = byte 0x9000)
class SDP_REG:
    """Single Data Processor — used as transparent passthrough for conv-only"""
    BASE = 0x2400
    # Status / LUT (S_ registers)
    S_STATUS                = BASE + 0x00   # 0x2400
    S_POINTER               = BASE + 0x01   # 0x2401
    S_LUT_ACCESS_CFG        = BASE + 0x02   # 0x2402
    S_LUT_ACCESS_DATA       = BASE + 0x03   # 0x2403
    S_LUT_CFG               = BASE + 0x04   # 0x2404
    S_LUT_INFO              = BASE + 0x05   # 0x2405
    S_LUT_LE_START          = BASE + 0x06   # 0x2406
    S_LUT_LE_END            = BASE + 0x07   # 0x2407
    S_LUT_LO_START          = BASE + 0x08   # 0x2408
    S_LUT_LO_END            = BASE + 0x09   # 0x2409
    S_LUT_LE_SLOPE_SCALE    = BASE + 0x0A   # 0x240A
    S_LUT_LE_SLOPE_SHIFT    = BASE + 0x0B   # 0x240B
    S_LUT_LO_SLOPE_SCALE    = BASE + 0x0C   # 0x240C
    S_LUT_LO_SLOPE_SHIFT    = BASE + 0x0D   # 0x240D
    # D_ operation registers
    D_OP_ENABLE             = BASE + 0x0E   # 0x240E
    D_DATA_CUBE_WIDTH       = BASE + 0x0F   # 0x240F
    D_DATA_CUBE_HEIGHT      = BASE + 0x10   # 0x2410
    D_DATA_CUBE_CHANNEL     = BASE + 0x11   # 0x2411
    D_DST_BASE_ADDR_LOW     = BASE + 0x12   # 0x2412
    D_DST_BASE_ADDR_HIGH    = BASE + 0x13   # 0x2413
    D_DST_LINE_STRIDE       = BASE + 0x14   # 0x2414
    D_DST_SURFACE_STRIDE    = BASE + 0x15   # 0x2415
    # BS sub-unit
    D_DP_BS_CFG             = BASE + 0x16   # 0x2416  [0]=bypass
    D_DP_BS_ALU_CFG         = BASE + 0x17   # 0x2417
    D_DP_BS_ALU_SRC_VALUE   = BASE + 0x18   # 0x2418
    D_DP_BS_MUL_CFG         = BASE + 0x19   # 0x2419
    D_DP_BS_MUL_SRC_VALUE   = BASE + 0x1A   # 0x241A
    # BN sub-unit
    D_DP_BN_CFG             = BASE + 0x1B   # 0x241B  [0]=bypass
    D_DP_BN_ALU_CFG         = BASE + 0x1C   # 0x241C
    D_DP_BN_ALU_SRC_VALUE   = BASE + 0x1D   # 0x241D
    D_DP_BN_MUL_CFG         = BASE + 0x1E   # 0x241E
    D_DP_BN_MUL_SRC_VALUE   = BASE + 0x1F   # 0x241F
    # EW sub-unit
    D_DP_EW_CFG             = BASE + 0x20   # 0x2420  [0]=bypass
    D_DP_EW_ALU_CFG         = BASE + 0x21   # 0x2421
    D_DP_EW_ALU_SRC_VALUE   = BASE + 0x22   # 0x2422
    D_DP_EW_ALU_CVT_OFFSET  = BASE + 0x23   # 0x2423
    D_DP_EW_ALU_CVT_SCALE   = BASE + 0x24   # 0x2424
    D_DP_EW_ALU_CVT_TRUNCATE = BASE + 0x25  # 0x2425
    D_DP_EW_MUL_CFG         = BASE + 0x26   # 0x2426
    D_DP_EW_MUL_SRC_VALUE   = BASE + 0x27   # 0x2427
    D_DP_EW_MUL_CVT_OFFSET  = BASE + 0x28   # 0x2428
    D_DP_EW_MUL_CVT_SCALE   = BASE + 0x29   # 0x2429
    D_DP_EW_MUL_CVT_TRUNCATE = BASE + 0x2A  # 0x242A
    D_DP_EW_TRUNCATE_VALUE  = BASE + 0x2B   # 0x242B
    # Feature / destination
    D_FEATURE_MODE_CFG      = BASE + 0x2C   # 0x242C  [0]=flying
    D_DST_DMA_CFG           = BASE + 0x2D   # 0x242D
    D_DST_BATCH_STRIDE      = BASE + 0x2E   # 0x242E
    D_DATA_FORMAT           = BASE + 0x2F   # 0x242F
    # Output converter
    D_CVT_OFFSET            = BASE + 0x30   # 0x2430
    D_CVT_SCALE             = BASE + 0x31   # 0x2431
    D_CVT_SHIFT             = BASE + 0x32   # 0x2432
    # Status (RO)
    D_STATUS                = BASE + 0x33   # 0x2433
    D_STATUS_NAN_INPUT_NUM  = BASE + 0x34   # 0x2434
    D_STATUS_INF_INPUT_NUM  = BASE + 0x35   # 0x2435
    D_STATUS_NAN_OUTPUT_NUM = BASE + 0x36   # 0x2436
    # Performance
    D_PERF_ENABLE           = BASE + 0x37   # 0x2437
    D_PERF_WDMA_WRITE_STALL = BASE + 0x38   # 0x2438
    D_PERF_LUT_UFLOW        = BASE + 0x39   # 0x2439
    D_PERF_LUT_OFLOW        = BASE + 0x3A   # 0x243A
    D_PERF_OUT_SATURATION   = BASE + 0x3B   # 0x243B
    D_PERF_LUT_HYBRID       = BASE + 0x3C   # 0x243C
    D_PERF_LUT_LE_HIT       = BASE + 0x3D   # 0x243D
    D_PERF_LUT_LO_HIT       = BASE + 0x3E   # 0x243E


# ═══════════════════════════════════════════════════════════════════════
#  POOLING PIPELINE REGISTERS (existing)
# ═══════════════════════════════════════════════════════════════════════

# PDP Core Registers (Base: 0x2C00)
class PDP_REG:
    """PDP (Planar Data Processor) Register Addresses"""
    BASE = 0x2C00
    
    # Status and Control
    S_STATUS            = BASE + 0x00  # 0x2C00
    S_POINTER           = BASE + 0x01  # 0x2C01
    D_OP_ENABLE         = BASE + 0x02  # 0x2C02
    
    # Input Data Cube Configuration
    D_DATA_CUBE_IN_WIDTH    = BASE + 0x03  # 0x2C03
    D_DATA_CUBE_IN_HEIGHT   = BASE + 0x04  # 0x2C04
    D_DATA_CUBE_IN_CHANNEL  = BASE + 0x05  # 0x2C05
    
    # Output Data Cube Configuration
    D_DATA_CUBE_OUT_WIDTH   = BASE + 0x06  # 0x2C06
    D_DATA_CUBE_OUT_HEIGHT  = BASE + 0x07  # 0x2C07
    D_DATA_CUBE_OUT_CHANNEL = BASE + 0x08  # 0x2C08
    
    # Operation Configuration
    D_OPERATION_MODE_CFG    = BASE + 0x09  # 0x2C09
    D_NAN_FLUSH_TO_ZERO     = BASE + 0x0A  # 0x2C0A
    
    # Partial Width Configuration
    D_PARTIAL_WIDTH_IN      = BASE + 0x0B  # 0x2C0B
    D_PARTIAL_WIDTH_OUT     = BASE + 0x0C  # 0x2C0C
    
    # Pooling Kernel Configuration
    D_POOLING_KERNEL_CFG    = BASE + 0x0D  # 0x2C0D
    D_RECIP_KERNEL_HEIGHT   = BASE + 0x0F  # 0x2C0F
    D_RECIP_KERNEL_WIDTH    = BASE + 0x0E  # 0x2C0E
    
    # Pooling Padding Configuration
    D_POOLING_PADDING_CFG           = BASE + 0x10  # 0x2C10
    D_POOLING_PADDING_VALUE_1_CFG   = BASE + 0x11  # 0x2C11
    D_POOLING_PADDING_VALUE_2_CFG   = BASE + 0x12  # 0x2C12
    D_POOLING_PADDING_VALUE_3_CFG   = BASE + 0x13  # 0x2C13
    D_POOLING_PADDING_VALUE_4_CFG   = BASE + 0x14  # 0x2C14
    D_POOLING_PADDING_VALUE_5_CFG   = BASE + 0x15  # 0x2C15
    D_POOLING_PADDING_VALUE_6_CFG   = BASE + 0x16  # 0x2C16
    D_POOLING_PADDING_VALUE_7_CFG   = BASE + 0x17  # 0x2C17
    
    # Source Memory Configuration
    D_SRC_BASE_ADDR_LOW     = BASE + 0x18  # 0x2C18
    D_SRC_BASE_ADDR_HIGH    = BASE + 0x19  # 0x2C19
    D_SRC_LINE_STRIDE       = BASE + 0x1A  # 0x2C1A
    D_SRC_SURFACE_STRIDE    = BASE + 0x1B  # 0x2C1B
    
    # Destination Memory Configuration
    D_DST_BASE_ADDR_LOW     = BASE + 0x1C  # 0x2C1C
    D_DST_BASE_ADDR_HIGH    = BASE + 0x1D  # 0x2C1D
    D_DST_LINE_STRIDE       = BASE + 0x1E  # 0x2C1E
    D_DST_SURFACE_STRIDE    = BASE + 0x1F  # 0x2C1F
    D_DST_RAM_CFG           = BASE + 0x20  # 0x2C20
    
    # Data Format
    D_DATA_FORMAT           = BASE + 0x21  # 0x2C21
    
    # Statistics and Debug
    D_INF_INPUT_NUM         = BASE + 0x22  # 0x2C22
    D_NAN_INPUT_NUM         = BASE + 0x23  # 0x2C23
    D_NAN_OUTPUT_NUM        = BASE + 0x24  # 0x2C24
    
    # Performance Monitoring
    D_PERF_ENABLE           = BASE + 0x25  # 0x2C25
    D_PERF_WRITE_STALL      = BASE + 0x26  # 0x2C26
    
    # Debug/CYA
    D_CYA                   = BASE + 0x27  # 0x2C27


# PDP RDMA Registers (Base: 0x2800)
class PDP_RDMA_REG:
    """PDP RDMA (Read DMA) Register Addresses"""
    BASE = 0x2800
    
    # Status and Control
    S_STATUS            = BASE + 0x00  # 0x2800
    S_POINTER           = BASE + 0x01  # 0x2801
    D_OP_ENABLE         = BASE + 0x02  # 0x2802
    
    # Input Data Cube Configuration
    D_DATA_CUBE_IN_WIDTH    = BASE + 0x03  # 0x2803
    D_DATA_CUBE_IN_HEIGHT   = BASE + 0x04  # 0x2804
    D_DATA_CUBE_IN_CHANNEL  = BASE + 0x05  # 0x2805
    
    # Flying Mode
    D_FLYING_MODE           = BASE + 0x06  # 0x2806
    
    # Source Memory Configuration
    D_SRC_BASE_ADDR_LOW     = BASE + 0x07  # 0x2807
    D_SRC_BASE_ADDR_HIGH    = BASE + 0x08  # 0x2808
    D_SRC_LINE_STRIDE       = BASE + 0x09  # 0x2809
    D_SRC_SURFACE_STRIDE    = BASE + 0x0A  # 0x280A
    D_SRC_RAM_CFG           = BASE + 0x0B  # 0x280B
    
    # Data Format and Operation
    D_DATA_FORMAT           = BASE + 0x0C  # 0x280C
    D_OPERATION_MODE_CFG    = BASE + 0x0D  # 0x280D
    
    # Pooling Configuration
    D_POOLING_KERNEL_CFG    = BASE + 0x0E  # 0x280E
    D_POOLING_PADDING_CFG   = BASE + 0x0F  # 0x280F
    
    # Partial Width
    D_PARTIAL_WIDTH_IN      = BASE + 0x10  # 0x2810
    
    # Performance Monitoring
    D_PERF_ENABLE           = BASE + 0x11  # 0x2811
    D_PERF_READ_STALL       = BASE + 0x12  # 0x2812
    
    # Debug/CYA
    D_CYA                   = BASE + 0x13  # 0x2813
