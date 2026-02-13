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
