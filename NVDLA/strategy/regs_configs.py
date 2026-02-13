from strategy.Layers_regs_addresses import PDP_REG 
from strategy.Layers_regs_addresses import PDP_RDMA_REG
    
    


class RegistrationConfigs:
    def pooling_configs(self, layer_configs):
        return [  
            # PDP CORE - Status and Control
            (PDP_REG.S_STATUS, 0x00000000),
            (PDP_REG.S_POINTER, 0x00000000),
            
            # PDP CORE - Input Data Cube
            (PDP_REG.D_DATA_CUBE_IN_WIDTH, 0x00000000),
            (PDP_REG.D_DATA_CUBE_IN_HEIGHT, 0x00000000),
            (PDP_REG.D_DATA_CUBE_IN_CHANNEL, 0x00000000),
            
            # PDP CORE - Output Data Cube
            (PDP_REG.D_DATA_CUBE_OUT_WIDTH, 0x00000000),
            (PDP_REG.D_DATA_CUBE_OUT_HEIGHT, 0x00000000),
            (PDP_REG.D_DATA_CUBE_OUT_CHANNEL, 0x00000000),
            
            # PDP CORE - Operation Mode
            (PDP_REG.D_OPERATION_MODE_CFG, 0x00000010),
            (PDP_REG.D_NAN_FLUSH_TO_ZERO, 0x00000000),
            
            # PDP CORE - Partial Width
            (PDP_REG.D_PARTIAL_WIDTH_IN, 0x27C9B90A),
            (PDP_REG.D_PARTIAL_WIDTH_OUT, 0x2C571256),
            
            # PDP CORE - Pooling Kernel
            (PDP_REG.D_POOLING_KERNEL_CFG, 0x00110202),
            (PDP_REG.D_RECIP_KERNEL_HEIGHT, 0x00005555),
            (PDP_REG.D_RECIP_KERNEL_WIDTH, 0x00005555),
            
            # PDP CORE - Pooling Padding
            (PDP_REG.D_POOLING_PADDING_CFG, 0x00000022),
            (PDP_REG.D_POOLING_PADDING_VALUE_1_CFG, 0x00000010),
            (PDP_REG.D_POOLING_PADDING_VALUE_2_CFG, 0x00000020),
            (PDP_REG.D_POOLING_PADDING_VALUE_3_CFG, 0x00000030),
            (PDP_REG.D_POOLING_PADDING_VALUE_4_CFG, 0x00000040),
            (PDP_REG.D_POOLING_PADDING_VALUE_5_CFG, 0x00000050),
            (PDP_REG.D_POOLING_PADDING_VALUE_6_CFG, 0x00000060),
            (PDP_REG.D_POOLING_PADDING_VALUE_7_CFG, 0x00000070),
            
            # PDP CORE - Source Memory
            (PDP_REG.D_SRC_BASE_ADDR_LOW, 0x00000000),
            (PDP_REG.D_SRC_BASE_ADDR_HIGH, 0x00000000),
            (PDP_REG.D_SRC_LINE_STRIDE, 0x000003A0),
            (PDP_REG.D_SRC_SURFACE_STRIDE, 0x00000D00),
            
            # PDP CORE - Destination Memory
            (PDP_REG.D_DST_BASE_ADDR_LOW, 0x00000100),    # 256
            (PDP_REG.D_DST_BASE_ADDR_HIGH, 0x00000000),
            (PDP_REG.D_DST_LINE_STRIDE, 0x00002000),
            (PDP_REG.D_DST_SURFACE_STRIDE, 0x000026C0),
            (PDP_REG.D_DST_RAM_CFG, 0x00000001),
            
            # PDP CORE - Data Format
            (PDP_REG.D_DATA_FORMAT, 0x00000000),
            
            # PDP CORE - Statistics
            (PDP_REG.D_INF_INPUT_NUM, 0x00000000),
            (PDP_REG.D_NAN_INPUT_NUM, 0x00000000),
            (PDP_REG.D_NAN_OUTPUT_NUM, 0x00000000),
            
            # PDP CORE - Performance Monitoring
            (PDP_REG.D_PERF_ENABLE, 0x00000001),
            (PDP_REG.D_PERF_WRITE_STALL, 0x00000000),
            
            # PDP CORE - Debug
            (PDP_REG.D_CYA, 0x19D97A33),
            
            # PDP RDMA - Status and Control
            (PDP_RDMA_REG.S_STATUS, 0x00000000),
            (PDP_RDMA_REG.S_POINTER, 0x00000000),
            
            # PDP RDMA - Input Data Cube
            (PDP_RDMA_REG.D_DATA_CUBE_IN_WIDTH, 0x00000000),
            (PDP_RDMA_REG.D_DATA_CUBE_IN_HEIGHT, 0x00000000),
            (PDP_RDMA_REG.D_DATA_CUBE_IN_CHANNEL, 0x00000000),
            
            # PDP RDMA - Flying Mode
            (PDP_RDMA_REG.D_FLYING_MODE, 0x00000001),
            
            # PDP RDMA - Source Memory
            (PDP_RDMA_REG.D_SRC_BASE_ADDR_LOW, 0x00000000),
            (PDP_RDMA_REG.D_SRC_BASE_ADDR_HIGH, 0x00000000),
            (PDP_RDMA_REG.D_SRC_LINE_STRIDE, 0x000003A0),
            (PDP_RDMA_REG.D_SRC_SURFACE_STRIDE, 0x00000D00),
            (PDP_RDMA_REG.D_SRC_RAM_CFG, 0x00000001),
            
            # PDP RDMA - Data Format and Operation
            (PDP_RDMA_REG.D_DATA_FORMAT, 0x00000000),
            (PDP_RDMA_REG.D_OPERATION_MODE_CFG, 0x00000000),
            
            # PDP RDMA - Pooling Configuration
            (PDP_RDMA_REG.D_POOLING_KERNEL_CFG, 0x00000012),
            (PDP_RDMA_REG.D_POOLING_PADDING_CFG, 0x00000006),
            (PDP_RDMA_REG.D_PARTIAL_WIDTH_IN, 0x27C9B90A),
            
            # PDP RDMA - Performance Monitoring
            (PDP_RDMA_REG.D_PERF_ENABLE, 0x00000000),
            (PDP_RDMA_REG.D_PERF_READ_STALL, 0x00000000),
            
            # PDP RDMA - Debug
            (PDP_RDMA_REG.D_CYA, 0x81E7F8F3),
            
            # ENABLE OPERATIONS (Must be last!)
            (PDP_REG.D_OP_ENABLE, 0x00000001),
            (PDP_RDMA_REG.D_OP_ENABLE, 0x00000001)
          ]   # NVDLA_PDP_RDMA_D_OP_ENABLE_0
    
    def conv_configs(self):
        pass
    
    def fullyConnected_configs(self):
        pass
    
    def activation_configs(self):
        pass
    
    def normalization_configs(self):
        pass
    
    def regularization_configs(self):
        pass