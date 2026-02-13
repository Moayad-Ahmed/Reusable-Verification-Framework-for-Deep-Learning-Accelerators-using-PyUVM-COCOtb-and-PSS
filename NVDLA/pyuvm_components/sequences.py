from pyuvm import *
from pyuvm_components.seq_item import PdpTransaction


class PdpTestSequence(uvm_sequence):
    def __init__(self, name, input_file=None):
        super().__init__(name)
        self.input_file = input_file

    async def body(self):
        seq_item = PdpTransaction("pdp_tx")

        # ----- Input DRAM data -----
        seq_item.input_file = self.input_file
        seq_item.input_base_addr = 0
        seq_item.input_byte_count = len(self.input_file)

        # ----- PDP + PDP-RDMA register writes -----
        seq_item.register_writes = [
            # -- PDP registers --
            (0x2C01, 0x00000000),   # NVDLA_PDP_S_POINTER_0
            (0x2C10, 0x00000022),   # NVDLA_PDP_D_POOLING_PADDING_CFG_0
            (0x2C1D, 0x00000000),   # NVDLA_PDP_D_DST_BASE_ADDR_HIGH_0
            (0x2C09, 0x00000010),   # NVDLA_PDP_D_OPERATION_MODE_CFG_0
            (0x2C11, 0x00000010),   # NVDLA_PDP_D_POOLING_PADDING_VALUE_1_CFG_0
            (0x2C0D, 0x00110202),   # NVDLA_PDP_D_POOLING_KERNEL_CFG_0
            (0x2C07, 0x00000000),   # NVDLA_PDP_D_DATA_CUBE_OUT_HEIGHT_0
            (0x2C20, 0x00000001),   # NVDLA_PDP_D_DST_RAM_CFG_0
            (0x2C04, 0x00000000),   # NVDLA_PDP_D_DATA_CUBE_IN_HEIGHT_0
            (0x2C16, 0x00000060),   # NVDLA_PDP_D_POOLING_PADDING_VALUE_6_CFG_0
            (0x2C26, 0x00000000),   # NVDLA_PDP_D_PERF_WRITE_STALL_0
            (0x2C03, 0x00000000),   # NVDLA_PDP_D_DATA_CUBE_IN_WIDTH_0
            (0x2C0B, 0x27C9B90A),   # NVDLA_PDP_D_PARTIAL_WIDTH_IN_0
            (0x2C15, 0x00000050),   # NVDLA_PDP_D_POOLING_PADDING_VALUE_5_CFG_0
            (0x2C1F, 0x000026C0),   # NVDLA_PDP_D_DST_SURFACE_STRIDE_0
            (0x2C0F, 0x00005555),   # NVDLA_PDP_D_RECIP_KERNEL_HEIGHT_0
            (0x2C1B, 0x00000D00),   # NVDLA_PDP_D_SRC_SURFACE_STRIDE_0
            (0x2C05, 0x00000000),   # NVDLA_PDP_D_DATA_CUBE_IN_CHANNEL_0
            (0x2C21, 0x00000000),   # NVDLA_PDP_D_DATA_FORMAT_0
            (0x2C06, 0x00000000),   # NVDLA_PDP_D_DATA_CUBE_OUT_WIDTH_0
            (0x2C27, 0x19D97A33),   # NVDLA_PDP_D_CYA_0
            (0x2C13, 0x00000030),   # NVDLA_PDP_D_POOLING_PADDING_VALUE_3_CFG_0
            (0x2C1C, 0x00000100),   # NVDLA_PDP_D_DST_BASE_ADDR_LOW_0   (256)
            (0x2C14, 0x00000040),   # NVDLA_PDP_D_POOLING_PADDING_VALUE_4_CFG_0
            (0x2C24, 0x00000000),   # NVDLA_PDP_D_NAN_OUTPUT_NUM_0
            (0x2C08, 0x00000000),   # NVDLA_PDP_D_DATA_CUBE_OUT_CHANNEL_0
            (0x2C0E, 0x00005555),   # NVDLA_PDP_D_RECIP_KERNEL_WIDTH_0
            (0x2C1A, 0x000003A0),   # NVDLA_PDP_D_SRC_LINE_STRIDE_0
            (0x2C22, 0x00000000),   # NVDLA_PDP_D_INF_INPUT_NUM_0
            (0x2C1E, 0x00002000),   # NVDLA_PDP_D_DST_LINE_STRIDE_0
            (0x2C0A, 0x00000000),   # NVDLA_PDP_D_NAN_FLUSH_TO_ZERO_0
            (0x2C0C, 0x2C571256),   # NVDLA_PDP_D_PARTIAL_WIDTH_OUT_0
            (0x2C12, 0x00000020),   # NVDLA_PDP_D_POOLING_PADDING_VALUE_2_CFG_0
            (0x2C18, 0x00000000),   # NVDLA_PDP_D_SRC_BASE_ADDR_LOW_0
            (0x2C23, 0x00000000),   # NVDLA_PDP_D_NAN_INPUT_NUM_0
            (0x2C19, 0x00000000),   # NVDLA_PDP_D_SRC_BASE_ADDR_HIGH_0
            (0x2C25, 0x00000001),   # NVDLA_PDP_D_PERF_ENABLE_0
            (0x2C17, 0x00000070),   # NVDLA_PDP_D_POOLING_PADDING_VALUE_7_CFG_0
            (0x2C00, 0x00000000),   # NVDLA_PDP_S_STATUS_0
            # -- PDP RDMA registers --
            (0x2801, 0x00000000),   # NVDLA_PDP_RDMA_S_POINTER_0
            (0x2805, 0x00000000),   # NVDLA_PDP_RDMA_D_DATA_CUBE_IN_CHANNEL_0
            (0x2810, 0x27C9B90A),   # NVDLA_PDP_RDMA_D_PARTIAL_WIDTH_IN_0
            (0x2807, 0x00000000),   # NVDLA_PDP_RDMA_D_SRC_BASE_ADDR_LOW_0
            (0x280C, 0x00000000),   # NVDLA_PDP_RDMA_D_DATA_FORMAT_0
            (0x2803, 0x00000000),   # NVDLA_PDP_RDMA_D_DATA_CUBE_IN_WIDTH_0
            (0x280F, 0x00000006),   # NVDLA_PDP_RDMA_D_POOLING_PADDING_CFG_0
            (0x2812, 0x00000000),   # NVDLA_PDP_RDMA_D_PERF_READ_STALL_0
            (0x280D, 0x00000000),   # NVDLA_PDP_RDMA_D_OPERATION_MODE_CFG_0
            (0x2811, 0x00000000),   # NVDLA_PDP_RDMA_D_PERF_ENABLE_0
            (0x2800, 0x00000000),   # NVDLA_PDP_RDMA_S_STATUS_0
            (0x2806, 0x00000001),   # NVDLA_PDP_RDMA_D_FLYING_MODE_0
            (0x2809, 0x000003A0),   # NVDLA_PDP_RDMA_D_SRC_LINE_STRIDE_0
            (0x280B, 0x00000001),   # NVDLA_PDP_RDMA_D_SRC_RAM_CFG_0
            (0x2813, 0x81E7F8F3),   # NVDLA_PDP_RDMA_D_CYA_0
            (0x280E, 0x00000012),   # NVDLA_PDP_RDMA_D_POOLING_KERNEL_CFG_0
            (0x2808, 0x00000000),   # NVDLA_PDP_RDMA_D_SRC_BASE_ADDR_HIGH_0
            (0x2804, 0x00000000),   # NVDLA_PDP_RDMA_D_DATA_CUBE_IN_HEIGHT_0
            (0x280A, 0x00000D00),   # NVDLA_PDP_RDMA_D_SRC_SURFACE_STRIDE_0
            # -- Enable operations --
            (0x2C02, 0x00000001),   # NVDLA_PDP_D_OP_ENABLE_0
            (0x2802, 0x00000001),   # NVDLA_PDP_RDMA_D_OP_ENABLE_0
        ]

        # ----- Expected output info -----
        seq_item.output_base_addr = 0x100     # DST_BASE_ADDR_LOW
        seq_item.output_length = 8

        seq_item.expected_crc = 0x1DC317C3
        seq_item.expected_output_data = [0x12, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E]

        await self.start_item(seq_item)
        await self.finish_item(seq_item)