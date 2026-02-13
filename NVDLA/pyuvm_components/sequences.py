from pyuvm import *
from pyuvm_components.seq_item import PdpTransaction
import binascii

# ---- CRC Calculation ----
def calc_crc32(data_bytes: list) -> int:
    """Standard CRC-32 over a list of byte values."""
    return binascii.crc32(bytes(data_bytes))


class PdpTestSequence(uvm_sequence):
    def __init__(self, name, input_file=None):
        super().__init__(name)
        self.input_file = input_file

    async def body(self):
        seq_item = PdpTransaction("pdp_tx")

        # ----- PDP + PDP-RDMA register writes -----
        # Config: 3x3 input, 2x2 kernel, stride 1, no padding, average pooling, INT8, 1 channel
        # All dimension/size fields are 0-based (reg = actual - 1)
        seq_item.register_writes = {
            # -- PDP registers --
            0x2C00: 0x00000000,   # NVDLA_PDP_S_STATUS_0
            0x2C01: 0x00000000,   # NVDLA_PDP_S_POINTER_0
            0x2C03: 0x00000002,   # NVDLA_PDP_D_DATA_CUBE_IN_WIDTH_0       (3-1=2)
            0x2C04: 0x00000002,   # NVDLA_PDP_D_DATA_CUBE_IN_HEIGHT_0      (3-1=2)
            0x2C05: 0x00000000,   # NVDLA_PDP_D_DATA_CUBE_IN_CHANNEL_0     (1-1=0)
            0x2C06: 0x00000001,   # NVDLA_PDP_D_DATA_CUBE_OUT_WIDTH_0      (2-1=1)
            0x2C07: 0x00000001,   # NVDLA_PDP_D_DATA_CUBE_OUT_HEIGHT_0     (2-1=1)
            0x2C08: 0x00000000,   # NVDLA_PDP_D_DATA_CUBE_OUT_CHANNEL_0    (1-1=0)
            0x2C09: 0x00000010,   # NVDLA_PDP_D_OPERATION_MODE_CFG_0       (avg, off-flying, no split)
            0x2C0A: 0x00000000,   # NVDLA_PDP_D_NAN_FLUSH_TO_ZERO_0
            0x2C0B: 0x00000002,   # NVDLA_PDP_D_PARTIAL_WIDTH_IN_0         (first=2, non-split)
            0x2C0C: 0x00000001,   # NVDLA_PDP_D_PARTIAL_WIDTH_OUT_0        (first=1, non-split)
            0x2C0D: 0x00000101,   # NVDLA_PDP_D_POOLING_KERNEL_CFG_0       (kw=1→2, kh=1→2, sw=0→1, sh=0→1)
            0x2C0E: 0x00008000,   # NVDLA_PDP_D_RECIP_KERNEL_WIDTH_0       (1/2 in Q0.16 = 0x8000)
            0x2C0F: 0x00008000,   # NVDLA_PDP_D_RECIP_KERNEL_HEIGHT_0      (1/2 in Q0.16 = 0x8000)
            0x2C10: 0x00000000,   # NVDLA_PDP_D_POOLING_PADDING_CFG_0      (no padding)
            0x2C11: 0x00000000,   # NVDLA_PDP_D_POOLING_PADDING_VALUE_1_CFG_0
            0x2C12: 0x00000000,   # NVDLA_PDP_D_POOLING_PADDING_VALUE_2_CFG_0
            0x2C13: 0x00000000,   # NVDLA_PDP_D_POOLING_PADDING_VALUE_3_CFG_0
            0x2C14: 0x00000000,   # NVDLA_PDP_D_POOLING_PADDING_VALUE_4_CFG_0
            0x2C15: 0x00000000,   # NVDLA_PDP_D_POOLING_PADDING_VALUE_5_CFG_0
            0x2C16: 0x00000000,   # NVDLA_PDP_D_POOLING_PADDING_VALUE_6_CFG_0
            0x2C17: 0x00000000,   # NVDLA_PDP_D_POOLING_PADDING_VALUE_7_CFG_0
            0x2C18: 0x00000000,   # NVDLA_PDP_D_SRC_BASE_ADDR_LOW_0
            0x2C19: 0x00000000,   # NVDLA_PDP_D_SRC_BASE_ADDR_HIGH_0
            0x2C1A: 0x00000018,   # NVDLA_PDP_D_SRC_LINE_STRIDE_0          (3 elem × 8 B = 24)
            0x2C1B: 0x00000048,   # NVDLA_PDP_D_SRC_SURFACE_STRIDE_0       (24 × 3 = 72)
            0x2C1C: 0x00000100,   # NVDLA_PDP_D_DST_BASE_ADDR_LOW_0        (256)
            0x2C1D: 0x00000000,   # NVDLA_PDP_D_DST_BASE_ADDR_HIGH_0
            0x2C1E: 0x00000010,   # NVDLA_PDP_D_DST_LINE_STRIDE_0          (2 elem × 8 B = 16)
            0x2C1F: 0x00000020,   # NVDLA_PDP_D_DST_SURFACE_STRIDE_0       (16 × 2 = 32)
            0x2C20: 0x00000001,   # NVDLA_PDP_D_DST_RAM_CFG_0
            0x2C21: 0x00000000,   # NVDLA_PDP_D_DATA_FORMAT_0              (INT8)
            0x2C22: 0x00000000,   # NVDLA_PDP_D_INF_INPUT_NUM_0
            0x2C23: 0x00000000,   # NVDLA_PDP_D_NAN_INPUT_NUM_0
            0x2C24: 0x00000000,   # NVDLA_PDP_D_NAN_OUTPUT_NUM_0
            0x2C25: 0x00000001,   # NVDLA_PDP_D_PERF_ENABLE_0
            0x2C26: 0x00000000,   # NVDLA_PDP_D_PERF_WRITE_STALL_0
            0x2C27: 0x00000000,   # NVDLA_PDP_D_CYA_0
            # -- PDP RDMA registers --
            0x2800: 0x00000000,   # NVDLA_PDP_RDMA_S_STATUS_0
            0x2801: 0x00000000,   # NVDLA_PDP_RDMA_S_POINTER_0
            0x2803: 0x00000002,   # NVDLA_PDP_RDMA_D_DATA_CUBE_IN_WIDTH_0  (3-1=2)
            0x2804: 0x00000002,   # NVDLA_PDP_RDMA_D_DATA_CUBE_IN_HEIGHT_0 (3-1=2)
            0x2805: 0x00000000,   # NVDLA_PDP_RDMA_D_DATA_CUBE_IN_CHANNEL_0 (1-1=0)
            0x2806: 0x00000001,   # NVDLA_PDP_RDMA_D_FLYING_MODE_0         (off-flying)
            0x2807: 0x00000000,   # NVDLA_PDP_RDMA_D_SRC_BASE_ADDR_LOW_0
            0x2808: 0x00000000,   # NVDLA_PDP_RDMA_D_SRC_BASE_ADDR_HIGH_0
            0x2809: 0x00000018,   # NVDLA_PDP_RDMA_D_SRC_LINE_STRIDE_0     (24)
            0x280A: 0x00000048,   # NVDLA_PDP_RDMA_D_SRC_SURFACE_STRIDE_0  (72)
            0x280B: 0x00000001,   # NVDLA_PDP_RDMA_D_SRC_RAM_CFG_0
            0x280C: 0x00000000,   # NVDLA_PDP_RDMA_D_DATA_FORMAT_0         (INT8)
            0x280D: 0x00000000,   # NVDLA_PDP_RDMA_D_OPERATION_MODE_CFG_0  (no split)
            0x280E: 0x00000001,   # NVDLA_PDP_RDMA_D_POOLING_KERNEL_CFG_0  (kw=1→2, sw=0→1)
            0x280F: 0x00000000,   # NVDLA_PDP_RDMA_D_POOLING_PADDING_CFG_0 (no padding)
            0x2810: 0x00000002,   # NVDLA_PDP_RDMA_D_PARTIAL_WIDTH_IN_0    (first=2)
            0x2811: 0x00000000,   # NVDLA_PDP_RDMA_D_PERF_ENABLE_0
            0x2812: 0x00000000,   # NVDLA_PDP_RDMA_D_PERF_READ_STALL_0
            0x2813: 0x00000000,   # NVDLA_PDP_RDMA_D_CYA_0
            # -- Enable operations (must be last) --
            0x2C02: 0x00000001,   # NVDLA_PDP_D_OP_ENABLE_0
            0x2802: 0x00000001,   # NVDLA_PDP_RDMA_D_OP_ENABLE_0
        }

        # ----- Input DRAM data -----
        seq_item.input_file = self.input_file

        src_base_addr_low = seq_item.register_writes[0x2C18]
        src_base_addr_high = seq_item.register_writes[0x2C19]
        seq_item.input_base_addr = (src_base_addr_high << 32) | src_base_addr_low

        # ----- Expected output info -----
        dst_base_addr_low = seq_item.register_writes[0x2C1C]
        dst_base_addr_high = seq_item.register_writes[0x2C1D]
        seq_item.output_base_addr = (dst_base_addr_high << 32) | dst_base_addr_low

        seq_item.output_length = 32           # 4 elements × 8 bytes each

        # 2×2 output, each element = 8 bytes (1 data byte + 7 zeros)
        # avg(10,10,20,20)=15  avg(10,20,20,10)=15
        # avg(20,20,20,20)=20  avg(20,10,20,10)=15
        seq_item.expected_output_data = [
            0x0F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # (0,0) = 15
            0x0F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # (0,1) = 15
            0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # (1,0) = 20
            0x0F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # (1,1) = 15
        ]
        seq_item.expected_crc = calc_crc32(seq_item.expected_output_data)

        await self.start_item(seq_item)
        await self.finish_item(seq_item)