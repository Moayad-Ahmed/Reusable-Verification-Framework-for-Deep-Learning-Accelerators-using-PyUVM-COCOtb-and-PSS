import logging
import binascii

import cocotb
from cocotb.queue import Queue
from cocotb.triggers import RisingEdge, Timer
from cocotbext.axi import AxiMaster, AxiBus

from pyuvm import utility_classes

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NvdlaBFM(metaclass=utility_classes.Singleton):
    """
    NVDLA Bus Functional Model

    Provides:
      - Reset control
      - AXI master for DRAM access
      - DRAM memory write / read
      - CSB register write / read / poll
      - Interrupt wait
      - CRC-32 calculation

    Inter-component synchronization queues
    (used by the split DataAgent / CsbAgent architecture):

    ┌──────────────────────────┬──────────────────┬────────────────────────────────────────┐
    │ Queue                    │ Producer         │ Consumer                               │
    ├──────────────────────────┼──────────────────┼────────────────────────────────────────┤
    │ data_ready_queue         │ DataDriver       │ CsbDriver  (gate reg writes)           │
    │ data_observed_queue      │ DataDriver       │ DataMonitor (passive observation)      │
    │ output_config_queue      │ CsbDriver        │ CsbMonitor (output address / layout)   │
    │ iteration_done_queue     │ Scoreboard       │ Virtual sequence (gate next iteration) │
    └──────────────────────────┴──────────────────┴────────────────────────────────────────┘
    """

    def __init__(self):
        self.dut = cocotb.top

        # ---- NEW: DataDriver → CsbDriver ----
        # Signals that all DRAM data (input + weights) is fully written.
        # CsbDriver blocks on this before issuing any register writes so the
        # hardware never starts reading DRAM before data is present.
        self.data_ready_queue = Queue(maxsize=0)

        # ---- NEW: DataDriver → DataMonitor ----
        # A separate copy of the data-load event so the passive DataMonitor
        # can observe it without consuming the token CsbDriver is waiting for.
        self.data_observed_queue = Queue(maxsize=0)

        # ---- EXISTING: CsbDriver → CsbMonitor ----
        # Carries the CsbTransaction so the monitor knows the output DRAM
        # address, pixel layout, and expected (golden) bytes.
        self.output_config_queue = Queue(maxsize=0)

        # ---- EXISTING: Scoreboard → Virtual sequence ----
        # One token per completed check; prevents the next iteration from
        # starting before the current one is fully verified.
        self.iteration_done_queue = Queue(maxsize=0)

        self.axi_master = None

    # ══════════════════════════════════════════════════════════════════
    #  Reset Control
    # ══════════════════════════════════════════════════════════════════

    async def reset(self):
        """Reset NVDLA to initialize all signals to known values"""
        dut = self.dut

        logger.info("Reset asserted")

        # Control signals
        dut.test_mode.value = 0
        dut.global_clk_ovr_on.value = 0
        dut.tmc2slcg_disable_clock_gating.value = 0
        dut.dla_reset_rstn.value = 0
        dut.direct_reset_.value = 1

        # CSB interface idle
        dut.csb2nvdla_valid.value = 0
        dut.csb2nvdla_addr.value = 0
        dut.csb2nvdla_wdat.value = 0
        dut.csb2nvdla_write.value = 0
        dut.csb2nvdla_nposted.value = 0

        # Power-bus tied to zero
        dut.nvdla_pwrbus_ram_c_pd.value = 0
        dut.nvdla_pwrbus_ram_ma_pd.value = 0
        dut.nvdla_pwrbus_ram_mb_pd.value = 0
        dut.nvdla_pwrbus_ram_p_pd.value = 0
        dut.nvdla_pwrbus_ram_o_pd.value = 0
        dut.nvdla_pwrbus_ram_a_pd.value = 0

        # ext2dbb signals
        dut.ext2dbb_awvalid.value = 0
        dut.ext2dbb_awlen.value = 0
        dut.ext2dbb_awsize.value = 0
        dut.ext2dbb_awburst.value = 0
        dut.ext2dbb_awaddr.value = 0
        dut.ext2dbb_awid.value = 0
        dut.ext2dbb_wvalid.value = 0
        dut.ext2dbb_wdata.value = 0
        dut.ext2dbb_wlast.value = 0
        dut.ext2dbb_wstrb.value = 0
        dut.ext2dbb_bready.value = 1
        dut.ext2dbb_arvalid.value = 0
        dut.ext2dbb_arlen.value = 0
        dut.ext2dbb_arsize.value = 0
        dut.ext2dbb_arburst.value = 0
        dut.ext2dbb_araddr.value = 0
        dut.ext2dbb_arid.value = 0
        dut.ext2dbb_rready.value = 1

        # Hold reset for 10000 ns
        await Timer(10000, unit="ns")

        dut.dla_reset_rstn.value = 1
        dut.direct_reset_.value = 1
        logger.info("Reset deasserted")

        # Wait some time to observe stable state
        await Timer(5000, unit="ns")

        # Create AXI master now that clocks are running and signals are stable
        await self.init_axi()

    # ══════════════════════════════════════════════════════════════════
    #  AXI Master
    # ══════════════════════════════════════════════════════════════════

    async def init_axi(self):
        """Create the AXI master after reset is established and signals are stable."""
        if self.axi_master is None:
            self.axi_master = AxiMaster(
                AxiBus.from_prefix(self.dut, "ext2dbb"),
                self.dut.dla_core_clk,
                self.dut.dla_reset_rstn,
                reset_active_level=False,
            )

    # ══════════════════════════════════════════════════════════════════
    #  DRAM Memory Access
    # ══════════════════════════════════════════════════════════════════

    async def write_in_dram(self, input_data_path: str, base_addr: int):
        """
        Load a hex file into simulated DRAM at base_addr.

        Each line of the file is one unsigned byte in two-digit hex.
        Bytes are grouped into 8-byte little-endian qwords and written
        via the AXI master.

        Args:
            input_data_path : path to the .dat hex file
            base_addr       : byte offset in DRAM where data should start
        """
        with open(input_data_path, "r") as file:
            lines = [ln.strip() for ln in file if ln.strip()]

        byte_data = [int(ln, 16) for ln in lines]

        num_qwords = len(byte_data) // 8
        for i in range(num_qwords):
            qword = 0
            for j in range(8):
                qword |= byte_data[i * 8 + j] << (j * 8)
            await self.axi_master.write_qword(base_addr + i * 8, qword)

        await RisingEdge(self.dut.dla_core_clk)

    async def read_from_dram(
        self,
        base_addr: int,
        num_pixels: int,
        pixel_bytes: int = 8,
        data_bytes_per_pixel: int = 1,
    ):
        """
        Read output data from DRAM, extracting all channel bytes per pixel.

        NVDLA stores each spatial pixel in ``pixel_bytes`` (atom-aligned).
        The first ``data_bytes_per_pixel`` bytes of each pixel contain
        actual channel data; the rest is padding and is discarded.

        Args:
            base_addr             : DRAM start address of the output surface
            num_pixels            : number of spatial output pixels (H × W)
            pixel_bytes           : atom-aligned stride per pixel in DRAM
            data_bytes_per_pixel  : real data bytes per pixel (channels × bpe)

        Returns:
            list[int]: flat list of data bytes in pixel-major, channel-minor order
        """
        actual_output = []

        for i in range(num_pixels):
            pixel_addr = base_addr + i * pixel_bytes
            for ch in range(data_bytes_per_pixel):
                data = await self.axi_master.read_byte(pixel_addr + ch)
                actual_output.append(data)

        await RisingEdge(self.dut.dla_core_clk)
        return actual_output

    # ══════════════════════════════════════════════════════════════════
    #  CSB Register Access
    # ══════════════════════════════════════════════════════════════════

    async def reg_write(self, addr: int, data: int):
        """Non-posted CSB register write. Waits for ready then wr_complete."""
        dut = self.dut

        dut.csb2nvdla_valid.value = 1
        dut.csb2nvdla_write.value = 1
        dut.csb2nvdla_nposted.value = 1
        dut.csb2nvdla_addr.value = addr
        dut.csb2nvdla_wdat.value = data

        await RisingEdge(dut.dla_csb_clk)
        while dut.csb2nvdla_ready.value != 1:
            await RisingEdge(dut.dla_csb_clk)

        dut.csb2nvdla_valid.value = 0
        dut.csb2nvdla_write.value = 0

        await RisingEdge(dut.dla_csb_clk)
        while dut.nvdla2csb_wr_complete.value != 1:
            await RisingEdge(dut.dla_csb_clk)

        logger.info("CSB WRITE: addr=0x%04x  data=0x%08x", addr, data)

    async def reg_read(self, addr: int) -> int:
        """CSB register read. Returns the 32-bit register value."""
        dut = self.dut

        dut.csb2nvdla_valid.value = 1
        dut.csb2nvdla_write.value = 0
        dut.csb2nvdla_addr.value = addr

        await RisingEdge(dut.dla_csb_clk)
        while dut.csb2nvdla_ready.value != 1:
            await RisingEdge(dut.dla_csb_clk)

        dut.csb2nvdla_valid.value = 0

        await RisingEdge(dut.dla_csb_clk)
        while dut.nvdla2csb_valid.value != 1:
            await RisingEdge(dut.dla_csb_clk)

        data = int(dut.nvdla2csb_data.value)
        logger.info("CSB READ:  addr=0x%04x -> data=0x%08x", addr, data)
        return data

    async def poll_reg(self, addr: int, expected: int, timeout_cycles: int = 100000):
        """
        Poll a CSB register until its value equals expected.

        Used for CBUF credit polling during convolution pipeline startup.

        Args:
            addr            : register address to poll
            expected        : value to wait for
            timeout_cycles  : maximum read attempts before raising TimeoutError
        """
        for _ in range(timeout_cycles):
            value = await self.reg_read(addr)
            if value == expected:
                logger.info("POLL OK:   addr=0x%04x  value=0x%08x", addr, expected)
                return
            await RisingEdge(self.dut.dla_csb_clk)
        raise TimeoutError(
            f"Poll timeout: addr=0x{addr:04x}, expected=0x{expected:08x}"
        )

    # ══════════════════════════════════════════════════════════════════
    #  Interrupt
    # ══════════════════════════════════════════════════════════════════

    async def wait_for_interrupt(self):
        """Block until dla_intr is asserted (inference complete)."""
        while self.dut.dla_intr.value != 1:
            await RisingEdge(self.dut.dla_core_clk)
        logger.info("NVDLA interrupt received!")

    # ══════════════════════════════════════════════════════════════════
    #  CRC Helpers
    # ══════════════════════════════════════════════════════════════════

    def calc_crc32(self, data_bytes: list) -> int:
        """Standard CRC-32 over a list of byte values."""
        return binascii.crc32(bytes(data_bytes))

    def calc_surface_crc(self, base_addr: int, length: int) -> int:
        """Read a memory surface and return its CRC-32."""
        data = self.read_memory(base_addr, length)
        return self.calc_crc32(data)
