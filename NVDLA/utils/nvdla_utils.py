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
      - CSB register write / read
      - Interrupt wait
      - CRC-32 calculation
    """

    def __init__(self):
        self.dut = cocotb.top
        self.output_config_queue = Queue(maxsize=0)
        self.iteration_done_queue = Queue(maxsize=0)
        self.axi_master = None

    # ---- Reset Control ----
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


    # ---- AXI master creation ----
    async def init_axi(self):
        """Create the AXI master after reset is established and signals are stable."""
        if self.axi_master is None:
            self.axi_master = AxiMaster(AxiBus.from_prefix(self.dut, "ext2dbb"), self.dut.dla_core_clk, 
                                        self.dut.dla_reset_rstn, reset_active_level=False)

    # --- DRAM Memory Access ----
    async def write_in_dram(self, input_data_path: list, base_addr: int):
        # Extract lines from the input file (one hex byte per line)
        with open(input_data_path, "r") as file:
            lines = [ln.strip() for ln in file if ln.strip()]

        # Convert hex lines to byte values
        byte_data = [int(ln, 16) for ln in lines]

        # Group every 8 bytes into a little-endian qword and write to DRAM
        num_qwords = len(byte_data) // 8
        for i in range(num_qwords):
            qword = 0
            for j in range(8):
                qword |= byte_data[i * 8 + j] << (j * 8)
            await self.axi_master.write_qword(base_addr + i * 8, qword)

        await RisingEdge(self.dut.dla_core_clk)

    async def read_from_dram(self, base_addr: int, num_pixels: int,
                              pixel_bytes: int = 8,
                              data_bytes_per_pixel: int = 1):
        """
        Read output data from DRAM, extracting all channel bytes per pixel.

        NVDLA stores each spatial pixel in ``pixel_bytes`` (atom-aligned).
        The first ``data_bytes_per_pixel`` bytes of each pixel contain
        actual channel data; the rest is padding.

        Args:
            base_addr:            DRAM start address of the output surface.
            num_pixels:           Number of spatial output pixels to read.
            pixel_bytes:          Total bytes per pixel in memory (atom-aligned,
                                  e.g. 8 for 1-ch INT8, 16 for 10-ch INT8).
            data_bytes_per_pixel: Number of real data bytes per pixel
                                  (= channels × bytes_per_element).

        Returns:
            list[int]: Flat list of data bytes across all pixels / channels,
                       in pixel-major, channel-minor order.
        """
        actual_output = []

        for i in range(num_pixels):
            pixel_addr = base_addr + i * pixel_bytes
            for ch in range(data_bytes_per_pixel):
                data = await self.axi_master.read_byte(pixel_addr + ch)
                actual_output.append(data)

        await RisingEdge(self.dut.dla_core_clk)

        return actual_output

    # ---- Configuration Registers Write ----
    async def reg_write(self, addr: int, data: int):
        """Non-posted CSB registers write"""
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

    # ---- Configuration Registers Read ----
    async def reg_read(self, addr: int):
        """CSB registers read"""
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

    # ----- Interrupt Handling -----
    async def wait_for_interrupt(self):
        """Wait until ``dla_intr`` is asserted."""
        while self.dut.dla_intr.value != 1:
            await RisingEdge(self.dut.dla_core_clk)
        logger.info("NVDLA interrupt received!")

    # ---- CRC Calculation ----
    def calc_crc32(self, data_bytes: list) -> int:
        """Standard CRC-32 over a list of byte values."""
        return binascii.crc32(bytes(data_bytes))

    def calc_surface_crc(self, base_addr: int, length: int) -> int:
        """Read a memory surface and return its CRC-32."""
        data = self.read_memory(base_addr, length)
        return self.calc_crc32(data)