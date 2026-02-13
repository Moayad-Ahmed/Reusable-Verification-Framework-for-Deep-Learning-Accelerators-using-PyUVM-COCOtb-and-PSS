import logging
import binascii

import cocotb
from cocotb.queue import Queue
from cocotb.triggers import RisingEdge, Timer

from pyuvm import utility_classes

logger = logging.getLogger(__name__)


class NvdlaBFM(metaclass=utility_classes.Singleton):
    """
    NVDLA Bus Functional Model

    Provides:
      - CSB register write / read
      - DRAM memory load / read
      - Reset control
      - Interrupt wait
      - CRC-32 calculation
    """

    def __init__(self):
        self.dut = cocotb.top
        self.output_config_queue = Queue(maxsize=0)

    
    async def reset(self):
        """Reset NVDLA to initialize all signals to known values"""
        dut = self.dut

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

        # Hold reset for 10000 ns
        await Timer(10000, unit="ns")

        dut.dla_reset_rstn.value = 1
        dut.direct_reset_.value = 1
        logger.info("Reset released!")

        # Wait some time to observe stable state
        await Timer(5000, unit="ns")

    # ---- CSB Write ----
    async def csb_write(self, addr: int, data: int):
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

    # ---- CSB Read ----
    async def csb_read(self, addr: int):
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

    # ---- DRAM Memory Access ----
    def load_memory_from_file(self, filepath: str, base_addr: int, count: int):
        """Load a hex-data (.dat) file into the DRAM model (like $readmemh).

        The file is expected to contain one hex byte value per line.
        """
        with open(filepath, "r") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]

        data = [int(ln, 16) & 0xFF for ln in lines[:count]]
        self.load_memory(base_addr, data)

    def load_memory(self, base_addr: int, data: list):
        """Write a list of byte-values into the DRAM memory array."""
        for i, byte_val in enumerate(data):
            self.dut.dram_dut.memory[base_addr + i].value = byte_val
        logger.info("Loaded %d bytes to DRAM @ 0x%08x", len(data), base_addr)

    def read_memory(self, base_addr: int, length: int) -> list:
        """Read bytes from the DRAM memory array."""
        data = []
        for i in range(length):
            val = int(self.dut.dram_dut.memory[base_addr + i].value)
            data.append(val)
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