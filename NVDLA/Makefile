# defaults
COCOTB_REDUCED_LOG_FMT = True
SIM ?= questa-compat
TOPLEVEL_LANG ?= verilog

# Convert MSYS2/Git-Bash paths (/d/...) to Windows paths (D:/...)
# Note: cocotb overrides OS to "Msys", so we cannot rely on ifeq ($(OS),Windows_NT)
WPWD := $(shell pwd -W 2>/dev/null || pwd)

# Add questa installation path to PATH
ifeq ($(SIM),questa-compat)
ifndef QUESTA_HOME
$(error QUESTA_HOME is not set. Please set it to your Questa installation path)
endif

export PATH := $(QUESTA_HOME):$(PATH)
endif

# Add Python source directories to PYTHONPATH (use ; separator for Windows)
export PYTHONPATH := $(WPWD)/pyuvm_components;$(PYTHONPATH)

# Adding Verilog sources
VERILOG_INCLUDE_DIRS += $(WPWD)/rtl
VERILOG_INCLUDE_DIRS += $(WPWD)/rtl/vmod/include
VERILOG_INCLUDE_DIRS += $(WPWD)/rtl/vmod/vlibs
VERILOG_INCLUDE_DIRS += $(WPWD)/rtl/vmod/rams/model
VERILOG_INCLUDE_DIRS += $(WPWD)/rtl/vmod/rams/synth
VERILOG_INCLUDE_DIRS += $(WPWD)/rtl/vmod/nvdla
VERILOG_INCLUDE_DIRS += $(WPWD)/rtl/spec/odif
VERILOG_INCLUDE_DIRS += $(WPWD)/rtl/spec/manual
VERILOG_INCLUDE_DIRS += $(WPWD)/rtl/vmod/defs

COMPILE_ARGS += -f $(WPWD)/rtl_sources.f
ifeq ($(SIM),questa-compat)
COMPILE_ARGS += +define+SYNTHESIS
endif

VERILOG_SOURCES += $(WPWD)/rtl/dram.sv
VERILOG_SOURCES += $(WPWD)/rtl/NVDLA_top.sv

# TOPLEVEL is the name of the toplevel module in your Verilog file
TOPLEVEL = NVDLA_top

# MODULE is the basename of the Python test file
MODULE = pyuvm_components.test

# set the HDL time unit and precision
COCOTB_HDL_TIMEUNIT = 1ns
COCOTB_HDL_TIMEPRECISION = 1ns

# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim

report_code_coverage:
	vcover report NVDLA_top.ucdb -details -all -codeAll -output code_coverage_report.txt

cleanall:
	@rm -rf __pycache__
	@rm -rf pyuvm_components/__pycache__
	@rm -rf strategy/__pycache__
	@rm -rf utils/__pycache__
	@rm -rf sim_build
	@rm -rf .pytest_cache
	@rm -rf modelsim.ini
	@rm -rf transcript
	@rm -rf cdma2cbuf_data_rtl.dat
	@rm -rf cdma2cbuf_weight_rtl.dat
	@rm -rf *.ucdb
	@rm -rf *.xml
	@rm -rf *.yml
	@rm -rf *.vstf
	@rm -rf *.vcd
	@echo "Cleanup completed successfully!"
