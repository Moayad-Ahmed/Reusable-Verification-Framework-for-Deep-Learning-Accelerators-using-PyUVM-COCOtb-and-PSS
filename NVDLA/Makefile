# defaults
COCOTB_REDUCED_LOG_FMT = True
SIM ?= questa-compat
TOPLEVEL_LANG ?= verilog

# Add questa installation path to PATH
ifeq ($(SIM),questa-compat)
ifndef QUESTA_HOME
$(error QUESTA_HOME is not set. Please set it to your Questa installation path)
endif

export PATH := $(QUESTA_HOME):$(PATH)
endif

# Add Python source directories to PYTHONPATH (use ; separator for Windows)
export PYTHONPATH := $(PWD)/pyuvm_components;$(PYTHONPATH)

# Adding Verilog sources
VERILOG_INCLUDE_DIRS += $(PWD)/rtl
VERILOG_INCLUDE_DIRS += $(PWD)/rtl/vmod/include
VERILOG_INCLUDE_DIRS += $(PWD)/rtl/vmod/vlibs
VERILOG_INCLUDE_DIRS += $(PWD)/rtl/vmod/rams/model
VERILOG_INCLUDE_DIRS += $(PWD)/rtl/vmod/rams/synth
VERILOG_INCLUDE_DIRS += $(PWD)/rtl/vmod/nvdla
VERILOG_INCLUDE_DIRS += $(PWD)/rtl/spec/odif
VERILOG_INCLUDE_DIRS += $(PWD)/rtl/spec/manual
VERILOG_INCLUDE_DIRS += $(PWD)/rtl/vmod/defs

COMPILE_ARGS += -f $(PWD)/rtl_sources.f

VERILOG_SOURCES += $(PWD)/rtl/dram.sv
VERILOG_SOURCES += $(PWD)/rtl/NVDLA_top.sv

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
