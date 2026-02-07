# defaults
COCOTB_REDUCED_LOG_FMT = True
SIM ?= questa-compat
TOPLEVEL_LANG ?= verilog

# Add questa installation path to PATH
export PATH := G:\Programs\Questasim\win64:$(PATH)

# Design files
VERILOG_SOURCES += $(PWD)/Design_Files/convolution_layer_generic.v

# Top-level module
TOPLEVEL = convolution_layer_generic

# Python test module
MODULE = test

# set the HDL time unit and precision
COCOTB_HDL_TIMEUNIT = 1ns
COCOTB_HDL_TIMEPRECISION = 1ns

# Include cocotb makefiles
include $(shell cocotb-config --makefiles)/Makefile.sim

ifdef COVERAGE
    COMPILE_ARGS += --coverage
    EXTRA_ARGS += --coverage
endif

# Clean target
cleanall:
	@rm -rf __pycache__
	@rm -rf sim_build
	@rm -rf modelsim.ini
	@rm -rf transcript
	@rm -rf *.txt
	@rm -rf *.ucdb
	@rm -rf *.xml
	@rm -rf *.yml
	@rm -rf *.vstf
	@rm -rf *.vcd
	@echo "Cleanup completed successfully!"
