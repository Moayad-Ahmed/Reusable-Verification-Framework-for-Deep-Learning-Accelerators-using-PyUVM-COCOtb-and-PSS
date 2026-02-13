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
export PYTHONPATH := $(PWD)/pyuvm_components;$(PWD)/strategy;$(PWD)/utils;$(PYTHONPATH)

# Adding Verilog sources
VERILOG_SOURCES += $(PWD)/rtl/pooling_layer_generic.v
VERILOG_SOURCES += $(PWD)/rtl/convolution_layer_generic.v
VERILOG_SOURCES += $(PWD)/rtl/fully_connected_layer.v
VERILOG_SOURCES += $(PWD)/rtl/Activation_layer_DUT.v
VERILOG_SOURCES += $(PWD)/rtl/CNN_top.v

# TOPLEVEL is the name of the toplevel module in your Verilog file
TOPLEVEL = CNN_top

# MODULE is the basename of the Python test file
MODULE = test

# set the HDL time unit and precision
COCOTB_HDL_TIMEUNIT = 1ns
COCOTB_HDL_TIMEPRECISION = 1ns

# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim

report_code_coverage:
	vcover report CNN_top.ucdb -details -all -codeAll -output code_coverage_report.txt

cleanall:
	@rm -rf __pycache__
	@rm -rf sim_build
	@rm -rf .pytest_cache
	@rm -rf modelsim.ini
	@rm -rf transcript
	@rm -rf *.ucdb
	@rm -rf *.xml
	@rm -rf *.yml
	@rm -rf *.vstf
	@rm -rf *.vcd
	@echo "Cleanup completed successfully!"
