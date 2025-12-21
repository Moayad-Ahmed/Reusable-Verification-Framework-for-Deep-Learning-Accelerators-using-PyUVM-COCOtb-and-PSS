# defaults
COCOTB_REDUCED_LOG_FMT = True
SIM ?= questa-compat
TOPLEVEL_LANG ?= verilog

# Add questa installation path to PATH
export PATH := C:\questasim64_10.6c\win64:$(PATH)

# Enable code coverage
VLOG_ARGS += +cover -covercells
VSIM_ARGS += -classdebug -cover -coverage
VSIM_ARGS += -do setup_coverage.do

# Adding Verilog sources
VERILOG_SOURCES += $(PWD)/Design_files/pooling_layer_generic.v

# TOPLEVEL is the name of the toplevel module in your Verilog file
TOPLEVEL = pooling_layer_generic

# MODULE is the basename of the Python test file
MODULE = test

# set the HDL time unit and precision
COCOTB_HDL_TIMEUNIT = 1ns
COCOTB_HDL_TIMEPRECISION = 1ns

# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim

report_code_coverage:
	vcover report pooling_layer_db.ucdb -details -all -codeAll -output code_coverage_report.txt

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
