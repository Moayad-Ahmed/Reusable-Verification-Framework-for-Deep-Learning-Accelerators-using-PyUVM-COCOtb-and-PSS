# NVDLA Verification Framework

This directory contains the verification environment for the NVIDIA Deep Learning Accelerator (NVDLA) pooling functionality using PyUVM and Cocotb.

> 📖 **New to this framework?** Check out [QUICK_START.md](QUICK_START.md) for a fast setup guide!

## Quick Start

### Prerequisites

1. **Set QUESTA_HOME environment variable:**
   ```powershell
   # Windows PowerShell
   $env:QUESTA_HOME = "C:\questasim64_2024.1\win64"
   ```

2. **Install required Python packages:**
   ```bash
   pip install cocotb pyuvm cocotbext-axi
   ```

### Running Tests

You can run the verification tests using either method:

#### Method 1: Using Python Runner (Recommended)

```bash
python test_runner.py
```

**Features:**
- Automatic cleanup of previous run artifacts
- Uses `rtl_sources.f` directly (same as Makefile)
- Handles all RTL files including `.vlib` library modules
- Sets up include directories automatically
- Generates waveforms for debugging
- More portable across different environments

#### Method 2: Using Makefile

```bash
make
```

**Additional Makefile targets:**
```bash
make cleanall              # Clean all build artifacts
make report_code_coverage  # Generate code coverage report
```

## Project Structure

```
NVDLA/
├── pyuvm_components/      # Verification components
│   ├── test.py           # Test definitions
│   ├── env.py            # UVM environment
│   ├── agent.py          # UVM agent
│   ├── driver.py         # Driver component
│   ├── monitor.py        # Monitor component
│   ├── scoreboard.py     # Scoreboard with golden model
│   ├── sequences.py      # Test sequences
│   └── seq_item.py       # Sequence item
│
├── strategy/              # Layer strategy and PSS
│   ├── pooling_strategy.py
│   ├── regs_configs.py
│   └── Layer_Factory.py
│
├── utils/                 # Utility functions
│   └── nvdla_utils.py    # NVDLA BFM and helpers
│
├── rtl/                   # RTL design files
│   ├── NVDLA_top.sv      # Top-level module
│   ├── dram.sv           # DRAM model
│   └── vmod/             # NVDLA RTL modules
│
├── yaml/                  # Configuration files
│   └── nvdla_pooling_config.yaml
│
├── input_files/           # Test input data files
│
├── rtl_sources.f          # RTL source file list
├── Makefile              # Make-based build
└── test_runner.py        # Python-based runner
```

## Test Configuration

The test configuration is specified in YAML files under the `yaml/` directory:

- `nvdla_pooling_config.yaml` - Pooling layer configuration

Input data files are located in the `input_files/` directory.

## Verification Components

### Test (`test.py`)
- **PdpBasicTest**: Runs the pooling test through NVDLA PDP (Pooling Data Processor)

### Sequences (`sequences.py`)
- **PdpTestSequence**: Configures and runs pooling operations

### Scoreboard (`scoreboard.py`)
- Compares DUT output with golden model results
- Uses Python-based golden model for reference

## Debugging

### Viewing Waveforms

After running tests, waveform files are generated in the `sim_build/` directory:
- For QuestaSim: Open `sim_build/*.wlf` with vsim

### Coverage Reports

Generate code coverage report:
```bash
make report_code_coverage
```

This creates `code_coverage_report.txt` with detailed coverage information.

## Common Issues

### ModuleNotFoundError: No module named 'cocotbext'

**Solution:**
```bash
pip install cocotbext-axi
```

### QUESTA_HOME not set

**Solution:**
```powershell
$env:QUESTA_HOME = "C:\questasim64_2024.1\win64"
```

### Permission errors during cleanup

The test runner automatically handles cleanup, but if you encounter issues:
```bash
make cleanall
```

## Documentation

For more detailed information, see:
- `docs/architecture.md` - Framework architecture
- `docs/verification_flow.md` - Verification flow details
- `docs/Configurations/PDP_Configuration_Quick_Guide.txt` - PDP configuration guide
- `docs/Configurations/PDP_Registers_Usage_Guide.txt` - PDP register usage
