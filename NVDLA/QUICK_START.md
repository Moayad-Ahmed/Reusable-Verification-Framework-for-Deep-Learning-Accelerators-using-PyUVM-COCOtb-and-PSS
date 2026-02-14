# NVDLA Quick Start Guide

## ✅ Prerequisites

1. **Set QUESTA_HOME**:
   ```powershell
   $env:QUESTA_HOME = "C:\questasim64_2024.1\win64"
   ```

2. **Install Python packages**:
   ```bash
   pip install cocotb pyuvm cocotbext-axi
   ```

## 🚀 Running Tests

### Option 1: Python Runner (Recommended)

```bash
cd NVDLA
python test_runner.py
```

**What it does:**
- ✅ Automatically cleans previous build artifacts
- ✅ Parses all 485+ RTL source files from `rtl_sources.f`
- ✅ Handles `.vlib` library files correctly
- ✅ Sets up all include directories
- ✅ Configures PYTHONPATH automatically
- ✅ Compiles the design and runs tests
- ✅ Generates waveforms in `sim_build/vsim.wlf`
- ✅ Creates test results in `sim_build/results.xml`

**Expected output:**
```
Cleaning artifacts...
Cleanup completed successfully!
Parsing RTL sources from rtl_sources.f...
Found 485 RTL source files
PYTHONPATH set to: ...
Building NVDLA design...
[Compilation output with "Errors: 0"]
Running NVDLA tests...
[Test execution]
Test execution completed!
```

**Test results:**
- Results file: `sim_build/results.xml`
- Waveforms: `sim_build/vsim.wlf`
- Transcript: `sim_build/transcript`

### Option 2: Makefile

```bash
cd NVDLA
make
```

## 📊 Verifying Success

After running, check:

1. **No compilation errors:**
   ```
   Errors: 0, Warnings: 77
   ```

2. **Test passed:**
   ```xml
   <testcase name="PdpBasicTest" ... />
   <!-- No <failure> tag means PASSED -->
   ```

3. **Scoreboard results** (in transcript):
   ```
   20 correct cases, 0 failed cases
   ```

## 🔍 Viewing Waveforms

```bash
# In QuestaSim
vsim -view sim_build/vsim.wlf
```

## 🧹 Cleaning Up

The Python runner automatically cleans before each run. To manually clean:

```bash
# Using Makefile
make cleanall

# Or manually delete
Remove-Item -Recurse -Force sim_build, __pycache__
```

## ⚡ Performance

- **Compilation**: ~5-10 seconds
- **Test execution**: ~30-45 seconds
- **Total time**: ~40-55 seconds
- **Simulation time**: 805.2 μs (805,200 ns)

## 🎯 What the Test Does

The `PdpBasicTest`:
1. Initializes NVDLA's PDP (Pooling Data Processor)
2. Configures pooling parameters from YAML config
3. Loads input data from `input_files/pdp_1x1x1_3x3_ave_int8_0_in.dat`
4. Writes configuration registers via CSB (Configuration Space Bus)
5. Enables the PDP and starts processing
6. Monitors the output via AXI bus
7. Compares results with Python golden model
8. Runs 20 test iterations
9. Reports PASS/FAIL for each iteration

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'cocotbext'`
**Solution:**
```bash
pip install cocotbext-axi
```

### Issue: `QUESTA_HOME is not set`
**Solution:**
```powershell
$env:QUESTA_HOME = "C:\questasim64_2024.1\win64"
```

### Issue: Compilation errors
**Solution:** Ensure all RTL files exist. The runner parses 485 files from `rtl_sources.f`.

## 📚 Additional Resources

- [README.md](README.md) - Full documentation
- [RUNNING_TESTS.md](RUNNING_TESTS.md) - Comparison of running methods
- [docs/](../docs/) - Architecture and verification flow details
- [docs/Configurations/](../docs/Configurations/) - PDP configuration guides

## 🎉 Success Indicators

You'll know the test passed when you see:
- ✅ `Errors: 0`
- ✅ `Check phase completed: 20 correct cases, 0 failed cases`
- ✅ `Test execution completed!`
- ✅ `results.xml` with no `<failure>` tags
