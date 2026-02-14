# Running the NVDLA Framework - Method Comparison

This document compares the two methods available for running the NVDLA verification framework.

## Methods Overview

| Feature | Makefile | Python Runner |
|---------|----------|---------------|
| **Command** | `make` | `python test_runner.py` |
| **Setup Complexity** | Requires Make installation | Pure Python (no Make needed) |
| **Portability** | Platform-dependent (Make syntax varies) | Cross-platform (pure Python) |
| **Automatic Cleanup** | Manual (`make cleanall`) | Automatic before each run |
| **Source Management** | Uses `-f rtl_sources.f` | Uses `-f rtl_sources.f` |
| **Include Dirs** | Manual specification | Automatic setup |
| **PYTHONPATH Setup** | Manual export | Automatic configuration |
| **Verbosity** | Less verbose | More verbose with progress messages |
| **Customization** | Edit Makefile | Edit test_runner.py |
| **IDE Integration** | Limited | Better (can debug Python directly) |

## Detailed Comparison

### 1. Using Makefile

**Pros:**
- ✅ Traditional approach, familiar to hardware engineers
- ✅ Integration with Cocotb's built-in Make infrastructure
- ✅ Standard in hardware verification projects
- ✅ Additional targets (coverage, clean, etc.)

**Cons:**
- ❌ Requires Make to be installed
- ❌ Platform-specific syntax differences
- ❌ Less transparent - harder to see what's happening
- ❌ Manual cleanup required

**When to use:**
- You're familiar with Makefiles
- Your team prefers traditional hardware flows
- You need to integrate with other Make-based tools
- You want to use make-specific features

**Example:**
```bash
cd NVDLA
make                           # Run tests
make cleanall                  # Clean artifacts
make report_code_coverage      # Generate coverage report
```

### 2. Using Python Runner

**Pros:**
- ✅ Pure Python - no additional tools needed
- ✅ Cross-platform compatible
- ✅ Automatic cleanup before each run
- ✅ More verbose progress messages
- ✅ Easier to customize and extend
- ✅ Better IDE integration for debugging
- ✅ Can be imported and used programmatically

**Cons:**
- ❌ Less standard in hardware verification
- ❌ Requires understanding Python code structure

**When to use:**
- You prefer Python-based workflows
- You want automatic cleanup
- You need better cross-platform compatibility
- You want to customize the flow in Python
- You're integrating with Python-based CI/CD systems

**Example:**
```bash
cd NVDLA
python test_runner.py
```

## Recommended Workflow

### For Development

Use **Python Runner** during active development:
```bash
python test_runner.py
```

**Why?**
- Automatic cleanup prevents stale artifacts
- More verbose output helps with debugging
- Faster iteration (no need to remember `make cleanall`)

### For CI/CD Integration

**Python Runner** is recommended:
```python
# In your CI script (e.g., GitHub Actions, Jenkins)
import sys
sys.path.append('NVDLA')
from test_runner import test_nvdla_runner

test_nvdla_runner()
```

**Alternatively, use Makefile if your CI is Make-based:**
```bash
make cleanall && make
```

### For Team/Production

Choose based on team preference:
- **Hardware teams** → Makefile (familiar territory)
- **Software/Python teams** → Python Runner (more natural)
- **Mixed teams** → Support both (they coexist without conflict)

## Environment Setup

Both methods require:

1. **QUESTA_HOME environment variable:**
   ```powershell
   # Windows
   $env:QUESTA_HOME = "C:\questasim64_2024.1\win64"
   ```

2. **Python packages:**
   ```bash
   pip install cocotb pyuvm cocotbext-axi
   ```

## File Structure

Both methods use the same underlying structure:

```
NVDLA/
├── test_runner.py          # Python runner (NEW)
├── Makefile                # Make-based runner (EXISTING)
├── rtl_sources.f           # RTL file list (shared)
├── pyuvm_components/       # Test components (shared)
├── rtl/                    # RTL files (shared)
├── strategy/               # Strategies (shared)
└── utils/                  # Utils (shared)
```

## Troubleshooting

### Python Runner Issues

**Problem:** `ModuleNotFoundError: No module named 'cocotb_tools'`

**Solution:**
```bash
pip install cocotb
```

**Problem:** `FileNotFoundError: rtl_sources.f not found`

**Solution:** Ensure you're running from the NVDLA directory:
```bash
cd NVDLA
python test_runner.py
```

### Makefile Issues

**Problem:** `make: command not found`

**Solution:** Install Make or use Python runner instead

**Problem:** Paths with spaces causing issues

**Solution:** Use Python runner which handles Windows paths better

## Conclusion

Both methods accomplish the same goal - running the NVDLA verification framework. Choose based on:

- **Team preference** and existing workflows
- **Platform requirements** (Python runner is more portable)
- **Integration needs** (Python runner better for Python CI/CD)
- **Familiarity** (Makefile for hardware engineers, Python for software engineers)

**Recommendation:** Try both and use what works best for your workflow. They coexist peacefully and produce identical results.
