# Reusable Verification Framework for Deep Learning Accelerators using PyUVM, Cocotb and PSS

This repository presents a project focused on the development of a reusable, scalable, and portable verification framework for Deep Neural Network (DNN) hardware accelerators. The primary goal is to significantly reduce the verification effort required across different DNN accelerator designs by introducing a generic and modular testbench architecture.

The framework utilizes Python-based verification methodologies, specifically Pyuvm and Cocotb, to build reusable UVM-like verification components tailored to fundamental DNN layers such as Convolution, Max Pooling, ReLU, and Fully Connected layers. These components are designed to be highly parameterizable, enabling efficient and adaptable layer-wise verification for a wide range of accelerator configurations, including variations in tensor sizes, data precision, and architectural parameters.

In addition, the project integrates the Portable Stimulus Standard (PSS) to enable automated scenario generation. Instead of relying on manually written, isolated test cases for individual layers, PSS is used to describe and generate complex, multi-layer verification scenarios that closely resemble real DNN workloads. This allows the verification flow to seamlessly generate realistic model execution paths, such as Convolution → ReLU → Pooling → Fully Connected, improving coverage and scalability.

Through this approach, the project aims to establish a modern verification framework that supports both layer-level and end-to-end verification of DNN accelerators, enhances reusability across designs, and aligns with industry-standard verification practices.

## 🎯 Project Objectives

- Develop a **reusable verification framework** for DNN accelerator

- Enable **generic layer-wise verification** (Conv, ReLU, Pooling, FC)

- Reduce verification effort across multiple accelerator configurations

- Move from single-layer tests to multi-layer realistic model flows

- Integrate **PSS-based scenario generation** for automation and portability

## ⚙ Key Technologies

- **Cocotb** – Python-based coroutine-driven testbench

- **Pyuvm** – UVM-like verification methodology in Python

- **PSS (Portable Stimulus Standard)** – High-level, portable scenario modeling

- **Python Golden Models** – Reference models using NumPy / PyTorch-style computation

## 🧩 Framework Architecture

```
PSS Scenarios
     │
     ▼
Stimulus Generation (Pyuvm Sequences)
     │
     ▼
Driver  ───► DUT (DNN Accelerator RTL)
     │               │
     ▼               ▼
Monitor          Output Signals
     │
     ▼
Scoreboard  ◄── Golden Model (Python)
     │
     ▼
Coverage & Reporting
```

## 🚀 Getting Started

> 💡 **Quick Start**: See [NVDLA/QUICK_START.md](NVDLA/QUICK_START.md) for a fast setup guide!

### Prerequisites

- Python 3.8 or higher
- QuestaSim/ModelSim
- Required Python packages:
  ```bash
  pip install -r requirements.txt
  ```

- Set `QUESTA_HOME` environment variable:
  ```bash
  # Windows PowerShell
  $env:QUESTA_HOME = "C:\questasim64_2024.1\win64"
  
  # Linux/Mac
  export QUESTA_HOME="/path/to/questasim/bin"
  ```

### Running Tests

This framework supports two verification targets:

#### 1. NVDLA Framework (Hardware Accelerator)

Located in the `NVDLA/` directory, this verifies the NVIDIA Deep Learning Accelerator (NVDLA) pooling functionality.

**Option A: Using Makefile**
```bash
cd NVDLA
make
```

**Option B: Using Python Runner**
```bash
cd NVDLA
python test_runner.py
```

#### 2. Standalone Layers (Generic DNN Layers)

Located in the `Standalone_Layers/` directory, this verifies standalone implementations of common DNN layers (Conv, Pool, FC, Activation).

**Option A: Using Makefile**
```bash
cd Standalone_Layers
make
```

**Option B: Using Python Runner**
```bash
cd Standalone_Layers
python test_runner.py
```

### Cleaning Build Artifacts

To clean all generated files and artifacts:

**Using Makefile:**
```bash
make cleanall
```

**Using Python Runner:**
The test runner automatically cleans artifacts before each run.

## 📁 Project Structure

```
├── NVDLA/                          # NVDLA accelerator verification
│   ├── pyuvm_components/           # PyUVM testbench components
│   ├── rtl/                        # RTL design files
│   ├── strategy/                   # Layer strategies and PSS
│   ├── utils/                      # Utility functions
│   ├── yaml/                       # Configuration files
│   ├── Makefile                    # Make-based runner
│   └── test_runner.py              # Python-based runner
│
├── Standalone_Layers/              # Standalone layer verification
│   ├── pyuvm_components/           # PyUVM testbench components
│   ├── rtl/                        # RTL design files
│   ├── strategy/                   # Layer strategies
│   ├── utils/                      # Utility functions
│   ├── yaml_files/                 # Test configurations
│   ├── Makefile                    # Make-based runner
│   └── test_runner.py              # Python-based runner
│
└── docs/                           # Documentation
    ├── architecture.md
    ├── pss_overview.md
    └── verification_flow.md
```
