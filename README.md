# Reusable Verification Framework for Deep Learning Accelerators using PyUVM, COCOtb and PSS

This repository presents a reusable, scalable, and portable verification framework for Deep Neural Network (DNN) hardware accelerators.
The framework leverages Python-based verification (Pyuvm + Cocotb) and Portable Stimulus Standard (PSS) to enable layer-wise and end-to-end verification of DNN accelerators across different designs.

Traditional verification approaches rely on writing individual tests per layer and configuration, which does not scale well for modern DNN accelerators.
This project addresses that limitation by introducing generic, parameterizable verification components and scenario-based stimulus generation.

## 🎯 Project Objectives

- Develop a **reusable verification framework** for DNN accelerators

- Enable **generic layer-wise verification** (Conv, ReLU, Pooling, FC)

- Reduce verification effort across multiple accelerator configurations

- Move from single-layer tests to multi-layer realistic model flows

- Integrate **PSS-based scenario generation** for automation and portability

## 🧱 Key Technologies

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
