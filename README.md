# Reusable Verification Framework for Deep Learning Accelerators using PyUVM, COCOtb and PSS

This repository presents a project focused on the development of a reusable, scalable, and portable verification framework for Deep Neural Network (DNN) hardware accelerators. The primary goal is to significantly reduce the verification effort required across different DNN accelerator designs by introducing a generic and modular testbench architecture.

The framework utilizes Python-based verification methodologies, specifically Pyuvm and Cocotb, to build reusable UVM-like verification components tailored to fundamental DNN layers such as Convolution, Max Pooling, ReLU, and Fully Connected layers. These components are designed to be highly parameterizable, enabling efficient and adaptable layer-wise verification for a wide range of accelerator configurations, including variations in tensor sizes, data precision, and architectural parameters.

In addition, the project integrates the Portable Stimulus Standard (PSS) to enable automated scenario generation. Instead of relying on manually written, isolated test cases for individual layers, PSS is used to describe and generate complex, multi-layer verification scenarios that closely resemble real DNN workloads. This allows the verification flow to seamlessly generate realistic model execution paths, such as Convolution → ReLU → Pooling → Fully Connected, improving coverage and scalability.

Through this approach, the project aims to establish a modern verification framework that supports both layer-level and end-to-end verification of DNN accelerators, enhances reusability across designs, and aligns with industry-standard verification practices.

## 🎯 Project Objectives

- Develop a **reusable verification framework** for DNN accelerators

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
