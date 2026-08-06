# A Hardware-Efficient Online Softmax Engine for FlashAttention using Base-2 Shift-LUT Exponentiation

> Hardware-efficient implementation of the Online Softmax recurrence for FlashAttention using a novel Base-2 Shift-LUT exponentiation engine.

---

## Overview

This repository contains the complete RTL, Python evaluation framework, FPGA implementation, and ASIC implementation flow accompanying our paper:

> **A Hardware-Efficient Online Softmax Engine for FlashAttention using Base-2 Shift-LUT Exponentiation**

Unlike conventional implementations that compute the exponential using iterative algorithms such as **CORDIC**, this work exploits the bounded exponent range of FlashAttention's online softmax formulation to replace iterative exponentiation with:

- Base-2 logarithmic decomposition
- Small ROM lookup
- Exact barrel shifting

resulting in significantly lower hardware cost while maintaining model accuracy.

The repository contains:

- RTL implementation
- FPGA implementation (Vivado)
- ASIC synthesis & P&R flow (Cadence Genus + Innovus)
- Python golden model
- End-to-end GPT-2 evaluation
- CORDIC baseline implementation
- Verification infrastructure

---

# Repository Structure

```
.
├── rtl/
│   ├── shift_lut_exp/
│   ├── fp32_mac/
│   ├── fp_div_synth/
│   ├── accumulator_update/
│   ├── softmax_engine_top/
│   └── bf16_delta/
│   └── bf16_compare/
│
├── python/
│   ├── softmax_kerenl_golden_reference/
│   ├── evaluation/

└── README.md
```

---

# Project Motivation

FlashAttention reformulates softmax into an online recurrence that computes

- running maximum
- running denominator
- running weighted output

for every incoming attention score.

The exponential evaluation therefore becomes part of the critical datapath.

Most existing hardware implementations rely on

- Hyperbolic CORDIC
- Polynomial approximation
- Piecewise approximation

which introduce iterative latency, larger area, and higher power.

This work asks:

> **Can the exponential be computed without iteration?**

The answer is **yes**.

Since the exponent entering FlashAttention is always bounded and non-positive,

$$
e^{-\delta} = 2^{-\delta \log_2 e}
$$

The exponent naturally separates into

$$
z = k + f
$$

allowing

$$
2^{-z} = 2^{-k} \times 2^{-f}
$$

where

- **Integer component** → Barrel Shift
- **Fractional component** → 32-entry ROM Lookup

No iterative refinement is required.

---

# Architecture

The complete online-softmax pipeline consists of four stages.

```
Input

↓

S0
Input Register

↓

S1
Comparator
+
BF16 Delta Generator

↓

S2
Shift-LUT Exponentiation

↓

S3
Accumulator Update

↓

FP32 Divider

↓

Softmax Output
```

The Shift-LUT exponentiation stage performs

1. Base-2 scaling
2. Saturation
3. BF16 → Q4.5 conversion
4. Integer/Fraction split
5. ROM lookup
6. Barrel shift recombination

---

# Key Features

- Fully pipelined Online Softmax engine
- BF16 input interface
- FP32 internal accumulation
- Exact Base-2 Shift-LUT exponentiation
- Vendor-independent RTL
- FPGA validated
- ASIC synthesized
- Python golden model
- End-to-end GPT-2 evaluation
- CORDIC baseline included

---

# Repository Contents

## RTL

Includes complete synthesizable SystemVerilog implementation.

Modules include

- bf16_compare
- bf16_delta
- shift_lut_exp
- accumulator_update
- fp32_mac
- fp_div_synth
- softmax_engine_top

---

## Python

Contains

- Golden reference implementation and evaluation which comprises:
  - Bit-exact BF16 emulation
  - Error analysis
  - KL divergence evaluation
  - GPT-2 evaluation
  - Perplexity experiments
  - ROM size ablation
  - Accuracy benchmarking

---

## FPGA Flow

Validated on

**Xilinx Spartan-7**

Tools:

- Vivado

Includes

- synthesis
- implementation
- timing reports
- utilization
- power reports

---

## ASIC Flow

Validated using

- UMC 65nm Standard Cell Library

EDA tools

- Cadence Genus
- Cadence Innovus
- Voltus

Includes

- synthesis scripts
- floorplanning
- CTS
- routing
- power reports
- timing reports

---

# Verification

The design was verified at multiple abstraction levels.

### RTL Verification

- Module-level testbenches
- Randomized stimulus
- ULP tolerance checking

### Functional Verification

Python golden model

↓

RTL comparison

↓

Waveform validation

### Algorithm-Level Validation

GPT-2

↓

WikiText-2

↓

Perplexity

↓

Top-1 Agreement

↓

KL Divergence

---

# Experimental Results

## ASIC (65nm UMC)

| Metric | Shift-LUT | Improvement |
|----------|------------|----------------|
| Total Area | 29,561 µm² | **27.9% smaller** |
| Exponential Unit | 671 µm² | **18.7× smaller** |
| Total Power | 4.24 mW | **24.8% lower** |
| Fmax | 154.5 MHz | Comparable to CORDIC |

---

## FPGA (Spartan-7)

| Metric | Shift-LUT | Improvement |
|----------|------------|----------------|
| LUTs | 2050 | 22.8% fewer |
| Exponential LUTs | 56 | 13.2× fewer |
| Dynamic Power | 69 mW | 10.3% lower |
| Frequency | 107.64 MHz | 6.4% higher |

---

# Numerical Accuracy

Compared against FP32 baseline

- Mean relative error ≈ 0.8%
- Maximum relative error ≈ 1.5%
- GPT-2 perplexity within 0.3%
- Top-1 token agreement unchanged
- Low average KL divergence

---

# Novel Contributions

This work demonstrates that

- FlashAttention's exponent range is fundamentally bounded.
- Iterative exponentiation is therefore unnecessary.
- Exact Base-2 decomposition enables a hardware-friendly implementation.
- ROM lookup + barrel shift replace iterative CORDIC without sacrificing model accuracy.
- The resulting design significantly improves area and power while preserving numerical behavior.

---

## Running the Python Evaluation

The Python evaluation consists of two files:

- `softmax_kernel_golden_reference.py` – Implements the bit-exact golden reference model of the online softmax engine.
- `evaluation.py` – Runs functional verification, accuracy evaluation, and generates the reported metrics.

### Directory Structure

Ensure both files are placed in the **same directory**:

```
python/
├── evaluation.py
└── softmax_kernel_golden_reference.py
```

### Running

From within the `python/` directory, execute:

```bash
python evaluation.py
```

or, depending on your Python installation,

```bash
python3 evaluation.py
```

The script automatically imports `softmax_kernel_golden_reference.py` and performs the complete evaluation workflow.

---

# License

This repository is released under the MIT License.

---



