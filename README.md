# Deep Learning Side-Channel Attack on MT-TMVP Polynomial Multiplication

This repository contains the artifact for the paper:

> **Beyond the Blind Spot: Deep Learning-Based Side-Channel Attack on MT-TMVP in Post-Quantum Cryptography**
>
> Submitted to CHES 2026 (anonymous review).

The artifact includes hardware designs, data collection scripts, leakage analysis tools, the deep learning attack implementation, countermeasure RTL, and pre-computed results. Everything needed to replicate the experiments in the paper is provided here.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Requirements](#requirements)
4. [Quick Start: Inspecting Pre-Computed Results](#quick-start-inspecting-pre-computed-results)
5. [Full Replication Guide](#full-replication-guide)
   - [Step 1: Bitstream Generation](#step-1-bitstream-generation)
   - [Step 2: Input Generation](#step-2-input-generation)
   - [Step 3: Trace Capture](#step-3-trace-capture)
   - [Step 4: Dataset Verification](#step-4-dataset-verification)
   - [Step 5: TVLA Leakage Assessment](#step-5-tvla-leakage-assessment)
   - [Step 6: DL-SCA Model Training and Evaluation](#step-6-dl-sca-model-training-and-evaluation)
   - [Step 7: Countermeasure Evaluation](#step-7-countermeasure-evaluation)
6. [Expected Results](#expected-results)
7. [Hardware Verification](#hardware-verification)
8. [License](#license)

---

## Overview

MT-TMVP (Modular Tiled Toeplitz Matrix-Vector Product) is a resource-efficient polynomial multiplier designed for lattice-based post-quantum cryptography on FPGAs. This work evaluates the side-channel security of a Verilog MT-TMVP implementation targeting NTRU (n=509, q=256, ternary secret key) on a Xilinx Artix-7 FPGA.

The attack pipeline has three stages:

1. **Leakage detection.** TVLA confirms exploitable information leakage in the power traces.
2. **Deep learning attack.** A residual CNN recovers individual ternary coefficients of the secret polynomial from single power traces.
3. **Countermeasure evaluation.** Three combined hardware countermeasures (arithmetic masking, random delay insertion, dummy computation rounds) are applied, and the attack is repeated to measure protection effectiveness.

---

## Repository Structure

```
.
├── README.md                 # This file
├── LICENSE                   # MIT license
├── requirements.txt          # Python dependencies
├── .gitignore
│
├── hardware/
│   ├── unprotected/          # Original MT-TMVP design
│   │   ├── rtl/
│   │   │   ├── core/         # MT-TMVP arithmetic modules (6 files)
│   │   │   │   ├── MatrixVectorMultiplier.v
│   │   │   │   ├── TMVP2.v
│   │   │   │   ├── TMVP2_main.v
│   │   │   │   ├── TMVP3_main.v
│   │   │   │   ├── TMVP_top.v
│   │   │   │   └── dual_port_ram.v
│   │   │   └── wrapper/      # CW305 board adapter
│   │   │       ├── top.v
│   │   │       ├── params.vh
│   │   │       └── cw305/    # USB interface, CDC, clock config
│   │   ├── constraints/
│   │   │   └── cw305_main.xdc
│   │   └── vivado/
│   │       └── tmvp_cw305_minimal.tcl
│   │
│   └── protected/            # Hardened MT-TMVP design
│       ├── rtl/
│       │   ├── core/         # Modified modules + lfsr_prng.v (7 files)
│       │   └── wrapper/      # Modified top.v (dummy rounds)
│       ├── constraints/
│       │   └── cw305_main.xdc
│       └── vivado/
│           └── tmvp_cw305_protected.tcl
│
├── software/
│   ├── generate_inputs.py        # Step 2: exhaustive input generation
│   ├── capture_traces.py         # Step 3: power trace acquisition
│   ├── verify_dataset.py         # Step 4: dataset integrity checks
│   ├── tvla.py                   # Step 5: TVLA leakage assessment
│   ├── train_attack.py           # Step 6: DL-SCA training and evaluation
│   ├── comprehensive_analysis.py # Post-training validation
│   └── evaluate_countermeasures.py # Step 7: protected design evaluation
│
├── verification/
│   ├── mt_tmvp.c             # C reference (bit-identical to RTL)
│   └── verify_mt_tmvp.py     # Python verification against C and RTL
│
└── results/                  # Pre-computed results (JSON + reports)
    ├── unprotected/
    │   └── attack_results.json
    ├── protected/
    │   └── attack_results.json
    └── tvla/
        └── tvla_report.md
```

**Note on datasets.** The full trace dataset (59,049 traces) and trained model checkpoints are hosted separately. See the [Dataset](#dataset) section below.

---

## Requirements

### Hardware (for full replication)

| Component | Specification |
|---|---|
| FPGA board | NewAE CW305 (Xilinx Artix-7 XC7A100T) |
| Capture board | NewAE CW1173 ChipWhisperer-Lite |
| Connection | 20-pin SMA cable between CW305 and CW1173 |
| Host PC | USB 2.0, 8 GB+ RAM |

### Software

| Tool | Version |
|---|---|
| Python | 3.9+ |
| PyTorch | 2.0+ (CUDA recommended) |
| Xilinx Vivado | 2020.2+ (bitstream generation only) |
| ChipWhisperer | 5.7+ (trace capture only) |

Install Python dependencies:

```bash
pip install -r requirements.txt
```

For GPU-accelerated training, install the appropriate PyTorch CUDA build from https://pytorch.org.

---

## Quick Start: Inspecting Pre-Computed Results

No hardware is needed to inspect the included results.

### Unprotected implementation

```bash
python -c "
import json
with open('results/unprotected/attack_results.json') as f:
    d = json.load(f)
print(f'Average test accuracy: {d[\"avg_test_accuracy\"]*100:.2f}%')
print(f'All-correct rate:      {d[\"all_correct_rate\"]*100:.2f}%')
for r in d['per_coefficient']:
    m = r['test_metrics']
    print(f'  Coeff {r[\"coeff_idx\"]}: acc={m[\"accuracy\"]*100:.2f}%, F1={m[\"f1_macro\"]:.4f}')
"
```

### Protected implementation

```bash
python -c "
import json
with open('results/protected/attack_results.json') as f:
    d = json.load(f)
print(f'Average test accuracy: {d[\"avg_test_accuracy\"]*100:.2f}%')
print(f'All-correct rate:      {d[\"all_correct_rate\"]*100:.2f}%')
for r in d['per_coefficient']:
    m = r['test_metrics']
    print(f'  Coeff {r[\"coeff_idx\"]}: acc={m[\"accuracy\"]*100:.2f}%, F1={m[\"f1_macro\"]:.4f}')
"
```

### TVLA report

Open `results/tvla/tvla_report.md` for the full leakage assessment, including per-test statistics, plots, and interpretation.

---

## Dataset

The full dataset is available at:

> https://zenodo.org/records/19561308

It contains:

| Item | Description
|---|---
| `dataset/traces/` | 59,049 `.npz` files (one per input combination) |
| `dataset/inputs/` | Pre-generated f/g polynomials, labels, expected outputs |
| `models/` | 10 trained `.pth` checkpoints (one per coefficient) |

Each `.npz` file stores:

| Key | Shape | Type | Description |
|---|---|---|---|
| `wave` | (1, 20000) | float64 | Power trace (20,000 samples at 7.37 MHz) |
| `labels` | (1, 10) | int64 | Ternary coefficient labels {-1, 0, 1} |
| `dut_io_ram_f_data` | (1, 509) | int64 | Input f polynomial |
| `dut_io_ram_g_data` | (1, 509) | int64 | Input g polynomial |
| `dut_io_computed_data` | (1, 509) | int64 | Hardware output |

---

## Full Replication Guide

The following steps reproduce the complete experiment from bitstream to final results. Steps 1 through 4 require the CW305 hardware. Steps 5 through 7 can run on the pre-collected dataset.

### Step 1: Bitstream Generation

Generate the FPGA bitstream using Xilinx Vivado.

**Unprotected design:**

```bash
cd hardware/unprotected/vivado
vivado -mode batch -source tmvp_cw305_minimal.tcl
```

This produces `top_cw305_a100.bit`. Program the CW305 with this file.

**Protected design:**

```bash
cd hardware/protected/vivado
vivado -mode batch -source tmvp_cw305_protected.tcl
```

This produces `top_cw305_protected.bit`. The protected design adds `lfsr_prng.v` and modifies `MatrixVectorMultiplier.v` (arithmetic masking), `TMVP_top.v` (random delay), and `top.v` (dummy rounds).

### Step 2: Input Generation

Generate the exhaustive set of 3^10 = 59,049 ternary input combinations.

```bash
cd software
python generate_inputs.py --output-dir ../dataset --seed 42
```

This creates `dataset/inputs/` with:

- `all_f_combinations.npy` -- (59049, 509) all f polynomials
- `all_g_data.npy` -- (59049, 509) replicated fixed g polynomial
- `labels.npy` -- (59049, 10) ternary labels per coefficient
- `expected_outputs.npy` -- (59049, 509) software-computed reference outputs
- `config.json` -- generation parameters

The random seed (42) ensures deterministic reproduction.

### Step 3: Trace Capture

Capture power traces from the CW305 board.

```bash
python capture_traces.py \
    --input-dir ../dataset \
    --output-dir ../dataset \
    --num-samples 20000 \
    --gain 20.0 \
    --clock 7.37e6
```

Each trace is saved as `dataset/traces/traces_<index>.npz`. The full capture of 59,049 traces takes approximately 8 hours.

To resume after interruption:

```bash
python capture_traces.py --input-dir ../dataset --output-dir ../dataset --resume-from 10000
```

### Step 4: Dataset Verification

Verify the captured dataset for completeness and correctness.

```bash
python verify_dataset.py --dataset ../dataset
```

This checks that:
- All 59,049 trace files exist and are loadable.
- The g polynomial is identical across all traces.
- The f polynomials match the generated combinations.
- Labels correspond to the correct ternary values.

### Step 5: TVLA Leakage Assessment

Run the Test Vector Leakage Assessment to confirm exploitable leakage.

```bash
python tvla.py --dataset ../dataset --output ../results/tvla --traces 1000
```

This performs Welch's t-test (threshold |t| > 4.5) for each of the 10 target coefficients, comparing traces grouped by coefficient value. It produces per-coefficient plots and a summary report.

To analyze a single coefficient:

```bash
python tvla.py --dataset ../dataset --coefficient 0 --traces 2000
```

### Step 6: DL-SCA Model Training and Evaluation

Train the deep residual CNN attack models.

```bash
python train_attack.py \
    --dataset ../dataset \
    --output ../results/unprotected \
    --epochs 100 \
    --batch-size 256 \
    --patience 15
```

This trains one `DeepResidualCNN` model per coefficient (10 total) with a 70/15/15 train/validation/test split (seed=42). Each model classifies traces into three classes corresponding to ternary values {-1, 0, 1}.

**Model architecture:**

| Layer | Output shape | Details |
|---|---|---|
| Input | (B, 1, 20000) | Raw normalized trace |
| Stem | (B, 64, 5000) | Conv1d(1, 64, k=15), BN, ReLU, MaxPool(4) |
| ResBlock 1 + Pool | (B, 64, 1250) | 2x Conv1d(64, 64, k=7) + skip, MaxPool(4) |
| ResBlock 2 + Pool | (B, 64, 312) | Same, MaxPool(4) |
| ResBlock 3 + Pool | (B, 64, 78) | Same, MaxPool(4) |
| ResBlock 4 + Pool | (B, 64, 39) | Same, MaxPool(2) |
| Head | (B, 3) | Conv1d(64, 128, k=5), BN, ReLU, AdaptiveAvgPool(1), Dropout(0.3), Linear(128, 3) |

**Training configuration:**

| Parameter | Value |
|---|---|
| Optimizer | AdamW (lr=0.001, weight_decay=0.01) |
| Scheduler | OneCycleLR (max_lr=0.01, pct_start=0.1) |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Precision | Mixed (FP16 via torch.cuda.amp) |
| Augmentation | Gaussian noise (std=0.02) |
| Early stopping | Patience=15, monitoring validation accuracy |

To train specific coefficients only:

```bash
python train_attack.py --dataset ../dataset --output ../results/unprotected --coefficients 0 1 2
```

After training, run the comprehensive analysis to validate results:

```bash
python comprehensive_analysis.py --dataset ../dataset --results ../results/unprotected/attack_results.json
```

### Step 7: Countermeasure Evaluation

Evaluate the attack on traces collected from the protected design.

First, collect traces from the protected bitstream following Steps 2 through 4 with the protected FPGA configuration. Then run:

```bash
python evaluate_countermeasures.py \
    --protected-dataset ../dataset_protected \
    --original-dataset ../dataset \
    --models-dir ../results/unprotected/models \
    --original-results ../results/unprotected/attack_results.json \
    --output-dir ../results/protected
```

This loads the models trained on unprotected traces and evaluates them on protected traces, producing a comparison report.

---

## Expected Results

### Unprotected Implementation

| Coefficient | Test Accuracy (%) | F1 Score |
|---|---|---|
| f[0] | 100.00 | 1.0000 |
| f[1] | 99.99 | 0.9999 |
| f[2] | 100.00 | 1.0000 |
| f[3] | 99.99 | 0.9999 |
| f[4] | 100.00 | 1.0000 |
| f[5] | 100.00 | 1.0000 |
| f[6] | 99.98 | 0.9998 |
| f[7] | 100.00 | 1.0000 |
| f[8] | 99.95 | 0.9995 |
| f[9] | 100.00 | 1.0000 |
| **Average** | **99.99** | **0.9999** |

**All-Correct Rate (ACR): 99.91%** (8,850 out of 8,858 test traces with all 10 coefficients correct)

### Protected Implementation

| Coefficient | Test Accuracy (%) | F1 Score |
|---|---|---|
| f[0] | 98.65 | 0.9864 |
| f[1] | 97.64 | 0.9764 |
| f[2] | 92.55 | 0.9256 |
| f[3] | 95.09 | 0.9509 |
| f[4] | 94.72 | 0.9470 |
| f[5] | 93.99 | 0.9397 |
| f[6] | 91.87 | 0.9185 |
| f[7] | 93.70 | 0.9369 |
| f[8] | 90.79 | 0.9077 |
| f[9] | 97.64 | 0.9764 |
| **Average** | **94.66** | **0.9466** |

**All-Correct Rate (ACR): 57.94%** (reduction of 42 percentage points)

### TVLA Summary

| Test | Leakage Points | Max |t| | Status |
|---|---|---|---|
| Random vs. Random (sanity) | 0 | 3.94 | PASS |
| Fixed vs. Random | 2,818 | 16.53 | LEAKAGE |
| f-value (f[0]) | 1,543 | 17.81 | LEAKAGE |
| Output byte 0 | 1,659 | 13.16 | LEAKAGE |
| Multi-byte (10 bytes) | 8,803 | 15.77 | LEAKAGE |
| Hamming weight (g[0]) | 4,037 | 24.44 | LEAKAGE |

Threshold: |t| > 4.5 (99.9999% confidence).

### Countermeasure Details

The protected design applies three countermeasures simultaneously:

| Countermeasure | Target | Mechanism |
|---|---|---|
| Arithmetic masking | `MatrixVectorMultiplier.v` | LFSR-generated random mask added before DSP multiply, removed after accumulation |
| Random delay | `TMVP_top.v` | 0 to 63 random wait cycles inserted between input loading and computation |
| Dummy rounds | `top.v` | Full TMVP computation with random data executed before real computation |

**Resource overhead:** +16 DSP48E1 slices, approximately 2x latency.

---

## Hardware Verification

The C reference implementation in `verification/mt_tmvp.c` produces output that is bit-identical to the Verilog RTL. To verify:

```bash
cd verification
gcc -o mt_tmvp mt_tmvp.c -lm
./mt_tmvp
python verify_mt_tmvp.py
```

This confirms functional correctness of the polynomial multiplication independent of the side-channel measurement setup.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
