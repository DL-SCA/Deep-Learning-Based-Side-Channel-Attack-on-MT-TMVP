"""Evaluate countermeasure effectiveness against the DL-SCA attack.

Loads protected traces and the pre-trained models, runs inference, and
compares accuracy before and after countermeasures to quantify protection.
"""

import os
import sys
import json
import argparse
import time
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

# ============================================================================
# CONSTANTS — must match the training script exactly
# ============================================================================
SEED = 42
FULL_TRACE_LENGTH = 20000
NUM_CLASSES = 3
NUM_COEFFICIENTS = 10

# ============================================================================
# MODEL — exact copy from dl_sca_attack_v8_multigpu_resMod.py
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class DeepResidualCNN(nn.Module):
    def __init__(self, input_length=20000, num_classes=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(64), nn.MaxPool1d(4),
            ResidualBlock(64), nn.MaxPool1d(4),
            ResidualBlock(64), nn.MaxPool1d(4),
            ResidualBlock(64), nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.res_blocks(self.stem(x)))

# ============================================================================
# SPLIT — exact copy from training script
# ============================================================================

def create_split(num_samples, train_ratio=0.70, val_ratio=0.15, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(num_samples)
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    return indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]

# ============================================================================
# DATA LOADING
# ============================================================================

def load_traces_from_dir(traces_dir: Path, expected_trace_len: int = FULL_TRACE_LENGTH):
    """Load all traces from a directory of .npz files."""
    trace_files = sorted(traces_dir.glob("traces_*.npz"),
                         key=lambda x: int(x.stem.split('_')[1]))
    n = len(trace_files)
    if n == 0:
        raise FileNotFoundError(f"No trace files found in {traces_dir}")

    # Peek at first file
    first = np.load(trace_files[0])
    actual_len = first['wave'].flatten().shape[0]
    label_len = first['labels'].flatten().shape[0]

    traces = np.zeros((n, actual_len), dtype=np.float32)
    labels = np.zeros((n, label_len), dtype=np.int64)
    indices = np.zeros(n, dtype=np.int64)

    for i, tf in enumerate(tqdm(trace_files, desc="Loading traces", leave=False)):
        data = np.load(tf)
        traces[i] = data['wave'].flatten()
        labels[i] = data['labels'].flatten()
        # Extract the original combination index from filename
        indices[i] = int(tf.stem.split('_')[1])

    return traces, labels, indices


def load_model(model_path: Path, device):
    """Load a single pre-trained DeepResidualCNN model."""
    model = DeepResidualCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# ============================================================================
# INFERENCE
# ============================================================================

def run_inference(model, traces_tensor, device, batch_size=512):
    """Run inference and return predicted class indices."""
    model.eval()
    all_preds = []
    n = traces_tensor.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = traces_tensor[i:min(i+batch_size, n)].to(device, non_blocking=True)
            with autocast():
                outputs = model(batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            del batch, outputs
    return np.array(all_preds)

# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate countermeasure effectiveness against DL-SCA models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--protected-dataset", type=str, required=True,
                        help="Path to protected dataset directory (with traces/ subfolder)")
    parser.add_argument("--original-dataset", type=str, required=True,
                        help="Path to original 59k dataset directory (Dataset/sca_dataset_v4)")
    parser.add_argument("--models-dir", type=str, required=True,
                        help="Path to trained models directory (ResNet_results/models)")
    parser.add_argument("--original-results", type=str, required=True,
                        help="Path to attack_results.json from original training run")
    parser.add_argument("--output-dir", type=str, default="countermeasure_results",
                        help="Output directory for results and plots")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for inference (default: 512)")
    args = parser.parse_args()

    protected_path = Path(args.protected_dataset).resolve()
    original_path = Path(args.original_dataset).resolve()
    models_dir = Path(args.models_dir).resolve()
    original_results_path = Path(args.original_results).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("COUNTERMEASURE EFFECTIVENESS EVALUATION")
    print("=" * 70)
    print(f"Protected dataset : {protected_path}")
    print(f"Original dataset  : {original_path}")
    print(f"Models directory   : {models_dir}")
    print(f"Original results   : {original_results_path}")
    print(f"Output directory   : {output_dir}")
    print(f"Device             : {device}")
    print("=" * 70)

    # =========================================================================
    # Step 1: Load protected traces
    # =========================================================================
    print("\n[1/7] Loading protected (post-countermeasure) traces...")
    prot_traces_dir = protected_path / "traces"
    prot_traces, prot_labels, prot_indices = load_traces_from_dir(prot_traces_dir)
    n_protected = len(prot_traces)
    prot_trace_len = prot_traces.shape[1]
    print(f"  Loaded {n_protected} protected traces, trace length = {prot_trace_len}")
    print(f"  Labels shape: {prot_labels.shape}")
    print(f"  Sample indices range: {prot_indices.min()} .. {prot_indices.max()}")

    # =========================================================================
    # Step 2: Reconstruct train/val/test split from original 59k dataset
    # =========================================================================
    print("\n[2/7] Reconstructing original train/val/test split...")
    ORIGINAL_N = 59049  # 3^10
    orig_train_idx, orig_val_idx, orig_test_idx = create_split(ORIGINAL_N, 0.70, 0.15, SEED)
    print(f"  Original split: train={len(orig_train_idx)}, val={len(orig_val_idx)}, test={len(orig_test_idx)}")

    # Build lookup: original_index -> split_name
    split_map = {}
    for idx in orig_train_idx:
        split_map[idx] = 'train'
    for idx in orig_val_idx:
        split_map[idx] = 'val'
    for idx in orig_test_idx:
        split_map[idx] = 'test'

    # Classify each protected sample
    prot_split_labels = []
    for idx in prot_indices:
        prot_split_labels.append(split_map.get(int(idx), 'unknown'))
    prot_split_labels = np.array(prot_split_labels)

    n_train = np.sum(prot_split_labels == 'train')
    n_val = np.sum(prot_split_labels == 'val')
    n_test = np.sum(prot_split_labels == 'test')
    n_unknown = np.sum(prot_split_labels == 'unknown')

    print(f"  Protected samples breakdown:")
    print(f"    Originally in TRAIN set : {n_train}")
    print(f"    Originally in VAL set   : {n_val}")
    print(f"    Originally in TEST set  : {n_test}")
    if n_unknown > 0:
        print(f"    Unknown                 : {n_unknown}")

    train_mask = prot_split_labels == 'train'
    val_mask = prot_split_labels == 'val'
    test_mask = prot_split_labels == 'test'
    # Combined "seen" (train+val) vs "unseen" (test)
    seen_mask = train_mask | val_mask
    unseen_mask = test_mask

    # =========================================================================
    # Step 3: Load original training normalization stats
    # =========================================================================
    print("\n[3/7] Computing normalization statistics from original training data...")
    # We need the exact same normalization as training.
    # The training script computes mean/std from the TRAINING split of the 59k traces.
    # We must load enough original traces to compute that.
    orig_traces_dir = original_path / "traces"
    print("  Loading original traces for normalization (this may take a while)...")
    orig_traces, orig_labels_loaded, _ = load_traces_from_dir(orig_traces_dir)
    orig_trace_len = orig_traces.shape[1]
    print(f"  Loaded {len(orig_traces)} original traces, trace length = {orig_trace_len}")

    # Compute normalization from training split (exactly as training script does)
    global_mean = orig_traces[orig_train_idx].mean()
    global_std = orig_traces[orig_train_idx].std() + 1e-8
    print(f"  Normalization: mean={global_mean:.6f}, std={global_std:.6f}")

    # =========================================================================
    # Step 4: Normalize and prepare tensors
    # =========================================================================
    print("\n[4/7] Preparing tensors...")

    # Handle potential trace length mismatch
    if prot_trace_len != orig_trace_len:
        print(f"  WARNING: Protected trace length ({prot_trace_len}) != original ({orig_trace_len})")
        target_len = min(prot_trace_len, orig_trace_len)
        print(f"  Truncating/padding to {target_len}")
        if prot_trace_len > target_len:
            prot_traces = prot_traces[:, :target_len]
        elif prot_trace_len < target_len:
            pad = np.zeros((n_protected, target_len - prot_trace_len), dtype=np.float32)
            prot_traces = np.hstack([prot_traces, pad])

    # Normalize protected traces with the SAME statistics
    prot_traces_norm = (prot_traces - global_mean) / global_std
    prot_tensor = torch.FloatTensor(prot_traces_norm).unsqueeze(1)  # (N, 1, L)
    print(f"  Protected tensor shape: {prot_tensor.shape}")

    # Also prepare original test traces for "before" comparison
    orig_test_traces_norm = (orig_traces[orig_test_idx] - global_mean) / global_std
    orig_test_tensor = torch.FloatTensor(orig_test_traces_norm).unsqueeze(1)
    orig_test_labels = orig_labels_loaded[orig_test_idx]
    print(f"  Original test tensor shape: {orig_test_tensor.shape}")

    # Free large arrays
    del orig_traces, prot_traces, prot_traces_norm
    gc.collect()

    # =========================================================================
    # Step 5: Load original results for comparison
    # =========================================================================
    print("\n[5/7] Loading original (unprotected) attack results...")
    with open(original_results_path) as f:
        orig_results = json.load(f)

    orig_per_coeff = {}
    for entry in orig_results['per_coefficient']:
        c = entry['coeff_idx']
        orig_per_coeff[c] = entry['test_metrics']
    print(f"  Loaded results for {len(orig_per_coeff)} coefficients")
    print(f"  Original avg test accuracy: {orig_results['avg_test_accuracy']*100:.2f}%")

    # =========================================================================
    # Step 6: Run inference with all 10 models
    # =========================================================================
    print("\n[6/7] Running inference with pre-trained models...")

    results_per_coeff = {}
    all_prot_preds = {}
    all_orig_test_preds = {}

    for coeff_idx in range(NUM_COEFFICIENTS):
        model_path = models_dir / f"coeff_{coeff_idx}_best.pth"
        if not model_path.exists():
            print(f"  WARNING: Model not found: {model_path}, skipping")
            continue

        print(f"\n  --- Coefficient {coeff_idx} ---")
        model = load_model(model_path, device)

        # === Run on PROTECTED traces ===
        prot_preds = run_inference(model, prot_tensor, device, args.batch_size)
        all_prot_preds[coeff_idx] = prot_preds

        # True labels for this coefficient (class index = label + 1)
        true_labels = prot_labels[:, coeff_idx] + 1

        # Overall protected accuracy
        overall_acc = accuracy_score(true_labels, prot_preds)
        overall_f1 = f1_score(true_labels, prot_preds, average='macro', zero_division=0)

        # Per-split accuracy
        split_results = {}
        for split_name, mask in [('train', train_mask), ('val', val_mask),
                                  ('test', test_mask), ('seen', seen_mask),
                                  ('unseen', unseen_mask), ('all', np.ones(n_protected, dtype=bool))]:
            if mask.sum() == 0:
                continue
            s_true = true_labels[mask]
            s_pred = prot_preds[mask]
            split_results[split_name] = {
                'n_samples': int(mask.sum()),
                'accuracy': float(accuracy_score(s_true, s_pred)),
                'f1_macro': float(f1_score(s_true, s_pred, average='macro', zero_division=0)),
                'precision_macro': float(precision_score(s_true, s_pred, average='macro', zero_division=0)),
                'recall_macro': float(recall_score(s_true, s_pred, average='macro', zero_division=0)),
                'confusion_matrix': confusion_matrix(s_true, s_pred, labels=[0, 1, 2]).tolist(),
            }

        # === Run on ORIGINAL (unprotected) test traces for fair comparison ===
        orig_test_true = orig_test_labels[:, coeff_idx] + 1
        orig_preds = run_inference(model, orig_test_tensor, device, args.batch_size)
        all_orig_test_preds[coeff_idx] = orig_preds
        orig_test_acc_recomputed = float(accuracy_score(orig_test_true, orig_preds))
        orig_test_f1_recomputed = float(f1_score(orig_test_true, orig_preds, average='macro', zero_division=0))

        results_per_coeff[coeff_idx] = {
            'protected': split_results,
            'original_test_acc_from_json': float(orig_per_coeff.get(coeff_idx, {}).get('accuracy', 0)),
            'original_test_acc_recomputed': orig_test_acc_recomputed,
            'original_test_f1_recomputed': orig_test_f1_recomputed,
        }

        # Print summary for this coefficient
        orig_acc = results_per_coeff[coeff_idx]['original_test_acc_from_json'] * 100
        prot_all_acc = split_results['all']['accuracy'] * 100
        print(f"    Original (unprotected) test acc : {orig_acc:.2f}%")
        print(f"    Protected (all)         acc     : {prot_all_acc:.2f}%")
        if 'seen' in split_results:
            print(f"    Protected (seen/train+val) acc  : {split_results['seen']['accuracy']*100:.2f}%  (n={split_results['seen']['n_samples']})")
        if 'unseen' in split_results:
            print(f"    Protected (unseen/test)    acc  : {split_results['unseen']['accuracy']*100:.2f}%  (n={split_results['unseen']['n_samples']})")
        print(f"    Accuracy DROP               : {orig_acc - prot_all_acc:+.2f} pp")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    # =========================================================================
    # Step 7: Aggregate results and generate plots
    # =========================================================================
    print("\n[7/7] Generating summary and plots...")

    # --- Aggregate numbers ---
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_protected_samples': n_protected,
        'split_breakdown': {
            'train': int(n_train), 'val': int(n_val), 'test': int(n_test)
        },
        'per_coefficient': {},
        'comparison': {},
    }

    orig_accs = []
    prot_accs_all = []
    prot_accs_seen = []
    prot_accs_unseen = []
    orig_f1s = []
    prot_f1s_all = []

    for c in range(NUM_COEFFICIENTS):
        if c not in results_per_coeff:
            continue
        r = results_per_coeff[c]
        summary['per_coefficient'][c] = r

        orig_accs.append(r['original_test_acc_from_json'])
        prot_accs_all.append(r['protected']['all']['accuracy'])
        if 'seen' in r['protected']:
            prot_accs_seen.append(r['protected']['seen']['accuracy'])
        if 'unseen' in r['protected']:
            prot_accs_unseen.append(r['protected']['unseen']['accuracy'])
        orig_f1s.append(r.get('original_test_f1_recomputed', 0))
        prot_f1s_all.append(r['protected']['all']['f1_macro'])

    avg_orig = np.mean(orig_accs) * 100 if orig_accs else 0
    avg_prot_all = np.mean(prot_accs_all) * 100 if prot_accs_all else 0
    avg_prot_seen = np.mean(prot_accs_seen) * 100 if prot_accs_seen else 0
    avg_prot_unseen = np.mean(prot_accs_unseen) * 100 if prot_accs_unseen else 0
    random_baseline = 100.0 / NUM_CLASSES  # 33.33%

    summary['comparison'] = {
        'avg_original_test_accuracy': float(avg_orig),
        'avg_protected_all_accuracy': float(avg_prot_all),
        'avg_protected_seen_accuracy': float(avg_prot_seen),
        'avg_protected_unseen_accuracy': float(avg_prot_unseen),
        'accuracy_drop_pp': float(avg_orig - avg_prot_all),
        'random_baseline': float(random_baseline),
        'countermeasure_effective': bool(avg_prot_all < (random_baseline + 5)),
    }

    # --- Print final summary ---
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\n{'Coeff':<8} {'Orig Acc':<12} {'Prot (all)':<14} {'Prot (seen)':<14} {'Prot (unseen)':<14} {'Drop':<10}")
    print("-" * 72)
    for c in range(NUM_COEFFICIENTS):
        if c not in results_per_coeff:
            continue
        r = results_per_coeff[c]
        o = r['original_test_acc_from_json'] * 100
        pa = r['protected']['all']['accuracy'] * 100
        ps = r['protected'].get('seen', {}).get('accuracy', 0) * 100
        pu = r['protected'].get('unseen', {}).get('accuracy', 0) * 100
        drop = o - pa
        print(f"  C{c:<5} {o:>7.2f}%     {pa:>7.2f}%       {ps:>7.2f}%       {pu:>7.2f}%       {drop:>+7.2f}pp")
    print("-" * 72)
    print(f"  {'Avg':<6} {avg_orig:>7.2f}%     {avg_prot_all:>7.2f}%       {avg_prot_seen:>7.2f}%       {avg_prot_unseen:>7.2f}%       {avg_orig-avg_prot_all:>+7.2f}pp")
    print(f"\n  Random baseline (guessing): {random_baseline:.2f}%")

    if summary['comparison']['countermeasure_effective']:
        print("\n  >>> COUNTERMEASURES ARE EFFECTIVE <<<")
        print(f"  Protected accuracy ({avg_prot_all:.2f}%) is near random chance ({random_baseline:.2f}%).")
    else:
        print(f"\n  >>> COUNTERMEASURES PARTIALLY EFFECTIVE <<<")
        print(f"  Protected accuracy ({avg_prot_all:.2f}%) is still above random+5% ({random_baseline+5:.2f}%).")
        print(f"  The models can still extract some information from the protected traces.")

    # --- Save JSON results ---
    results_file = output_dir / "countermeasure_results.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_file}")

    # =========================================================================
    # PLOTS
    # =========================================================================

    coeffs = list(range(NUM_COEFFICIENTS))

    # --- Plot 1: Before vs After bar chart ---
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(NUM_COEFFICIENTS)
    width = 0.35
    bars_orig = ax.bar(x - width/2, [orig_accs[c]*100 for c in range(len(orig_accs))],
                       width, label='Unprotected (original)', color='#e74c3c', edgecolor='black', alpha=0.85)
    bars_prot = ax.bar(x + width/2, [prot_accs_all[c]*100 for c in range(len(prot_accs_all))],
                       width, label='Protected (countermeasures)', color='#2ecc71', edgecolor='black', alpha=0.85)
    ax.axhline(random_baseline, color='gray', linestyle='--', linewidth=2,
               label=f'Random baseline ({random_baseline:.1f}%)')
    ax.set_xlabel('Coefficient Index', fontsize=13)
    ax.set_ylabel('Test Accuracy (%)', fontsize=13)
    ax.set_title('DL-SCA Attack Accuracy: Unprotected vs. Protected', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i}' for i in coeffs])
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar in bars_orig:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars_prot:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / 'accuracy_before_vs_after.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: accuracy_before_vs_after.png")

    # --- Plot 2: Accuracy drop per coefficient ---
    drops = [orig_accs[c]*100 - prot_accs_all[c]*100 for c in range(len(orig_accs))]
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2ecc71' if d > 50 else '#f39c12' if d > 20 else '#e74c3c' for d in drops]
    bars = ax.bar(x, drops, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Coefficient Index', fontsize=13)
    ax.set_ylabel('Accuracy Drop (pp)', fontsize=13)
    ax.set_title('Attack Accuracy Drop After Countermeasures', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i}' for i in coeffs])
    ax.grid(True, alpha=0.3, axis='y')
    for bar, d in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{d:.1f}pp', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / 'accuracy_drop_per_coeff.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: accuracy_drop_per_coeff.png")

    # --- Plot 3: Seen vs Unseen on protected traces ---
    if prot_accs_seen and prot_accs_unseen:
        fig, ax = plt.subplots(figsize=(14, 7))
        width = 0.25
        bars1 = ax.bar(x - width, [prot_accs_seen[c]*100 for c in range(len(prot_accs_seen))],
                        width, label=f'Seen (train+val, n={int(n_train+n_val)})',
                        color='#3498db', edgecolor='black', alpha=0.85)
        bars2 = ax.bar(x, [prot_accs_unseen[c]*100 for c in range(len(prot_accs_unseen))],
                        width, label=f'Unseen (test, n={int(n_test)})',
                        color='#e67e22', edgecolor='black', alpha=0.85)
        bars3 = ax.bar(x + width, [prot_accs_all[c]*100 for c in range(len(prot_accs_all))],
                        width, label=f'All (n={n_protected})',
                        color='#9b59b6', edgecolor='black', alpha=0.85)
        ax.axhline(random_baseline, color='gray', linestyle='--', linewidth=2,
                   label=f'Random baseline ({random_baseline:.1f}%)')
        ax.set_xlabel('Coefficient Index', fontsize=13)
        ax.set_ylabel('Accuracy (%)', fontsize=13)
        ax.set_title('Protected Accuracy by Original Split Membership', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i}' for i in coeffs])
        ax.set_ylim(0, 105)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(plots_dir / 'accuracy_by_split.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved: accuracy_by_split.png")

    # --- Plot 4: Confusion matrices for each coefficient (protected, all samples) ---
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    for c in range(NUM_COEFFICIENTS):
        ax = axes[c // 5, c % 5]
        if c in results_per_coeff:
            cm = np.array(results_per_coeff[c]['protected']['all']['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['-1', '0', '1'], yticklabels=['-1', '0', '1'])
            acc = results_per_coeff[c]['protected']['all']['accuracy'] * 100
            ax.set_title(f'C{c} ({acc:.1f}%)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    plt.suptitle('Confusion Matrices — Protected Traces (All Samples)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / 'confusion_matrices_protected.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: confusion_matrices_protected.png")

    # --- Plot 5: F1 Score comparison ---
    fig, ax = plt.subplots(figsize=(14, 7))
    bars_orig_f1 = ax.bar(x - width/2, [orig_f1s[c] for c in range(len(orig_f1s))],
                           width, label='Unprotected F1', color='#e74c3c', edgecolor='black', alpha=0.85)
    bars_prot_f1 = ax.bar(x + width/2, [prot_f1s_all[c] for c in range(len(prot_f1s_all))],
                           width, label='Protected F1', color='#2ecc71', edgecolor='black', alpha=0.85)
    ax.set_xlabel('Coefficient Index', fontsize=13)
    ax.set_ylabel('F1 Score (macro)', fontsize=13)
    ax.set_title('F1 Score: Unprotected vs. Protected', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i}' for i in coeffs])
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(plots_dir / 'f1_before_vs_after.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: f1_before_vs_after.png")

    # --- Plot 6: Summary dashboard ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: bar comparison
    ax = axes[0, 0]
    categories = ['Unprotected\n(original)', 'Protected\n(all)', 'Protected\n(seen)', 'Protected\n(unseen)', 'Random\nbaseline']
    values = [avg_orig, avg_prot_all, avg_prot_seen, avg_prot_unseen, random_baseline]
    bar_colors = ['#e74c3c', '#2ecc71', '#3498db', '#e67e22', '#95a5a6']
    bars = ax.bar(categories, values, color=bar_colors, edgecolor='black', alpha=0.85)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontsize=12)
    ax.set_title('Average Attack Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')

    # Top-right: accuracy drop distribution
    ax = axes[0, 1]
    ax.hist(drops, bins=max(5, len(set(drops))), color='#2ecc71', edgecolor='black', alpha=0.8)
    ax.axvline(np.mean(drops), color='red', linestyle='--', linewidth=2,
               label=f'Mean drop: {np.mean(drops):.1f}pp')
    ax.set_xlabel('Accuracy Drop (pp)', fontsize=12)
    ax.set_ylabel('Number of Coefficients', fontsize=12)
    ax.set_title('Distribution of Accuracy Drops', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom-left: per-coefficient radar-style
    ax = axes[1, 0]
    ax.plot(coeffs, [orig_accs[c]*100 for c in range(len(orig_accs))], 'ro-',
            linewidth=2, markersize=8, label='Unprotected')
    ax.plot(coeffs, [prot_accs_all[c]*100 for c in range(len(prot_accs_all))], 'gs-',
            linewidth=2, markersize=8, label='Protected')
    ax.axhline(random_baseline, color='gray', linestyle='--', linewidth=1.5,
               label=f'Random ({random_baseline:.1f}%)')
    ax.fill_between(coeffs, [prot_accs_all[c]*100 for c in range(len(prot_accs_all))],
                    random_baseline, alpha=0.15, color='green')
    ax.set_xlabel('Coefficient Index', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Coefficient Accuracy Trend', fontsize=14, fontweight='bold')
    ax.set_xticks(coeffs)
    ax.set_xticklabels([f'C{i}' for i in coeffs])
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Bottom-right: text summary
    ax = axes[1, 1]
    ax.axis('off')
    text_lines = [
        f"COUNTERMEASURE EVALUATION SUMMARY",
        f"",
        f"Protected samples: {n_protected}",
        f"  - Originally train: {n_train}",
        f"  - Originally val:   {n_val}",
        f"  - Originally test:  {n_test}",
        f"",
        f"Average Accuracy:",
        f"  Unprotected (original):  {avg_orig:.2f}%",
        f"  Protected (all):         {avg_prot_all:.2f}%",
        f"  Protected (seen):        {avg_prot_seen:.2f}%",
        f"  Protected (unseen):      {avg_prot_unseen:.2f}%",
        f"  Random baseline:         {random_baseline:.2f}%",
        f"",
        f"Average accuracy drop:     {avg_orig - avg_prot_all:.2f}pp",
        f"",
        f"Verdict: {'EFFECTIVE' if summary['comparison']['countermeasure_effective'] else 'PARTIALLY EFFECTIVE'}",
    ]
    ax.text(0.05, 0.95, '\n'.join(text_lines), transform=ax.transAxes,
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Countermeasure Effectiveness Dashboard', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(plots_dir / 'dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: dashboard.png")

    # =========================================================================
    # DONE
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Results: {results_file}")
    print(f"Plots:   {plots_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
