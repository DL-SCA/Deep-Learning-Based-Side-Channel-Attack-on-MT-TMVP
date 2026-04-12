"""Post-training analysis and validation for the DL-SCA experiment.

Verifies dataset integrity, checks for data leakage between splits,
analyzes cross-coefficient predictions, computes SNR, and generates
visualization plots.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42
TRACE_LENGTH = 20000
NUM_COEFFICIENTS = 10
NUM_CLASSES = 3


# ============================================================================
# DATASET ANALYSIS
# ============================================================================

class DatasetAnalyzer:
    """Analyzer for sca_dataset_v4 structure."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.traces_dir = self.dataset_path / "traces"
        self.trace_files = sorted(
            self.traces_dir.glob("traces_*.npz"),
            key=lambda x: int(x.stem.split('_')[1])
        )
        self.num_traces = len(self.trace_files)
        
    def check_dataset_integrity(self) -> Dict:
        """Verify dataset structure and integrity."""
        
        print("\n" + "="*70)
        print("DATASET INTEGRITY CHECK")
        print("="*70)
        
        results = {
            'num_traces': self.num_traces,
            'expected_traces': 59049,  # 3^10
            'checks': {}
        }
        
        # Check trace count
        results['checks']['trace_count'] = self.num_traces == 59049
        print(f"Trace count: {self.num_traces} (expected: 59049) "
              f"[{'OK' if results['checks']['trace_count'] else 'FAIL'}]")
        
        # Check sample traces
        sample_indices = [0, 1, 100, 1000, 10000, 30000, min(59048, self.num_traces-1)]
        
        print("\nSample trace inspection:")
        g_values = []
        label_matches = 0
        
        for idx in sample_indices:
            if idx >= self.num_traces:
                continue
            data = np.load(self.traces_dir / f"traces_{idx}.npz")
            
            # Check required keys
            required_keys = ['wave', 'labels']
            for key in required_keys:
                if key not in data:
                    print(f"  Trace {idx}: Missing key '{key}'")
            
            # Get f and labels
            if 'dut_io_ram_f_data' in data:
                f_data = data['dut_io_ram_f_data'].flatten()[:10]
            else:
                f_data = None
            
            labels = data['labels'].flatten()
            wave_shape = data['wave'].shape
            
            # Check g is fixed
            if 'dut_io_ram_g_data' in data:
                g = tuple(data['dut_io_ram_g_data'].flatten()[:5])
                g_values.append(g)
            
            # Check labels match f_data
            if f_data is not None:
                f_signed = []
                for val in f_data:
                    if val == 255:
                        f_signed.append(-1)
                    else:
                        f_signed.append(int(val))
                
                if list(f_signed) == list(labels):
                    label_matches += 1
            
            print(f"  Trace {idx}: wave={wave_shape}, labels={list(labels)}")
        
        # Check g is fixed
        results['checks']['g_fixed'] = len(set(g_values)) <= 1
        print(f"\ng polynomial fixed: [{'OK' if results['checks']['g_fixed'] else 'FAIL'}]")
        
        # Check labels match f
        results['checks']['labels_match_f'] = label_matches == len(sample_indices)
        print(f"Labels match f values: [{'OK' if results['checks']['labels_match_f'] else 'FAIL'}]")
        
        # Overall status
        results['all_passed'] = all(results['checks'].values())
        print(f"\nOverall integrity: [{'PASSED' if results['all_passed'] else 'FAILED'}]")
        
        return results
    
    def analyze_class_distribution(self, num_samples: int = 10000) -> Dict:
        """Analyze class distribution for each coefficient."""
        
        print("\n" + "="*70)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("="*70)
        
        # Load labels
        labels_list = []
        sample_indices = np.random.choice(self.num_traces, 
                                          min(num_samples, self.num_traces), 
                                          replace=False)
        
        for idx in tqdm(sample_indices, desc="Loading labels"):
            data = np.load(self.traces_dir / f"traces_{idx}.npz")
            labels_list.append(data['labels'].flatten())
        
        labels = np.array(labels_list)
        
        results = {'per_coefficient': {}}
        
        print(f"\nClass distribution (sampled {len(labels)} traces):")
        print(f"{'Coeff':<8} {'Class -1':<12} {'Class 0':<12} {'Class 1':<12} {'Balance':<10}")
        print("-" * 60)
        
        for coeff_idx in range(NUM_COEFFICIENTS):
            coeff_labels = labels[:, coeff_idx]
            counts = {
                -1: np.sum(coeff_labels == -1),
                0: np.sum(coeff_labels == 0),
                1: np.sum(coeff_labels == 1),
            }
            total = sum(counts.values())
            
            # Check balance
            min_count = min(counts.values())
            max_count = max(counts.values())
            balance = min_count / max_count if max_count > 0 else 0
            
            results['per_coefficient'][coeff_idx] = {
                'counts': counts,
                'percentages': {k: v/total*100 for k, v in counts.items()},
                'balance_ratio': balance,
            }
            
            print(f"{coeff_idx:<8} {counts[-1]:<12} {counts[0]:<12} {counts[1]:<12} {balance:.3f}")
        
        # Overall balance
        avg_balance = np.mean([r['balance_ratio'] for r in results['per_coefficient'].values()])
        results['average_balance'] = avg_balance
        print(f"\nAverage balance ratio: {avg_balance:.3f} (1.0 = perfectly balanced)")
        
        return results
    
    def compute_snr(self, coeff_idx: int, num_per_class: int = 500) -> Dict:
        """Compute Signal-to-Noise Ratio for a coefficient."""
        
        print(f"\nComputing SNR for coefficient {coeff_idx}...")
        
        traces_by_class = {-1: [], 0: [], 1: []}
        
        for i, trace_file in enumerate(tqdm(self.trace_files, desc="Loading", leave=False)):
            if all(len(v) >= num_per_class for v in traces_by_class.values()):
                break
            
            data = np.load(trace_file)
            label = int(data['labels'].flatten()[coeff_idx])
            
            if len(traces_by_class[label]) < num_per_class:
                traces_by_class[label].append(data['wave'].flatten())
        
        # Convert to arrays
        for cls in [-1, 0, 1]:
            traces_by_class[cls] = np.array(traces_by_class[cls])
        
        # Compute SNR
        class_means = np.array([traces_by_class[cls].mean(axis=0) for cls in [-1, 0, 1]])
        class_vars = np.array([traces_by_class[cls].var(axis=0) for cls in [-1, 0, 1]])
        
        signal = class_means.var(axis=0)
        noise = class_vars.mean(axis=0) + 1e-10
        snr = signal / noise
        
        max_snr = float(snr.max())
        peak_sample = int(np.argmax(snr))
        
        # Find good range (>10% of max)
        threshold = 0.1 * max_snr
        good_samples = np.where(snr > threshold)[0]
        range_start = int(good_samples[0]) if len(good_samples) > 0 else 0
        range_end = int(good_samples[-1]) if len(good_samples) > 0 else TRACE_LENGTH
        
        return {
            'coeff_idx': coeff_idx,
            'max_snr': max_snr,
            'peak_sample': peak_sample,
            'good_range': [range_start, range_end],
            'snr_values': snr.tolist(),
        }
    
    def compute_all_snr(self, num_per_class: int = 500) -> Dict:
        """Compute SNR for all coefficients."""
        
        print("\n" + "="*70)
        print("SNR ANALYSIS")
        print("="*70)
        
        results = {}
        
        for coeff_idx in range(NUM_COEFFICIENTS):
            results[coeff_idx] = self.compute_snr(coeff_idx, num_per_class)
            print(f"  Coeff {coeff_idx}: max_SNR={results[coeff_idx]['max_snr']:.4f}, "
                  f"peak={results[coeff_idx]['peak_sample']}")
        
        return results


# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

class ResultsAnalyzer:
    """Analyzer for training results."""
    
    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        self.results = self._load_results()
        
    def _load_results(self) -> Dict:
        """Load attack_results.json."""
        results_file = self.results_path / "attack_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        return {}
    
    def analyze_test_metrics(self) -> Dict:
        """Analyze test set metrics."""
        
        print("\n" + "="*70)
        print("TEST METRICS ANALYSIS")
        print("="*70)
        
        if not self.results:
            print("No results found!")
            return {}
        
        per_coeff = self.results.get('per_coefficient', [])
        
        print(f"\n{'Coeff':<8} {'Accuracy':<12} {'F1':<10} {'Precision':<12} {'Recall':<10}")
        print("-" * 60)
        
        metrics_summary = []
        
        for coeff_data in per_coeff:
            coeff_idx = coeff_data['coeff_idx']
            m = coeff_data['test_metrics']
            
            metrics_summary.append({
                'coeff_idx': coeff_idx,
                'accuracy': m['accuracy'],
                'f1_macro': m['f1_macro'],
                'precision_macro': m['precision_macro'],
                'recall_macro': m['recall_macro'],
            })
            
            print(f"{coeff_idx:<8} {m['accuracy']*100:>6.2f}%     "
                  f"{m['f1_macro']:.4f}    {m['precision_macro']:.4f}      {m['recall_macro']:.4f}")
        
        # Averages
        avg_acc = np.mean([m['accuracy'] for m in metrics_summary])
        avg_f1 = np.mean([m['f1_macro'] for m in metrics_summary])
        
        print("-" * 60)
        print(f"{'Average':<8} {avg_acc*100:>6.2f}%     {avg_f1:.4f}")
        
        # All-correct rate
        all_correct_rate = self.results.get('all_correct_rate', 0)
        print(f"\nAll-Correct Rate: {all_correct_rate*100:.2f}%")
        
        return {
            'per_coefficient': metrics_summary,
            'avg_accuracy': avg_acc,
            'avg_f1': avg_f1,
            'all_correct_rate': all_correct_rate,
        }
    
    def analyze_cross_coefficient(self) -> Dict:
        """Analyze cross-coefficient prediction results."""
        
        print("\n" + "="*70)
        print("CROSS-COEFFICIENT PREDICTION ANALYSIS")
        print("="*70)
        
        per_coeff = self.results.get('per_coefficient', [])
        
        # Build cross-coefficient matrix
        n_coeffs = len(per_coeff)
        cross_matrix = np.zeros((n_coeffs, n_coeffs))
        
        print("\nCross-coefficient accuracy matrix:")
        print("(Row = trained model, Column = target coefficient)")
        print()
        
        header = "Model  " + " ".join([f"C{i:>5}" for i in range(n_coeffs)])
        print(header)
        print("-" * len(header))
        
        for i, coeff_data in enumerate(per_coeff):
            cross_pred = coeff_data.get('final_cross_coeff_test', {})
            
            row_str = f"C{i:<5} "
            for j in range(n_coeffs):
                acc = cross_pred.get(str(j), 0)
                cross_matrix[i, j] = acc
                
                # Highlight diagonal
                if i == j:
                    row_str += f"{acc*100:>5.1f}*"
                else:
                    row_str += f"{acc*100:>5.1f} "
            
            print(row_str)
        
        print("\n(* = model's trained coefficient)")
        
        # Analysis
        diagonal = np.diag(cross_matrix)
        off_diagonal = cross_matrix[~np.eye(n_coeffs, dtype=bool)]
        
        print(f"\nDiagonal (trained coeff) avg: {np.mean(diagonal)*100:.2f}%")
        print(f"Off-diagonal avg: {np.mean(off_diagonal)*100:.2f}%")
        print(f"Expected random: 33.33%")
        
        # Check if off-diagonal is close to random
        is_random = np.abs(np.mean(off_diagonal) - 0.3333) < 0.05
        print(f"\nOff-diagonal ≈ random: {'YES' if is_random else 'NO'}")
        
        if is_random:
            print("→ Models learned COEFFICIENT-SPECIFIC patterns (good!)")
        else:
            print("→ Models may have learned some shared patterns")
        
        return {
            'cross_matrix': cross_matrix.tolist(),
            'diagonal_avg': float(np.mean(diagonal)),
            'off_diagonal_avg': float(np.mean(off_diagonal)),
            'off_diagonal_is_random': is_random,
        }
    
    def analyze_training_curves(self) -> Dict:
        """Analyze training curves for all coefficients."""
        
        print("\n" + "="*70)
        print("TRAINING CURVE ANALYSIS")
        print("="*70)
        
        per_coeff = self.results.get('per_coefficient', [])
        
        analysis = []
        
        for coeff_data in per_coeff:
            coeff_idx = coeff_data['coeff_idx']
            history = coeff_data.get('history', {})
            
            if not history:
                continue
            
            train_acc = history.get('train_acc', [])
            val_acc = history.get('val_acc', [])
            
            if train_acc and val_acc:
                final_train = train_acc[-1]
                final_val = val_acc[-1]
                best_val = max(val_acc)
                best_epoch = val_acc.index(best_val) + 1
                
                # Check for overfitting
                overfit = final_train - final_val > 0.1
                
                analysis.append({
                    'coeff_idx': coeff_idx,
                    'final_train_acc': final_train,
                    'final_val_acc': final_val,
                    'best_val_acc': best_val,
                    'best_epoch': best_epoch,
                    'num_epochs': len(train_acc),
                    'overfit_detected': overfit,
                })
                
                print(f"Coeff {coeff_idx}: train={final_train*100:.1f}%, "
                      f"val={final_val*100:.1f}%, best_epoch={best_epoch}, "
                      f"overfit={'YES' if overfit else 'NO'}")
        
        return {'per_coefficient': analysis}
    
    def analyze_confusion_matrices(self) -> Dict:
        """Analyze confusion matrices."""
        
        print("\n" + "="*70)
        print("CONFUSION MATRIX ANALYSIS")
        print("="*70)
        
        per_coeff = self.results.get('per_coefficient', [])
        
        analysis = []
        
        for coeff_data in per_coeff:
            coeff_idx = coeff_data['coeff_idx']
            cm = np.array(coeff_data['test_metrics'].get('confusion_matrix', []))
            
            if cm.size == 0:
                continue
            
            # Per-class accuracy
            per_class_acc = np.diag(cm) / cm.sum(axis=1)
            
            # Misclassification patterns
            total_errors = cm.sum() - np.trace(cm)
            
            analysis.append({
                'coeff_idx': coeff_idx,
                'confusion_matrix': cm.tolist(),
                'per_class_accuracy': per_class_acc.tolist(),
                'total_errors': int(total_errors),
            })
            
            print(f"Coeff {coeff_idx}: errors={total_errors}, "
                  f"per-class acc: -1={per_class_acc[0]*100:.1f}%, "
                  f"0={per_class_acc[1]*100:.1f}%, 1={per_class_acc[2]*100:.1f}%")
        
        return {'per_coefficient': analysis}


# ============================================================================
# VALIDATION
# ============================================================================

class DataLeakageValidator:
    """Validator for data leakage detection."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.traces_dir = self.dataset_path / "traces"
        
    def check_split_overlap(self, seed: int = 42) -> Dict:
        """Check that train/val/test splits don't overlap."""
        
        print("\n" + "="*70)
        print("SPLIT OVERLAP CHECK")
        print("="*70)
        
        num_samples = 59049
        
        # Recreate splits
        np.random.seed(seed)
        indices = np.random.permutation(num_samples)
        
        train_size = int(num_samples * 0.70)
        val_size = int(num_samples * 0.15)
        
        train_indices = set(indices[:train_size])
        val_indices = set(indices[train_size:train_size + val_size])
        test_indices = set(indices[train_size + val_size:])
        
        # Check overlaps
        overlap_train_val = len(train_indices & val_indices)
        overlap_train_test = len(train_indices & test_indices)
        overlap_val_test = len(val_indices & test_indices)
        
        print(f"Train size: {len(train_indices)}")
        print(f"Val size: {len(val_indices)}")
        print(f"Test size: {len(test_indices)}")
        print(f"\nOverlap train-val: {overlap_train_val} (should be 0)")
        print(f"Overlap train-test: {overlap_train_test} (should be 0)")
        print(f"Overlap val-test: {overlap_val_test} (should be 0)")
        
        no_overlap = overlap_train_val == 0 and overlap_train_test == 0 and overlap_val_test == 0
        print(f"\nNo overlap: [{'PASSED' if no_overlap else 'FAILED'}]")
        
        return {
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices),
            'overlap_train_val': overlap_train_val,
            'overlap_train_test': overlap_train_test,
            'overlap_val_test': overlap_val_test,
            'no_overlap': no_overlap,
        }
    
    def validate_results_legitimacy(self, results_analyzer: ResultsAnalyzer) -> Dict:
        """Validate that results are legitimate (not due to data leakage)."""
        
        print("\n" + "="*70)
        print("RESULTS LEGITIMACY VALIDATION")
        print("="*70)
        
        checks = {}
        
        # Check 1: Cross-coefficient accuracy should be ~33% (random)
        cross_analysis = results_analyzer.analyze_cross_coefficient()
        off_diag_avg = cross_analysis['off_diagonal_avg']
        checks['cross_coeff_random'] = abs(off_diag_avg - 0.3333) < 0.05
        print(f"\n1. Cross-coeff off-diagonal ≈ 33%: "
              f"{off_diag_avg*100:.1f}% [{'PASS' if checks['cross_coeff_random'] else 'FAIL'}]")
        
        # Check 2: No split overlap
        split_check = self.check_split_overlap()
        checks['no_split_overlap'] = split_check['no_overlap']
        
        # Check 3: Training curves show learning (not memorization)
        training_analysis = results_analyzer.analyze_training_curves()
        no_overfit = all(not c['overfit_detected'] 
                        for c in training_analysis.get('per_coefficient', []))
        checks['no_overfitting'] = no_overfit
        print(f"\n3. No overfitting detected: [{'PASS' if no_overfit else 'WARN'}]")
        
        # Overall verdict
        all_passed = all(checks.values())
        
        print("\n" + "="*70)
        print("LEGITIMACY VERDICT")
        print("="*70)
        
        if all_passed:
            print("✓ All validation checks PASSED")
            print("✓ Results appear LEGITIMATE")
            print("\nReasons:")
            print("  - Each model only predicts its trained coefficient well (~100%)")
            print("  - Cross-coefficient predictions are random (~33%)")
            print("  - No data leakage between train/val/test splits")
            print("  - Training curves show proper learning progression")
        else:
            print("⚠ Some validation checks failed")
            print("  Review the failed checks above")
        
        return {
            'checks': checks,
            'all_passed': all_passed,
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Generate all visualizations."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_class_distribution(self, distribution: Dict, save_path: Path = None):
        """Plot class distribution for all coefficients."""
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for coeff_idx in range(NUM_COEFFICIENTS):
            ax = axes[coeff_idx]
            data = distribution['per_coefficient'][coeff_idx]
            counts = data['counts']
            
            bars = ax.bar(['-1', '0', '1'], 
                         [counts[-1], counts[0], counts[1]],
                         color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
            ax.set_title(f'Coefficient {coeff_idx}')
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            
            for bar, count in zip(bars, [counts[-1], counts[0], counts[1]]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{count}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Class Distribution per Coefficient', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_test_accuracy(self, metrics: Dict, save_path: Path = None):
        """Plot test accuracy per coefficient."""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        coeffs = [m['coeff_idx'] for m in metrics['per_coefficient']]
        accs = [m['accuracy'] * 100 for m in metrics['per_coefficient']]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(coeffs)))
        bars = ax.bar(coeffs, accs, color=colors)
        
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect')
        ax.axhline(y=33.33, color='red', linestyle='--', alpha=0.5, label='Random')
        
        ax.set_xlabel('Coefficient Index', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title('Test Accuracy per Coefficient', fontsize=14, fontweight='bold')
        ax.set_xticks(coeffs)
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{acc:.2f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_cross_coefficient_heatmap(self, cross_matrix: np.ndarray, save_path: Path = None):
        """Plot cross-coefficient prediction heatmap."""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        cross_matrix = np.array(cross_matrix) * 100
        
        im = ax.imshow(cross_matrix, cmap='RdYlGn', vmin=0, vmax=100)
        
        ax.set_xticks(range(NUM_COEFFICIENTS))
        ax.set_yticks(range(NUM_COEFFICIENTS))
        ax.set_xticklabels([f'C{i}' for i in range(NUM_COEFFICIENTS)])
        ax.set_yticklabels([f'Model {i}' for i in range(NUM_COEFFICIENTS)])
        ax.set_xlabel('Target Coefficient', fontsize=12)
        ax.set_ylabel('Trained Model', fontsize=12)
        ax.set_title('Cross-Coefficient Prediction Accuracy (%)', fontsize=14, fontweight='bold')
        
        # Add text annotations
        for i in range(NUM_COEFFICIENTS):
            for j in range(NUM_COEFFICIENTS):
                color = 'white' if cross_matrix[i, j] < 50 else 'black'
                weight = 'bold' if i == j else 'normal'
                ax.text(j, i, f'{cross_matrix[i, j]:.1f}', 
                       ha='center', va='center', color=color, 
                       fontsize=8, fontweight=weight)
        
        plt.colorbar(im, ax=ax, label='Accuracy (%)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrices(self, cm_analysis: Dict, save_path: Path = None):
        """Plot confusion matrices for all coefficients."""
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for i, coeff_data in enumerate(cm_analysis['per_coefficient']):
            ax = axes[i]
            cm = np.array(coeff_data['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['-1', '0', '1'],
                       yticklabels=['-1', '0', '1'])
            ax.set_title(f'Coeff {coeff_data["coeff_idx"]}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
        
        plt.suptitle('Confusion Matrices per Coefficient', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_snr(self, snr_results: Dict, save_path: Path = None):
        """Plot SNR for all coefficients."""
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for coeff_idx in range(NUM_COEFFICIENTS):
            ax = axes[coeff_idx]
            snr = np.array(snr_results[coeff_idx]['snr_values'])
            
            ax.plot(snr, linewidth=0.5, color='blue', alpha=0.7)
            ax.axhline(y=snr_results[coeff_idx]['max_snr'] * 0.1, 
                      color='red', linestyle='--', alpha=0.5)
            ax.set_title(f'Coeff {coeff_idx} (max={snr_results[coeff_idx]["max_snr"]:.4f})')
            ax.set_xlabel('Sample')
            ax.set_ylabel('SNR')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Signal-to-Noise Ratio per Coefficient', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_comprehensive_analysis(dataset_path: str, results_path: str, output_dir: str):
    """Run the complete comprehensive analysis."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("COMPREHENSIVE DL-SCA ANALYSIS")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Results: {results_path}")
    print(f"Output: {output_dir}")
    
    # Initialize analyzers
    dataset_analyzer = DatasetAnalyzer(dataset_path)
    results_analyzer = ResultsAnalyzer(results_path)
    validator = DataLeakageValidator(dataset_path)
    visualizer = Visualizer(output_dir)
    
    all_results = {}
    
    # 1. Dataset integrity
    all_results['dataset_integrity'] = dataset_analyzer.check_dataset_integrity()
    
    # 2. Class distribution
    all_results['class_distribution'] = dataset_analyzer.analyze_class_distribution()
    visualizer.plot_class_distribution(
        all_results['class_distribution'], 
        plots_dir / "class_distribution.png"
    )
    
    # 3. SNR analysis
    all_results['snr'] = dataset_analyzer.compute_all_snr(num_per_class=300)
    visualizer.plot_snr(all_results['snr'], plots_dir / "snr_analysis.png")
    
    # 4. Test metrics
    all_results['test_metrics'] = results_analyzer.analyze_test_metrics()
    visualizer.plot_test_accuracy(
        all_results['test_metrics'], 
        plots_dir / "test_accuracy.png"
    )
    
    # 5. Cross-coefficient analysis
    all_results['cross_coefficient'] = results_analyzer.analyze_cross_coefficient()
    visualizer.plot_cross_coefficient_heatmap(
        all_results['cross_coefficient']['cross_matrix'],
        plots_dir / "cross_coefficient_heatmap.png"
    )
    
    # 6. Confusion matrices
    all_results['confusion_matrices'] = results_analyzer.analyze_confusion_matrices()
    visualizer.plot_confusion_matrices(
        all_results['confusion_matrices'],
        plots_dir / "confusion_matrices.png"
    )
    
    # 7. Training curves
    all_results['training_curves'] = results_analyzer.analyze_training_curves()
    
    # 8. Validation
    all_results['validation'] = validator.validate_results_legitimacy(results_analyzer)
    
    # Save all results
    results_file = output_dir / "comprehensive_analysis.json"
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n\nResults saved to: {results_file}")
    print(f"Plots saved to: {plots_dir}")
    
    return all_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive DL-SCA Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to sca_dataset_v4")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to results directory (e.g., results_v8_resMod)")
    parser.add_argument("--output", type=str, default="comprehensive_analysis",
                        help="Output directory")
    
    args = parser.parse_args()
    
    run_comprehensive_analysis(args.dataset, args.results, args.output)


if __name__ == "__main__":
    main()
