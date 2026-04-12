"""Test Vector Leakage Assessment (TVLA) for MT-TMVP power traces.

Implements Welch's t-test to detect side-channel leakage across all
10 target coefficients. Generates per-coefficient plots and a summary report.
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

TVLA_THRESHOLD = 4.5  # Standard threshold for 99.9999% confidence
TRACE_LENGTH = 20000
NUM_COEFFICIENTS = 10


# ============================================================================
# TVLA ANALYZER CLASS
# ============================================================================

class TVLAAnalyzer:
    """TVLA analyzer for sca_dataset_v4 structure."""
    
    def __init__(self, dataset_path: str, output_dir: str = None):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir) if output_dir else Path("tvla_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.traces_dir = self.dataset_path / "traces"
        self.trace_files = sorted(
            self.traces_dir.glob("traces_*.npz"),
            key=lambda x: int(x.stem.split('_')[1])
        )
        self.num_traces = len(self.trace_files)
        
        print(f"TVLA Analyzer initialized")
        print(f"  Dataset: {self.dataset_path}")
        print(f"  Traces: {self.num_traces}")
        print(f"  Output: {self.output_dir}")
    
    def load_traces_by_coefficient(self, coeff_idx: int, max_per_class: int = 1000
                                   ) -> Dict[int, np.ndarray]:
        """
        Load traces grouped by coefficient value.
        
        Args:
            coeff_idx: Coefficient index (0-9)
            max_per_class: Maximum traces per class
            
        Returns:
            Dictionary mapping class (-1, 0, 1) to trace arrays
        """
        traces_by_class = {-1: [], 0: [], 1: []}
        counts = {-1: 0, 0: 0, 1: 0}
        
        # Determine sampling strategy based on coefficient position
        # Coefficient i varies every 3^i traces
        period = 3 ** coeff_idx
        
        print(f"  Loading traces for coefficient {coeff_idx} (period={period})...")
        
        for i, trace_file in enumerate(tqdm(self.trace_files, desc=f"Coeff {coeff_idx}", leave=False)):
            # Check if we have enough for all classes
            if all(c >= max_per_class for c in counts.values()):
                break
            
            data = np.load(trace_file)
            label = int(data['labels'].flatten()[coeff_idx])
            
            if counts[label] < max_per_class:
                traces_by_class[label].append(data['wave'].flatten())
                counts[label] += 1
        
        # Convert to arrays
        for cls in [-1, 0, 1]:
            if traces_by_class[cls]:
                traces_by_class[cls] = np.array(traces_by_class[cls], dtype=np.float32)
            else:
                traces_by_class[cls] = np.array([], dtype=np.float32)
        
        print(f"    Loaded: class -1: {len(traces_by_class[-1])}, "
              f"class 0: {len(traces_by_class[0])}, class 1: {len(traces_by_class[1])}")
        
        return traces_by_class
    
    @staticmethod
    def welch_ttest(group1: np.ndarray, group2: np.ndarray) -> np.ndarray:
        """
        Compute Welch's t-test for each sample point.
        
        Args:
            group1: First group of traces (N1, samples)
            group2: Second group of traces (N2, samples)
            
        Returns:
            Array of t-values for each sample point
        """
        n1, n2 = len(group1), len(group2)
        
        mean1 = np.mean(group1, axis=0)
        mean2 = np.mean(group2, axis=0)
        
        var1 = np.var(group1, axis=0, ddof=1)
        var2 = np.var(group2, axis=0, ddof=1)
        
        # Welch's t-statistic
        se = np.sqrt(var1/n1 + var2/n2 + 1e-12)
        t_values = (mean1 - mean2) / se
        
        return t_values
    
    def run_tvla_coefficient(self, coeff_idx: int, num_traces: int = 1000,
                             comparison: str = "0_vs_1") -> Dict:
        """
        Run TVLA for a specific coefficient.
        
        Args:
            coeff_idx: Coefficient index (0-9)
            num_traces: Number of traces per class
            comparison: Which classes to compare ("0_vs_1", "0_vs_-1", "-1_vs_1")
            
        Returns:
            Dictionary with TVLA results
        """
        print(f"\n{'='*60}")
        print(f"TVLA for Coefficient {coeff_idx} ({comparison})")
        print(f"{'='*60}")
        
        # Load traces
        traces_by_class = self.load_traces_by_coefficient(coeff_idx, num_traces)
        
        # Parse comparison
        if comparison == "0_vs_1":
            group1, group2 = traces_by_class[0], traces_by_class[1]
            label1, label2 = "0", "1"
        elif comparison == "0_vs_-1":
            group1, group2 = traces_by_class[0], traces_by_class[-1]
            label1, label2 = "0", "-1"
        else:  # -1_vs_1
            group1, group2 = traces_by_class[-1], traces_by_class[1]
            label1, label2 = "-1", "1"
        
        if len(group1) == 0 or len(group2) == 0:
            print(f"  ERROR: Not enough traces for comparison")
            return None
        
        # Compute t-test
        t_values = self.welch_ttest(group1, group2)
        
        # Analyze results
        max_t = np.max(np.abs(t_values))
        leakage_detected = max_t >= TVLA_THRESHOLD
        leakage_points = np.where(np.abs(t_values) >= TVLA_THRESHOLD)[0]
        
        # Find regions of interest (ROI)
        roi_start, roi_end = None, None
        if len(leakage_points) > 0:
            roi_start = int(leakage_points[0])
            roi_end = int(leakage_points[-1])
        
        results = {
            'coeff_idx': coeff_idx,
            'comparison': comparison,
            'num_traces_group1': len(group1),
            'num_traces_group2': len(group2),
            'max_t_value': float(max_t),
            'leakage_detected': leakage_detected,
            'num_leakage_points': len(leakage_points),
            'roi_start': roi_start,
            'roi_end': roi_end,
            't_values': t_values.tolist(),
        }
        
        print(f"  Group 1 ({label1}): {len(group1)} traces")
        print(f"  Group 2 ({label2}): {len(group2)} traces")
        print(f"  Max |t|: {max_t:.2f}")
        print(f"  Leakage detected: {'YES' if leakage_detected else 'NO'}")
        if leakage_detected:
            print(f"  Leakage points: {len(leakage_points)}")
            print(f"  ROI: [{roi_start}, {roi_end}]")
        
        return results
    
    def run_tvla_all_coefficients(self, num_traces: int = 1000) -> Dict:
        """Run TVLA for all 10 coefficients."""
        
        print("\n" + "="*70)
        print("TVLA ANALYSIS FOR ALL COEFFICIENTS")
        print("="*70)
        
        all_results = {}
        
        for coeff_idx in range(NUM_COEFFICIENTS):
            results = self.run_tvla_coefficient(coeff_idx, num_traces, "0_vs_1")
            if results:
                all_results[coeff_idx] = results
        
        return all_results
    
    def run_progressive_tvla(self, coeff_idx: int, 
                             trace_counts: List[int] = None) -> Dict:
        """
        Run TVLA with increasing number of traces to see convergence.
        
        Args:
            coeff_idx: Coefficient to analyze
            trace_counts: List of trace counts to test
            
        Returns:
            Dictionary with progressive results
        """
        if trace_counts is None:
            trace_counts = [100, 250, 500, 1000, 2000]
        
        print(f"\n{'='*60}")
        print(f"Progressive TVLA for Coefficient {coeff_idx}")
        print(f"{'='*60}")
        
        progressive_results = []
        
        for n in trace_counts:
            result = self.run_tvla_coefficient(coeff_idx, n, "0_vs_1")
            if result:
                progressive_results.append({
                    'num_traces': n,
                    'max_t': result['max_t_value'],
                    'leakage_detected': result['leakage_detected'],
                    'num_leakage_points': result['num_leakage_points'],
                })
        
        return {
            'coeff_idx': coeff_idx,
            'trace_counts': trace_counts,
            'results': progressive_results,
        }
    
    def plot_tvla_results(self, results: Dict, save_path: Path = None):
        """Plot TVLA results for a single coefficient."""
        
        coeff_idx = results['coeff_idx']
        t_values = np.array(results['t_values'])
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Full trace t-values
        ax = axes[0]
        ax.plot(t_values, linewidth=0.5, color='blue', alpha=0.7)
        ax.axhline(y=TVLA_THRESHOLD, color='red', linestyle='--', label=f'Threshold (±{TVLA_THRESHOLD})')
        ax.axhline(y=-TVLA_THRESHOLD, color='red', linestyle='--')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('t-value')
        ax.set_title(f'TVLA Results - Coefficient {coeff_idx} ({results["comparison"]})\n'
                     f'Max |t| = {results["max_t_value"]:.2f}, '
                     f'Leakage: {"YES" if results["leakage_detected"] else "NO"}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Zoomed view on ROI
        ax = axes[1]
        if results['roi_start'] is not None:
            roi_start = max(0, results['roi_start'] - 500)
            roi_end = min(len(t_values), results['roi_end'] + 500)
            ax.plot(range(roi_start, roi_end), t_values[roi_start:roi_end], 
                   linewidth=0.8, color='blue')
            ax.axhline(y=TVLA_THRESHOLD, color='red', linestyle='--')
            ax.axhline(y=-TVLA_THRESHOLD, color='red', linestyle='--')
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax.fill_between(range(roi_start, roi_end), 
                           -TVLA_THRESHOLD, TVLA_THRESHOLD, 
                           alpha=0.1, color='green', label='Safe zone')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('t-value')
            ax.set_title(f'Zoomed View: ROI [{results["roi_start"]}, {results["roi_end"]}]')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No significant leakage detected', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('No ROI to display')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Plot saved: {save_path}")
        plt.close()
    
    def plot_summary(self, all_results: Dict, save_path: Path = None):
        """Plot summary of TVLA results for all coefficients."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        coeffs = sorted(all_results.keys())
        max_t_values = [all_results[c]['max_t_value'] for c in coeffs]
        leakage_points = [all_results[c]['num_leakage_points'] for c in coeffs]
        
        # Max t-value per coefficient
        ax = axes[0, 0]
        colors = ['red' if t >= TVLA_THRESHOLD else 'green' for t in max_t_values]
        bars = ax.bar(coeffs, max_t_values, color=colors, alpha=0.7)
        ax.axhline(y=TVLA_THRESHOLD, color='red', linestyle='--', 
                  label=f'Threshold ({TVLA_THRESHOLD})')
        ax.set_xlabel('Coefficient Index')
        ax.set_ylabel('Max |t-value|')
        ax.set_title('Maximum t-value per Coefficient')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Number of leakage points
        ax = axes[0, 1]
        ax.bar(coeffs, leakage_points, color='steelblue', alpha=0.7)
        ax.set_xlabel('Coefficient Index')
        ax.set_ylabel('Number of Leakage Points')
        ax.set_title('Leakage Points per Coefficient (|t| ≥ 4.5)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Heatmap of t-values (subsampled)
        ax = axes[1, 0]
        t_matrix = []
        for c in coeffs:
            t_vals = np.array(all_results[c]['t_values'])
            # Subsample to 200 points for visualization
            subsample = t_vals[::100][:200]
            t_matrix.append(subsample)
        t_matrix = np.array(t_matrix)
        
        im = ax.imshow(np.abs(t_matrix), aspect='auto', cmap='hot', 
                      vmin=0, vmax=max(TVLA_THRESHOLD * 2, np.max(np.abs(t_matrix))))
        ax.set_xlabel('Sample Index (subsampled)')
        ax.set_ylabel('Coefficient Index')
        ax.set_title('|t-value| Heatmap Across Coefficients')
        ax.set_yticks(range(len(coeffs)))
        ax.set_yticklabels(coeffs)
        plt.colorbar(im, ax=ax, label='|t-value|')
        
        # Summary table
        ax = axes[1, 1]
        ax.axis('off')
        
        table_data = []
        for c in coeffs:
            r = all_results[c]
            table_data.append([
                f"C{c}",
                f"{r['max_t_value']:.1f}",
                "YES" if r['leakage_detected'] else "NO",
                f"{r['num_leakage_points']}",
                f"[{r['roi_start']}, {r['roi_end']}]" if r['roi_start'] else "N/A"
            ])
        
        table = ax.table(
            cellText=table_data,
            colLabels=['Coeff', 'Max |t|', 'Leakage', 'Points', 'ROI'],
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.set_title('TVLA Summary Table', fontsize=12, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Summary plot saved: {save_path}")
        plt.close()
    
    def generate_report(self, all_results: Dict) -> Dict:
        """Generate a summary report."""
        
        coeffs_with_leakage = [c for c, r in all_results.items() if r['leakage_detected']]
        avg_max_t = np.mean([r['max_t_value'] for r in all_results.values()])
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset': str(self.dataset_path),
            'threshold': TVLA_THRESHOLD,
            'num_coefficients_analyzed': len(all_results),
            'coefficients_with_leakage': coeffs_with_leakage,
            'num_coefficients_with_leakage': len(coeffs_with_leakage),
            'average_max_t': float(avg_max_t),
            'per_coefficient': {
                int(c): {
                    'max_t': r['max_t_value'],
                    'leakage_detected': r['leakage_detected'],
                    'num_leakage_points': r['num_leakage_points'],
                    'roi': [r['roi_start'], r['roi_end']] if r['roi_start'] else None,
                }
                for c, r in all_results.items()
            }
        }
        
        return report
    
    def run_full_analysis(self, num_traces: int = 1000, 
                          progressive: bool = False) -> Dict:
        """Run complete TVLA analysis."""
        
        print("\n" + "="*70)
        print("FULL TVLA ANALYSIS")
        print("="*70)
        print(f"Traces per class: {num_traces}")
        print(f"Threshold: {TVLA_THRESHOLD}")
        
        # Run TVLA for all coefficients
        all_results = self.run_tvla_all_coefficients(num_traces)
        
        # Generate plots
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        for coeff_idx, results in all_results.items():
            self.plot_tvla_results(results, plots_dir / f"tvla_coeff_{coeff_idx}.png")
        
        self.plot_summary(all_results, plots_dir / "tvla_summary.png")
        
        # Generate report
        report = self.generate_report(all_results)
        
        # Save report
        report_path = self.output_dir / "tvla_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved: {report_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("TVLA ANALYSIS SUMMARY")
        print("="*70)
        print(f"Coefficients analyzed: {len(all_results)}")
        print(f"Coefficients with leakage: {len(report['coefficients_with_leakage'])}")
        print(f"Average max |t|: {report['average_max_t']:.2f}")
        print(f"\nPer-coefficient results:")
        for c in sorted(all_results.keys()):
            r = all_results[c]
            status = "LEAKAGE" if r['leakage_detected'] else "OK"
            print(f"  Coeff {c}: max|t|={r['max_t_value']:6.2f} [{status}]")
        
        return report


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TVLA Analysis for sca_dataset_v4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tvla.py --dataset ../dataset
  python tvla.py --dataset ../dataset --traces 2000
  python tvla.py --dataset ../dataset --coefficient 0
        """
    )
    
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to sca_dataset_v4 directory")
    parser.add_argument("--output", type=str, default="tvla_results",
                        help="Output directory (default: tvla_results)")
    parser.add_argument("--traces", type=int, default=1000,
                        help="Number of traces per class (default: 1000)")
    parser.add_argument("--coefficient", type=int, default=None,
                        help="Analyze specific coefficient only")
    parser.add_argument("--progressive", action="store_true",
                        help="Run progressive TVLA analysis")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TVLAAnalyzer(args.dataset, args.output)
    
    if args.coefficient is not None:
        # Single coefficient analysis
        if args.progressive:
            results = analyzer.run_progressive_tvla(args.coefficient)
        else:
            results = analyzer.run_tvla_coefficient(args.coefficient, args.traces)
            if results:
                analyzer.plot_tvla_results(
                    results, 
                    analyzer.output_dir / f"tvla_coeff_{args.coefficient}.png"
                )
    else:
        # Full analysis
        analyzer.run_full_analysis(args.traces, args.progressive)


if __name__ == "__main__":
    main()
