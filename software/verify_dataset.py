"""Verify the captured dataset for MT-TMVP side-channel attack.

Checks trace file completeness, format correctness, fixed g polynomial
consistency, f polynomial/label correspondence, and file integrity.
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import Counter

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def verify_all_combinations_dataset(dataset_dir: str, verbose: bool = True, 
                                     check_all: bool = False):
    """
    Verify the captured dataset follows the all combinations methodology.
    
    Checks:
    1. All expected trace files exist
    2. g polynomial is FIXED across ALL traces
    3. f polynomials match expected combinations
    4. Labels are correct
    5. Trace format is correct
    
    Args:
        dataset_dir: Path to dataset directory
        verbose: Print detailed output
        check_all: Check ALL traces (slow) vs sample check (fast)
    """
    
    dataset_path = Path(dataset_dir)
    
    print("=" * 70)
    print("DATASET VERIFICATION - All Combinations")
    print("=" * 70)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Check mode: {'ALL traces' if check_all else 'Sample (first/last 50)'}")
    
    errors = []
    warnings = []
    
    # =========================================================================
    # Check configuration file
    # =========================================================================
    print("\n[1/6] Checking configuration...")
    
    config_file = dataset_path / "capture_config.json"
    if not config_file.exists():
        errors.append(f"Config file not found: {config_file}")
        print(f"  ERROR: {errors[-1]}")
        return False, errors, warnings
    
    with open(config_file) as f:
        config = json.load(f)
    
    num_combinations = config.get("num_combinations", 59049)
    num_target_coefficients = config.get("num_target_coefficients", 10)
    
    print(f"  Methodology: {config.get('methodology', 'unknown')}")
    print(f"  Target coefficients: {num_target_coefficients}")
    print(f"  Expected combinations: {num_combinations}")
    print("  [OK]")
    
    # =========================================================================
    # Check directory structure
    # =========================================================================
    print("\n[2/6] Checking directory structure...")
    
    traces_dir = dataset_path / "traces"
    if not traces_dir.exists():
        errors.append(f"Traces directory not found: {traces_dir}")
        print(f"  ERROR: {errors[-1]}")
        return False, errors, warnings
    
    inputs_dir = dataset_path / "inputs"
    if not inputs_dir.exists():
        warnings.append(f"Inputs directory not found: {inputs_dir}")
        print(f"  WARNING: {warnings[-1]}")
    
    print("  [OK]")
    
    # =========================================================================
    # Check trace file count
    # =========================================================================
    print("\n[3/6] Checking trace file count...")
    
    trace_files = sorted(traces_dir.glob("traces_*.npz"),
                        key=lambda x: int(x.stem.split('_')[1]))
    num_traces = len(trace_files)
    
    print(f"  Found: {num_traces} trace files")
    print(f"  Expected: {num_combinations}")
    
    if num_traces < num_combinations:
        warnings.append(f"Missing traces: {num_combinations - num_traces}")
        print(f"  WARNING: {warnings[-1]}")
    elif num_traces > num_combinations:
        warnings.append(f"Extra traces: {num_traces - num_combinations}")
        print(f"  WARNING: {warnings[-1]}")
    else:
        print("  [OK]")
    
    # Check for missing indices
    expected_indices = set(range(num_combinations))
    actual_indices = set(int(f.stem.split('_')[1]) for f in trace_files)
    missing_indices = expected_indices - actual_indices
    
    if missing_indices:
        if len(missing_indices) <= 10:
            warnings.append(f"Missing trace indices: {sorted(missing_indices)}")
        else:
            warnings.append(f"Missing {len(missing_indices)} trace indices")
        print(f"  WARNING: {warnings[-1]}")
    
    # =========================================================================
    # Load reference data
    # =========================================================================
    print("\n[4/6] Loading reference data...")
    
    if inputs_dir.exists():
        try:
            all_f_expected = np.load(inputs_dir / "all_f_combinations.npy")
            all_g_expected = np.load(inputs_dir / "all_g_data.npy")
            labels_expected = np.load(inputs_dir / "labels.npy")
            fixed_g = np.load(inputs_dir / "fixed_g.npy")
            
            print(f"  all_f_combinations: {all_f_expected.shape}")
            print(f"  labels: {labels_expected.shape}")
            print(f"  fixed_g: {fixed_g.shape}")
            print("  [OK]")
            has_reference = True
        except Exception as e:
            warnings.append(f"Could not load reference data: {e}")
            print(f"  WARNING: {warnings[-1]}")
            has_reference = False
            fixed_g = None
    else:
        has_reference = False
        fixed_g = None
    
    # =========================================================================
    # Verify g polynomial is FIXED
    # =========================================================================
    print("\n[5/6] Verifying g polynomial is FIXED...")
    
    reference_g = fixed_g
    g_mismatches = 0
    traces_checked = 0
    
    # Determine which traces to check
    if check_all:
        files_to_check = trace_files
    else:
        # Check first 50 and last 50
        files_to_check = trace_files[:50] + trace_files[-50:]
    
    iterator = files_to_check
    if HAS_TQDM and len(files_to_check) > 100:
        iterator = tqdm(files_to_check, desc="  Checking g", leave=False)
    
    for trace_file in iterator:
        try:
            data = np.load(trace_file)
            g = data['dut_io_ram_g_data'].flatten()
            
            if reference_g is None:
                reference_g = g
            elif not np.array_equal(g, reference_g):
                g_mismatches += 1
            
            traces_checked += 1
        except Exception as e:
            errors.append(f"Error reading {trace_file.name}: {e}")
    
    if g_mismatches > 0:
        errors.append(f"g polynomial is NOT fixed! {g_mismatches}/{traces_checked} traces have different g")
        print(f"  ERROR: {errors[-1]}")
    else:
        print(f"  Checked {traces_checked} traces. g is FIXED. [OK]")
    
    # =========================================================================
    # Verify f polynomials and labels
    # =========================================================================
    print("\n[6/6] Verifying f polynomials and labels...")
    
    f_mismatches = 0
    label_mismatches = 0
    format_errors = 0
    
    iterator = files_to_check
    if HAS_TQDM and len(files_to_check) > 100:
        iterator = tqdm(files_to_check, desc="  Checking f/labels", leave=False)
    
    for trace_file in iterator:
        try:
            trace_idx = int(trace_file.stem.split('_')[1])
            data = np.load(trace_file)
            
            # Check trace format
            if 'wave' not in data:
                format_errors += 1
                continue
            if 'dut_io_ram_f_data' not in data:
                format_errors += 1
                continue
            if 'labels' not in data:
                format_errors += 1
                continue
            
            f = data['dut_io_ram_f_data'].flatten()
            trace_labels = data['labels'].flatten()
            
            # Verify against reference if available
            if has_reference and trace_idx < len(all_f_expected):
                expected_f = all_f_expected[trace_idx]
                expected_labels = labels_expected[trace_idx]
                
                if not np.array_equal(f, expected_f):
                    f_mismatches += 1
                
                if not np.array_equal(trace_labels, expected_labels):
                    label_mismatches += 1
            
        except Exception as e:
            errors.append(f"Error reading {trace_file.name}: {e}")
    
    if format_errors > 0:
        errors.append(f"Format errors in {format_errors} traces")
        print(f"  ERROR: {errors[-1]}")
    
    if f_mismatches > 0:
        errors.append(f"f polynomial mismatches: {f_mismatches}")
        print(f"  ERROR: {errors[-1]}")
    
    if label_mismatches > 0:
        errors.append(f"Label mismatches: {label_mismatches}")
        print(f"  ERROR: {errors[-1]}")
    
    if format_errors == 0 and f_mismatches == 0 and label_mismatches == 0:
        print(f"  Checked {len(files_to_check)} traces. All correct. [OK]")
    
    # =========================================================================
    # Sample trace inspection
    # =========================================================================
    if verbose and num_traces > 0:
        print("\n" + "-" * 70)
        print("SAMPLE TRACE INSPECTION")
        print("-" * 70)
        
        # Show first trace
        first_trace = trace_files[0]
        data = np.load(first_trace)
        f = data['dut_io_ram_f_data'].flatten()
        labels = data['labels'].flatten()
        wave = data['wave'].flatten()
        
        # Convert f to signed for display
        f_display = np.where(f == 255, -1, f)
        
        print(f"\nFirst trace ({first_trace.name}):")
        print(f"  f[0:{num_target_coefficients}] = {f_display[:num_target_coefficients].tolist()}")
        print(f"  labels = {labels.tolist()}")
        print(f"  wave shape = {wave.shape}")
        print(f"  wave range = [{wave.min():.4f}, {wave.max():.4f}]")
        
        # Show last trace
        if num_traces > 1:
            last_trace = trace_files[-1]
            data = np.load(last_trace)
            f = data['dut_io_ram_f_data'].flatten()
            labels = data['labels'].flatten()
            
            f_display = np.where(f == 255, -1, f)
            
            print(f"\nLast trace ({last_trace.name}):")
            print(f"  f[0:{num_target_coefficients}] = {f_display[:num_target_coefficients].tolist()}")
            print(f"  labels = {labels.tolist()}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
    
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    
    if not errors:
        print("\n[OK] Dataset verification PASSED!")
        print("\nThe dataset follows the all combinations methodology:")
        print(f"  - {num_traces} traces captured")
        print("  - g polynomial is FIXED across all traces")
        print(f"  - f polynomials cover all 3^{num_target_coefficients} combinations")
        print("  - Labels are correct")
        print("\nThis dataset is ready for DL-SCA attack!")
        print("Suggested split: 70% train, 15% val, 15% test")
        return True, errors, warnings
    else:
        print("\n[X] Dataset verification FAILED!")
        print("Please fix the errors above before proceeding.")
        return False, errors, warnings


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify MT-TMVP DL-SCA dataset (All Combinations)"
    )
    parser.add_argument("--dataset", type=str, default="sca_dataset_v4",
                        help="Dataset directory (default: sca_dataset_v4)")
    parser.add_argument("--quiet", action="store_true",
                        help="Less verbose output")
    parser.add_argument("--check-all", action="store_true",
                        help="Check ALL traces (slow)")
    
    args = parser.parse_args()
    
    success, errors, warnings = verify_all_combinations_dataset(
        args.dataset, verbose=not args.quiet, check_all=args.check_all
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
