"""Generate input data for MT-TMVP side-channel attack.

Produces all 3^10 = 59,049 ternary coefficient combinations for f[0:10],
a fixed g polynomial, and expected TMVP outputs for hardware verification.
"""

import os
import numpy as np
import json
from datetime import datetime
from itertools import product
from typing import List, Tuple

# TMVP parameters
REAL_N = 509  # Number of coefficients
TERNARY_VALUES = [-1, 0, 1]  # Actual values
TERNARY_UNSIGNED = {-1: 255, 0: 0, 1: 1}  # Unsigned representation


def generate_g_polynomial(n: int = REAL_N, seed: int = None) -> np.ndarray:
    """Generate a G polynomial with byte values [0, 255]."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.randint(0, 256, size=n).astype(np.int64)


def generate_ternary_polynomial(n: int = REAL_N, seed: int = None) -> np.ndarray:
    """Generate a ternary polynomial with values {0, 1, 255}."""
    if seed is not None:
        np.random.seed(seed)
    values = [0, 1, 255]
    return np.random.choice(values, size=n).astype(np.int64)


def compute_tmvp_reference(f: List[int], g: List[int]) -> List[int]:
    """Compute TMVP result in software for verification."""
    n = len(f)
    result = []
    for i in range(n):
        acc = 0
        for j in range(n):
            idx = (i - j) % n
            if i < j:
                idx = n - (j - i)
            acc += f[idx] * g[j]
        result.append(acc & 0xFF)
    return result


def generate_all_combinations(num_coefficients: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate all possible combinations of f[0:num_coefficients].
    
    Returns:
        combinations: (3^num_coefficients, num_coefficients) array of coefficient values
        labels: Same as combinations (used for training labels per coefficient)
    """
    # Generate all combinations using itertools.product
    all_combos = list(product(TERNARY_VALUES, repeat=num_coefficients))
    combinations = np.array(all_combos, dtype=np.int64)
    
    # Labels are the same as combinations (each column is labels for that coefficient)
    labels = combinations.copy()
    
    return combinations, labels


def generate_all_combinations_inputs(
    output_dir: str,
    num_target_coefficients: int = 10,
    seed: int = 42
):
    """
    Generate inputs for ALL COMBINATIONS methodology.
    
    This generates all 3^10 = 59049 possible f vector combinations.
    No separate attack set - the training script will split into train/val/test.
    
    Args:
        output_dir: Output directory for generated inputs
        num_target_coefficients: Number of coefficients to target (default: 8)
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    num_combinations = 3 ** num_target_coefficients
    
    print("=" * 70)
    print("MT-TMVP DL-SCA Input Generation - ALL COMBINATIONS")
    print("=" * 70)
    print(f"\nTarget coefficients: 0 to {num_target_coefficients - 1}")
    print(f"Values per coefficient: {TERNARY_VALUES}")
    print(f"Total combinations: 3^{num_target_coefficients} = {num_combinations}")
    print(f"\nPolynomial length: {REAL_N}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    # Create directories
    inputs_dir = os.path.join(output_dir, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    
    # =========================================================================
    # Generate FIXED g polynomial (used for ALL traces)
    # =========================================================================
    print("\n[1/4] Generating FIXED g polynomial...")
    
    fixed_g = generate_g_polynomial(REAL_N)
    np.save(os.path.join(inputs_dir, "fixed_g.npy"), fixed_g)
    print(f"  fixed_g[0:10] = {fixed_g[:10].tolist()}")
    
    # =========================================================================
    # Generate base f polynomial (non-targeted coefficients)
    # =========================================================================
    print("\n[2/4] Generating base f polynomial...")
    
    base_f = generate_ternary_polynomial(REAL_N)
    # Set targeted coefficients to 0 (will be overwritten per combination)
    for i in range(num_target_coefficients):
        base_f[i] = 0
    np.save(os.path.join(inputs_dir, "base_f.npy"), base_f)
    print(f"  base_f[0:10] = {base_f[:10].tolist()}")
    print(f"  (coefficients 0-{num_target_coefficients-1} set to 0 as placeholders)")
    
    # =========================================================================
    # Generate ALL combinations
    # =========================================================================
    print(f"\n[3/4] Generating all {num_combinations} f combinations...")
    
    combinations, labels = generate_all_combinations(num_target_coefficients)
    print(f"  Combinations shape: {combinations.shape}")
    print(f"  First 5 combinations:")
    for i in range(min(5, len(combinations))):
        print(f"    {i}: {combinations[i].tolist()}")
    print(f"  ...")
    print(f"  Last 5 combinations:")
    for i in range(max(0, len(combinations)-5), len(combinations)):
        print(f"    {i}: {combinations[i].tolist()}")
    
    # Create full f polynomials for all combinations
    all_f_data = np.zeros((num_combinations, REAL_N), dtype=np.int64)
    
    for i in range(num_combinations):
        # Start with base_f
        f = base_f.copy()
        # Set targeted coefficients from this combination
        for j in range(num_target_coefficients):
            # Convert to unsigned representation
            f[j] = TERNARY_UNSIGNED[combinations[i, j]]
        all_f_data[i] = f
    
    # Save all f combinations
    np.save(os.path.join(inputs_dir, "all_f_combinations.npy"), all_f_data)
    print(f"  Saved all_f_combinations.npy: {all_f_data.shape}")
    
    # Save labels (the actual coefficient values, not unsigned)
    np.save(os.path.join(inputs_dir, "labels.npy"), labels)
    print(f"  Saved labels.npy: {labels.shape}")
    
    # Also save g data (same for all traces)
    all_g_data = np.tile(fixed_g, (num_combinations, 1))
    np.save(os.path.join(inputs_dir, "all_g_data.npy"), all_g_data)
    print(f"  Saved all_g_data.npy: {all_g_data.shape}")
    
    # Compute expected outputs for verification
    print("\n  Computing expected TMVP outputs...")
    expected_outputs = np.zeros((num_combinations, REAL_N), dtype=np.int64)
    for i in range(num_combinations):
        expected_outputs[i] = compute_tmvp_reference(
            all_f_data[i].tolist(), fixed_g.tolist()
        )
        if (i + 1) % 1000 == 0:
            print(f"    Computed {i + 1}/{num_combinations}")
    
    np.save(os.path.join(inputs_dir, "expected_outputs.npy"), expected_outputs)
    print(f"  Saved expected_outputs.npy: {expected_outputs.shape}")
    
    # =========================================================================
    # Save configuration
    # =========================================================================
    print(f"\n[4/4] Saving configuration...")
    
    config = {
        "methodology": "all_combinations",
        "description": f"All 3^{num_target_coefficients} combinations of f[0:{num_target_coefficients}] coefficients",
        "num_target_coefficients": num_target_coefficients,
        "target_coefficients": list(range(num_target_coefficients)),
        "ternary_values": TERNARY_VALUES,
        "num_combinations": num_combinations,
        "polynomial_length": REAL_N,
        "seed": seed,
        "generation_timestamp": datetime.now().isoformat(),
        "fixed_g_first_10": fixed_g[:10].tolist(),
        "base_f_first_10": base_f[:10].tolist(),
        "suggested_split": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15
        },
        "label_encoding": {
            "per_coefficient": "labels[i, j] is the value of f[j] for combination i",
            "values": "-1, 0, 1",
            "for_classification": "add 1 to get class index (0, 1, 2)"
        }
    }
    
    config_file = os.path.join(inputs_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved to: {config_file}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("INPUT GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\nStructure created:")
    print(f"  {output_dir}/")
    print(f"  └── inputs/")
    print(f"      ├── fixed_g.npy              ({REAL_N},) - FIXED g polynomial")
    print(f"      ├── base_f.npy               ({REAL_N},) - base f polynomial")
    print(f"      ├── all_f_combinations.npy   ({num_combinations}, {REAL_N}) - all f vectors")
    print(f"      ├── all_g_data.npy           ({num_combinations}, {REAL_N}) - g for all traces")
    print(f"      ├── labels.npy               ({num_combinations}, {num_target_coefficients}) - labels")
    print(f"      ├── expected_outputs.npy     ({num_combinations}, {REAL_N}) - expected TMVP outputs")
    print(f"      └── config.json")
    
    print(f"\nTotal combinations: {num_combinations}")
    print(f"Suggested split: 70% train ({int(num_combinations*0.7)}), "
          f"15% val ({int(num_combinations*0.15)}), "
          f"15% test ({int(num_combinations*0.15)})")
    
    print("\n" + "=" * 70)
    print("NEXT STEP:")
    print("=" * 70)
    print("Run capture_traces.py to collect power traces for all combinations")
    print("=" * 70)
    
    return config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate inputs for MT-TMVP DL-SCA (All Combinations)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_inputs.py --output-dir dataset
  python generate_inputs.py --output-dir dataset --num-coefficients 8
        """
    )
    
    parser.add_argument("--output-dir", type=str, default="dataset",
                        help="Output directory (default: dataset)")
    parser.add_argument("--num-coefficients", type=int, default=10,
                        help="Number of target coefficients (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    generate_all_combinations_inputs(
        output_dir=args.output_dir,
        num_target_coefficients=args.num_coefficients,
        seed=args.seed
    )
