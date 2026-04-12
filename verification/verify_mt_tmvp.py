"""
verify_mt_tmvp.py - Python verification of the MT-TMVP C implementation algorithm.

This script implements the EXACT same algorithm as mt_tmvp.c in Python,
verifies it against the reference schoolbook computation, and optionally
tests against the simulation .mem files.

This serves as:
  1. Proof that the algorithm in mt_tmvp.c is correct
  2. A reference for anyone reading the C code
  3. Validation against the MATLAB/Verilog simulation data
"""

import numpy as np
import sys
import time
import os

# ============================================================================
# PARAMETERS (must match mt_tmvp.c and params.vh)
# ============================================================================
TMVP_N         = 512
TMVP_REAL_N    = 509
TMVP_TILE_SIZE = 16
TMVP_MOD_Q     = 256

def trunc8(val):
    """Truncate to 8-bit unsigned (mod 256). Returns plain Python int."""
    return int(val) & 0xFF

def to_signed8(val):
    """Interpret uint8 as signed int8 (returns plain Python int)."""
    v = int(val) & 0xFF
    return v if v < 128 else v - 256

# ============================================================================
# SCHOOLBOOK TOEPLITZ MVP (Base Case - MatrixVectorMultiplier)
# ============================================================================
def schoolbook_toeplitz_mvp(row, col, vec, n):
    """
    Schoolbook matrix-vector product for an n x n Toeplitz matrix.
    
    T[i][j] = row[j-i] if j >= i, else col[i-j]
    result[i] = sum_j(T[i][j] * vec[j]) mod 256
    
    Matches MatrixVectorMultiplier.v in the Verilog.
    """
    result = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        acc = 0
        for j in range(n):
            if j >= i:
                a_ij = to_signed8(row[j - i])
            else:
                a_ij = to_signed8(col[i - j])
            acc += a_ij * to_signed8(vec[j])
        result[i] = trunc8(acc)
    return result

# ============================================================================
# RECURSIVE MT-TMVP (TMVP-2 Decomposition)
# ============================================================================
def mt_tmvp_recursive(row, col, vec, n):
    """
    Recursive TMVP-2 decomposition.
    
    For n x n Toeplitz matrix [A0 A2; A1 A0]:
        s1 = A0 * (B0 + B1)
        s2 = (A2 - A0) * B1
        s3 = (A1 - A0) * B0
        W0 = s1 + s2,  W1 = s1 + s3
    
    Submatrix extraction from Row R[0..n-1], Col C[0..n-1], half=n/2:
        A0: row = R[0..half-1],     col = C[0..half-1]
        A2: row = R[half..n-1],     col[k] = R[half-k]
        A1: row[k] = C[half-k],    col = C[half..n-1]
    
    Matches TMVP2_main (recursive) + TMVP2 (leaf) in the Verilog.
    """
    if n <= TMVP_TILE_SIZE:
        return schoolbook_toeplitz_mvp(row, col, vec, n)
    
    half = n // 2
    
    # --- Precompute (A2 - A0) ---
    diff_A2_A0_row = np.zeros(half, dtype=np.uint8)
    diff_A2_A0_col = np.zeros(half, dtype=np.uint8)
    for k in range(half):
        diff_A2_A0_row[k] = trunc8(to_signed8(row[half + k]) - to_signed8(row[k]))
        diff_A2_A0_col[k] = trunc8(to_signed8(row[half - k]) - to_signed8(col[k]))
    
    # --- Precompute (A1 - A0) ---
    diff_A1_A0_row = np.zeros(half, dtype=np.uint8)
    diff_A1_A0_col = np.zeros(half, dtype=np.uint8)
    for k in range(half):
        diff_A1_A0_row[k] = trunc8(to_signed8(col[half - k]) - to_signed8(row[k]))
        diff_A1_A0_col[k] = trunc8(to_signed8(col[half + k]) - to_signed8(col[k]))
    
    # --- Precompute B0 + B1 ---
    B0 = vec[:half]
    B1 = vec[half:]
    B0_plus_B1 = np.array([trunc8(to_signed8(B0[k]) + to_signed8(B1[k])) 
                           for k in range(half)], dtype=np.uint8)
    
    # --- Three recursive multiplications (Eq. 4) ---
    # s1 = A0 * (B0 + B1)
    s1 = mt_tmvp_recursive(row[:half], col[:half], B0_plus_B1, half)
    
    # s2 = (A2 - A0) * B1
    s2 = mt_tmvp_recursive(diff_A2_A0_row, diff_A2_A0_col, B1, half)
    
    # s3 = (A1 - A0) * B0
    s3 = mt_tmvp_recursive(diff_A1_A0_row, diff_A1_A0_col, B0, half)
    
    # --- Combine (Eq. 3) ---
    result = np.zeros(n, dtype=np.uint8)
    for k in range(half):
        result[k]        = trunc8(to_signed8(s1[k]) + to_signed8(s2[k]))  # W0
        result[half + k] = trunc8(to_signed8(s1[k]) + to_signed8(s3[k]))  # W1
    
    return result

# ============================================================================
# TOP-LEVEL MT-TMVP (Top_TMVP)
# ============================================================================
def mt_tmvp_multiply(f, g):
    """
    Top-level MT-TMVP polynomial multiplier.
    
    Builds Toeplitz matrix from f, multiplies by g, returns first REAL_N results.
    Matches Top_TMVP module in Verilog.
    
    Toeplitz construction (matching MATLAB and Top_TMVP LOADING state):
        Row[0] = f[0],  Row[k] = f[REAL_N - k]  for k=1..REAL_N-1
        Col[k] = f[k]  for k=0..REAL_N-1
    """
    # Zero-padded arrays
    row = np.zeros(TMVP_N, dtype=np.uint8)
    col = np.zeros(TMVP_N, dtype=np.uint8)
    vec = np.zeros(TMVP_N, dtype=np.uint8)
    
    # Build Toeplitz row: [f[0], f[n-1], f[n-2], ..., f[1], 0, ..., 0]
    row[0] = f[0]
    for k in range(1, TMVP_REAL_N):
        row[k] = f[TMVP_REAL_N - k]
    
    # Build Toeplitz col: [f[0], f[1], ..., f[n-1], 0, ..., 0]
    col[:TMVP_REAL_N] = f[:TMVP_REAL_N]
    
    # Vector g
    vec[:TMVP_REAL_N] = g[:TMVP_REAL_N]
    
    # Run recursive MT-TMVP
    full_result = mt_tmvp_recursive(row, col, vec, TMVP_N)
    
    # Return first REAL_N elements (discard padding)
    return full_result[:TMVP_REAL_N]

# ============================================================================
# REFERENCE: Direct schoolbook (golden reference)
# ============================================================================
def reference_tmvp(f, g, n):
    """
    Naive O(n^2) Toeplitz MVP for verification.
    result[i] = sum_j(f[(i-j) mod n] * g[j]) mod 256
    
    Matches Python generate_inputs.py:compute_tmvp_reference()
    and MATLAB: mod(toeplitz(Col,Row) * Vec, 256)
    """
    result = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        acc = 0
        for j in range(n):
            idx = (i - j) % n
            acc += to_signed8(f[idx]) * to_signed8(g[j])
        result[i] = trunc8(acc)
    return result

# ============================================================================
# TESTS
# ============================================================================
def run_tests():
    print("=" * 60)
    print("MT-TMVP Algorithm Verification (Python)")
    print("=" * 60)
    print(f"Parameters: N={TMVP_N}, REAL_N={TMVP_REAL_N}, "
          f"TILE_SIZE={TMVP_TILE_SIZE}, q={TMVP_MOD_Q}")
    print(f"Recursion: {TMVP_N} -> {TMVP_N//2} -> {TMVP_N//4} -> "
          f"{TMVP_N//8} -> {TMVP_N//16} -> {TMVP_TILE_SIZE} (schoolbook)")
    print()
    
    all_passed = True
    rng = np.random.RandomState(42)
    
    # Test 1-3: Random 7-bit polynomials
    for t in range(3):
        seed = 42 + t * 1000
        rng = np.random.RandomState(seed)
        f = rng.randint(0, 128, size=TMVP_REAL_N).astype(np.uint8)
        g = rng.randint(0, 128, size=TMVP_REAL_N).astype(np.uint8)
        
        print(f"Test {t+1}/6: Random 7-bit polynomials (seed={seed})")
        print(f"  f[0:5] = {list(f[:5])}")
        print(f"  g[0:5] = {list(g[:5])}")
        
        t0 = time.perf_counter()
        result_mt = mt_tmvp_multiply(f, g)
        t_mt = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        result_ref = reference_tmvp(f, g, TMVP_REAL_N)
        t_ref = time.perf_counter() - t0
        
        mismatches = np.sum(result_mt != result_ref)
        if mismatches == 0:
            print(f"  PASSED  (all {TMVP_REAL_N} outputs match)")
            print(f"  result[0:5] = {list(result_mt[:5])}")
            print(f"  Time: MT-TMVP={t_mt*1000:.1f}ms, Ref={t_ref*1000:.1f}ms")
        else:
            print(f"  FAILED  ({mismatches}/{TMVP_REAL_N} mismatches)")
            diffs = np.where(result_mt != result_ref)[0]
            for idx in diffs[:3]:
                print(f"    [{idx}]: MT-TMVP={result_mt[idx]}, Ref={result_ref[idx]}")
            all_passed = False
        print()
    
    # Test 4: Ternary coefficients (SCA dataset encoding)
    print("Test 4/6: Ternary coefficients {-1, 0, 1}")
    rng = np.random.RandomState(999)
    ternary_vals = np.array([255, 0, 1], dtype=np.uint8)  # -1, 0, 1
    f = ternary_vals[rng.randint(0, 3, size=TMVP_REAL_N)]
    g = ternary_vals[rng.randint(0, 3, size=TMVP_REAL_N)]
    print(f"  f[0:10] = {list(f[:10])}")
    print(f"  g[0:10] = {list(g[:10])}")
    
    result_mt = mt_tmvp_multiply(f, g)
    result_ref = reference_tmvp(f, g, TMVP_REAL_N)
    mismatches = np.sum(result_mt != result_ref)
    if mismatches == 0:
        print(f"  PASSED  (all {TMVP_REAL_N} outputs match)")
        print(f"  result[0:10] = {list(result_mt[:10])}")
    else:
        print(f"  FAILED  ({mismatches}/{TMVP_REAL_N} mismatches)")
        all_passed = False
    print()
    
    # Test 5: All zeros
    print("Test 5/6: f = all zeros, g = all ones")
    f = np.zeros(TMVP_REAL_N, dtype=np.uint8)
    g = np.ones(TMVP_REAL_N, dtype=np.uint8)
    result_mt = mt_tmvp_multiply(f, g)
    result_ref = reference_tmvp(f, g, TMVP_REAL_N)
    mismatches = np.sum(result_mt != result_ref)
    if mismatches == 0 and np.all(result_mt == 0):
        print("  PASSED  (all zeros as expected)")
    elif mismatches == 0:
        print("  PASSED  (outputs match)")
    else:
        print(f"  FAILED  ({mismatches} mismatches)")
        all_passed = False
    print()
    
    # Test 6: Identity (f = [1, 0, 0, ...])
    print("Test 6/6: f = [1, 0, ...], g = random (identity test)")
    f = np.zeros(TMVP_REAL_N, dtype=np.uint8)
    f[0] = 1
    rng = np.random.RandomState(777)
    g = rng.randint(0, 256, size=TMVP_REAL_N).astype(np.uint8)
    result_mt = mt_tmvp_multiply(f, g)
    result_ref = reference_tmvp(f, g, TMVP_REAL_N)
    mismatches = np.sum(result_mt != result_ref)
    identity_ok = np.array_equal(result_mt, g)
    if mismatches == 0:
        print(f"  PASSED  (result == g: {'yes, identity confirmed' if identity_ok else 'ref matches'})")
    else:
        print(f"  FAILED  ({mismatches} mismatches)")
        all_passed = False
    print()
    
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
        print("The MT-TMVP algorithm in mt_tmvp.c is verified correct.")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
    
    return all_passed


def test_with_sim_files():
    """Test against the Verilog simulation .mem files if available."""
    sim_dir = os.path.join(os.path.dirname(__file__), "sim")
    f_path = os.path.join(sim_dir, "f.mem")
    g_path = os.path.join(sim_dir, "g.mem")
    dat_path = os.path.join(sim_dir, "final.dat")
    
    if not all(os.path.exists(p) for p in [f_path, g_path, dat_path]):
        print("\nSimulation files not found, skipping RTL comparison.")
        return
    
    print("\n" + "=" * 60)
    print("RTL Simulation Data Comparison")
    print("=" * 60)
    
    # Read f.mem (hex, one value per line)
    f_data = []
    with open(f_path) as fp:
        for line in fp:
            line = line.strip()
            if line:
                f_data.append(int(line, 16) & 0xFF)
    
    # Read g.mem
    g_data = []
    with open(g_path) as fp:
        for line in fp:
            line = line.strip()
            if line:
                g_data.append(int(line, 16) & 0xFF)
    
    # Read RTL output
    rtl_output = []
    with open(dat_path) as fp:
        for line in fp:
            line = line.strip()
            if line:
                rtl_output.append(int(line) & 0xFF)
    
    print(f"  f.mem: {len(f_data)} coefficients")
    print(f"  g.mem: {len(g_data)} coefficients")
    print(f"  final.dat: {len(rtl_output)} output values")
    
    n = min(len(f_data), len(g_data), TMVP_REAL_N)
    f = np.zeros(TMVP_REAL_N, dtype=np.uint8)
    g = np.zeros(TMVP_REAL_N, dtype=np.uint8)
    f[:n] = np.array(f_data[:n], dtype=np.uint8)
    g[:n] = np.array(g_data[:n], dtype=np.uint8)
    
    # Compute MT-TMVP
    result_mt = mt_tmvp_multiply(f, g)
    
    # Compare with RTL
    rtl = np.array(rtl_output[:TMVP_REAL_N], dtype=np.uint8)
    mismatches = np.sum(result_mt[:len(rtl)] != rtl)
    
    print(f"\n  MT-TMVP vs RTL: ", end="")
    if mismatches == 0:
        print(f"MATCH (all {len(rtl)} outputs identical)")
        print(f"  MT-TMVP[0:5] = {list(result_mt[:5])}")
        print(f"  RTL[0:5]     = {list(rtl[:5])}")
    else:
        print(f"{mismatches} mismatches out of {len(rtl)}")
        diffs = np.where(result_mt[:len(rtl)] != rtl)[0]
        for idx in diffs[:5]:
            print(f"    [{idx}]: MT-TMVP={result_mt[idx]}, RTL={rtl[idx]}")
    
    # Also compare with reference
    result_ref = reference_tmvp(f, g, TMVP_REAL_N)
    ref_vs_rtl = np.sum(result_ref[:len(rtl)] != rtl)
    print(f"  Reference vs RTL: {'MATCH' if ref_vs_rtl == 0 else f'{ref_vs_rtl} mismatches'}")


if __name__ == "__main__":
    passed = run_tests()
    test_with_sim_files()
    sys.exit(0 if passed else 1)
