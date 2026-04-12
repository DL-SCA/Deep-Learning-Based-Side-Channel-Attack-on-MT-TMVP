/*
 * mt_tmvp.c - C Implementation of MT-TMVP Polynomial Multiplier
 *
 * Implements the Modular Tiled TMVP-based Polynomial Multiplication algorithm
 
 *   Top_TMVP  -->  TMVP2_main (recursive)  -->  TMVP2  -->  MatrixVectorMultiplier
 *
 * Parameters (for NTRU N=509):
 *   N         = 512  (padded dimension, power-of-2 multiple of TILE_SIZE)
 *   REAL_N    = 509  (actual polynomial degree)
 *   TILE_SIZE = 16   (base tile for schoolbook multiplication)
 *   q         = 256  (modulus, implicit via 8-bit truncation)
 *
 * Recursion levels (N=512):  512 -> 256 -> 128 -> 64 -> 32 -> 16 (schoolbook)
 *   = 5 levels of TMVP-2, each using 3 sub-multiplications instead of 4 (25% reduction)
 *   Total base multiplications: 3^5 = 243  (vs. naive 4^5 = 1024)
 *
 * Build:
 *   gcc -O2 -o mt_tmvp mt_tmvp.c
 *   cl /O2 mt_tmvp.c          (MSVC)
 *
 * Usage:
 *   ./mt_tmvp                  Run verification tests
 *   ./mt_tmvp --benchmark      Run performance benchmark
 *   ./mt_tmvp --file f.hex g.hex   Compute from hex input files
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* ============================================================================
 * CONFIGURABLE PARAMETERS
 *
 * These match the Verilog params.vh and Top_TMVP module parameters.
 * Adjust for other polynomial dimensions (e.g., N=720 for NTRU-677).
 * ============================================================================ */
#define TMVP_N          512
#define TMVP_REAL_N     509
#define TMVP_TILE_SIZE  16
#define TMVP_DATA_WIDTH 8
#define TMVP_MOD_Q      (1 << TMVP_DATA_WIDTH)  /* 256 */

/* Truncate to 8-bit unsigned (mod 256).
 * In two's complement, (val & 0xFF) is identical for signed/unsigned mod 256.
 * This matches the Verilog register truncation at every pipeline stage. */
static inline uint8_t trunc8(int val)
{
    return (uint8_t)(val & 0xFF);
}

/* ============================================================================
 * SCHOOLBOOK TOEPLITZ MATRIX-VECTOR PRODUCT  (Base Case)
 * ============================================================================
 *
 * Corresponds to: MatrixVectorMultiplier module in the Verilog.
 *
 * Computes W = T * B where T is an n x n Toeplitz matrix defined by:
 *   T[i][j] = row[j - i]   if j >= i
 *   T[i][j] = col[i - j]   if i >  j
 *
 * The hardware implements this with:
 *   1. Shift-register row loading (first row, then shift + new column element)
 *   2. Parallel element-wise multiply:  mult[k] = row[k] * vec[k]
 *   3. Pipelined adder tree (log2(n) stages) to sum products
 *   4. One result element per clock after initial pipeline fill
 *
 * In software, we compute this directly with O(n^2) multiply-accumulate.
 * All arithmetic is mod 256 (8-bit signed, matching DATA_WIDTH=8).
 *
 * Parameters:
 *   row[0..n-1]    : first row of the Toeplitz matrix
 *   col[0..n-1]    : first column (col[0] == row[0], the diagonal element)
 *   vec[0..n-1]    : input vector B
 *   result[0..n-1] : output vector W (caller-allocated)
 *   n              : dimension (typically TILE_SIZE = 16)
 */
static void schoolbook_toeplitz_mvp(
    const uint8_t *row, const uint8_t *col, const uint8_t *vec,
    uint8_t *result, int n)
{
    int i, j;
    for (i = 0; i < n; i++) {
        int acc = 0;
        for (j = 0; j < n; j++) {
            /* Toeplitz element A[i][j]:
             *   j >= i  =>  first-row element row[j-i]
             *   j <  i  =>  first-col element col[i-j]
             *
             * Cast to int8_t for signed multiplication (matching Verilog
             * `signed [DATA_WIDTH-1:0]` operands). The final mod-256 result
             * is the same regardless of signed/unsigned interpretation. */
            int a_ij = (j >= i) ? (int)(int8_t)row[j - i]
                                : (int)(int8_t)col[i - j];
            acc += a_ij * (int)(int8_t)vec[j];
        }
        result[i] = trunc8(acc);
    }
}

/* ============================================================================
 * RECURSIVE MT-TMVP  (TMVP-2 Decomposition)
 * ============================================================================
 *
 * Corresponds to: TMVP2_main (recursive tree) + TMVP2 (leaf) modules.
 *
 * For an n x n Toeplitz matrix T partitioned into (n/2 x n/2) blocks:
 *
 *   T = [A0  A2]    B = [B0]    W = [W0] = [s1 + s2]
 *       [A1  A0]        [B1]        [W1]   [s1 + s3]
 *
 *   s1 = A0 * (B0 + B1)          (Eq. 4 from paper)
 *   s2 = (A2 - A0) * B1
 *   s3 = (A1 - A0) * B0
 *
 * Submatrix extraction from parent first-row R[0..n-1] and first-col C[0..n-1]:
 *   A0:  row = R[0 .. half-1]        col = C[0 .. half-1]
 *   A2:  row = R[half .. n-1]        col[k] = R[half - k]
 *   A1:  row[k] = C[half - k]        col = C[half .. n-1]
 *
 * Proof of submatrix formulas:
 *   A0[i][j] = T[i][j]             => A0_row[k] = R[k],      A0_col[k] = C[k]
 *   A2[i][j] = T[i][j+half]        => A2_row[k] = R[k+half], A2_col[k] = R[half-k]
 *   A1[i][j] = T[i+half][j]        => A1_row[k] = C[half-k], A1_col[k] = C[k+half]
 *
 * These are all valid Toeplitz matrices (difference of Toeplitz is Toeplitz).
 *
 * Recursion terminates at n = TILE_SIZE with schoolbook multiplication.
 * For N=512, TILE_SIZE=16:  512 -> 256 -> 128 -> 64 -> 32 -> 16 (5 levels).
 */
static void mt_tmvp_recursive(
    const uint8_t *row, const uint8_t *col, const uint8_t *vec,
    uint8_t *result, int n)
{
    int k;
    int half;
    uint8_t *diff_A2_A0_row, *diff_A2_A0_col;
    uint8_t *diff_A1_A0_row, *diff_A1_A0_col;
    uint8_t *B0_plus_B1;
    uint8_t *s1, *s2, *s3;
    const uint8_t *B0, *B1;

    /* Base case: schoolbook at tile size (MatrixVectorMultiplier in HW) */
    if (n <= TMVP_TILE_SIZE) {
        schoolbook_toeplitz_mvp(row, col, vec, result, n);
        return;
    }

    half = n / 2;

    /* Allocate temporaries for this recursion level.
     * Stack usage per level: 8 * half bytes.
     * Total across 5 levels: 8*(256+128+64+32+16) = 3968 bytes. */
    diff_A2_A0_row = (uint8_t *)malloc(half);
    diff_A2_A0_col = (uint8_t *)malloc(half);
    diff_A1_A0_row = (uint8_t *)malloc(half);
    diff_A1_A0_col = (uint8_t *)malloc(half);
    B0_plus_B1     = (uint8_t *)malloc(half);
    s1             = (uint8_t *)malloc(half);
    s2             = (uint8_t *)malloc(half);
    s3             = (uint8_t *)malloc(half);

    /* ------------------------------------------------------------------
     * Precompute (A2 - A0) row and col
     *
     * Matches TMVP2_main FIRST_LOAD phase:
     *   ROW_RAM[N+k] = data_row_2 - data_row_1  (A2_row - A0_row)
     *   COL_RAM[2N-k] = data_row_1 - data_col_1  (which is R[k] - C[N-k])
     *     => (A2-A0)_col[m] = R[half-m] - C[m]
     * ------------------------------------------------------------------ */
    for (k = 0; k < half; k++) {
        /* (A2-A0)_row[k] = R[half+k] - R[k] */
        diff_A2_A0_row[k] = trunc8((int)(int8_t)row[half + k] -
                                    (int)(int8_t)row[k]);
        /* (A2-A0)_col[k] = R[half-k] - C[k] */
        diff_A2_A0_col[k] = trunc8((int)(int8_t)row[half - k] -
                                    (int)(int8_t)col[k]);
    }

    /* ------------------------------------------------------------------
     * Precompute (A1 - A0) row and col
     *
     * Matches TMVP2_main LAST_LOAD phase:
     *   Loads A1 data and computes (A1 - A0) for the third multiplication.
     * ------------------------------------------------------------------ */
    for (k = 0; k < half; k++) {
        /* (A1-A0)_row[k] = C[half-k] - R[k] */
        diff_A1_A0_row[k] = trunc8((int)(int8_t)col[half - k] -
                                    (int)(int8_t)row[k]);
        /* (A1-A0)_col[k] = C[half+k] - C[k] */
        diff_A1_A0_col[k] = trunc8((int)(int8_t)col[half + k] -
                                    (int)(int8_t)col[k]);
    }

    /* ------------------------------------------------------------------
     * Precompute B0 + B1
     *
     * Matches TMVP2_main VEC RAM storage during FIRST_LOAD:
     *   VEC_data_in[0] = data_vec_data_1 + data_vec_data_2  (B0 + B1)
     * ------------------------------------------------------------------ */
    B0 = vec;
    B1 = vec + half;
    for (k = 0; k < half; k++) {
        B0_plus_B1[k] = trunc8((int)(int8_t)B0[k] + (int)(int8_t)B1[k]);
    }

    /* ------------------------------------------------------------------
     * Three recursive multiplications (Eq. 4)
     *
     * In hardware, these execute sequentially in the TMVP2_main FSM:
     *   STEP_1: s1 = A0 * (B0 + B1)     -> stored in RESULT_RAM
     *   STEP_2: s2 = (A2-A0) * B1        -> output = s2 + RESULT_RAM = W0
     *   STEP_3: s3 = (A1-A0) * B0        -> output = s3 + RESULT_RAM = W1
     * ------------------------------------------------------------------ */

    /* s1 = A0 * (B0 + B1)
     * A0: row = R[0..half-1], col = C[0..half-1] (first halves of parent) */
    mt_tmvp_recursive(row, col, B0_plus_B1, s1, half);

    /* s2 = (A2 - A0) * B1 */
    mt_tmvp_recursive(diff_A2_A0_row, diff_A2_A0_col, B1, s2, half);

    /* s3 = (A1 - A0) * B0 */
    mt_tmvp_recursive(diff_A1_A0_row, diff_A1_A0_col, B0, s3, half);

    /* ------------------------------------------------------------------
     * Combine results (Eq. 3)
     *
     * In hardware, this is the RESULT phase of TMVP2_main:
     *   Output during STEP_2: m_axis_tdata = TMVP_result + RESULT_RAM = s2+s1
     *   Output during STEP_3: m_axis_tdata = TMVP_result + RESULT_RAM = s3+s1
     * ------------------------------------------------------------------ */
    for (k = 0; k < half; k++) {
        /* W0[k] = s1[k] + s2[k]  (upper half of result) */
        result[k]        = trunc8((int)(int8_t)s1[k] + (int)(int8_t)s2[k]);
        /* W1[k] = s1[k] + s3[k]  (lower half of result) */
        result[half + k] = trunc8((int)(int8_t)s1[k] + (int)(int8_t)s3[k]);
    }

    free(diff_A2_A0_row);
    free(diff_A2_A0_col);
    free(diff_A1_A0_row);
    free(diff_A1_A0_col);
    free(B0_plus_B1);
    free(s1);
    free(s2);
    free(s3);
}

/* ============================================================================
 * TOP-LEVEL MT-TMVP POLYNOMIAL MULTIPLIER
 * ============================================================================
 *
 * Corresponds to: Top_TMVP module in the Verilog.
 *
 * Computes: result = T(f) * g  in Z_256[x] / (x^n - 1)
 *
 * Where T(f) is the Toeplitz matrix constructed from polynomial f such that:
 *   T(f)[i][j] = f[(i - j) mod n]
 *
 * This matches:
 *   - MATLAB:  A = toeplitz(Col, Row);  c = mod(A * g, 256);
 *   - Python:  result[i] = sum_j(f[(i-j) % n] * g[j]) & 0xFF
 *
 * Steps (matching Top_TMVP FSM):
 *   1. LOADING: Build Toeplitz Row/Col from f, store g as Vec, zero-pad to N
 *   2. BUSY:    Run recursive MT-TMVP multiplication
 *   3. Output:  Extract first REAL_N results (discard zero-padding)
 *
 * Parameters:
 *   f[0..REAL_N-1]      : first polynomial (uint8_t, unsigned 8-bit encoding)
 *   g[0..REAL_N-1]      : second polynomial (uint8_t)
 *   result[0..REAL_N-1]  : output (caller-allocated)
 */
void mt_tmvp_multiply(const uint8_t *f, const uint8_t *g, uint8_t *result)
{
    int k;

    /* Allocate zero-padded arrays (calloc zeros the padding automatically) */
    uint8_t *row         = (uint8_t *)calloc(TMVP_N, sizeof(uint8_t));
    uint8_t *col         = (uint8_t *)calloc(TMVP_N, sizeof(uint8_t));
    uint8_t *vec         = (uint8_t *)calloc(TMVP_N, sizeof(uint8_t));
    uint8_t *full_result = (uint8_t *)calloc(TMVP_N, sizeof(uint8_t));

    /* ------------------------------------------------------------------
     * Build Toeplitz matrix first-row and first-column from f.
     *
     * Matches Top_TMVP LOADING state:
     *   counter_in == 0:  address_row_A = 0           => Row[0] = f[0]
     *   counter_in >  0:  address_row_A = REAL_N - k  => Row[REAL_N-k] = f[k]
     *   address_col_A = counter_in                     => Col[k] = f[k]
     *
     * Result:
     *   Row[0] = f[0],  Row[k] = f[REAL_N - k]  for k = 1..REAL_N-1
     *   Col[k] = f[k]                            for k = 0..REAL_N-1
     *
     * This creates T such that T[i][j] = f[(i-j) mod REAL_N], matching
     * the MATLAB: Row = [f(1); flip(f(2:end))], Col = f
     * ------------------------------------------------------------------ */
    row[0] = f[0];
    for (k = 1; k < TMVP_REAL_N; k++) {
        row[k] = f[TMVP_REAL_N - k];
    }
    /* row[REAL_N .. N-1] = 0 (from calloc) */

    for (k = 0; k < TMVP_REAL_N; k++) {
        col[k] = f[k];
    }
    /* col[REAL_N .. N-1] = 0 */

    /* Vector g, zero-padded */
    for (k = 0; k < TMVP_REAL_N; k++) {
        vec[k] = g[k];
    }
    /* vec[REAL_N .. N-1] = 0 */

    /* Run the recursive MT-TMVP on the padded N x N problem */
    mt_tmvp_recursive(row, col, vec, full_result, TMVP_N);

    /* Extract first REAL_N results (discard padding elements).
     * Matches Top_TMVP "Ignoring" section:
     *   if (counter_out == REAL_N) m_axis_tvalid <= 0  */
    memcpy(result, full_result, TMVP_REAL_N);

    free(row);
    free(col);
    free(vec);
    free(full_result);
}

/* ============================================================================
 * REFERENCE: DIRECT SCHOOLBOOK TOEPLITZ MVP  (Golden Reference)
 * ============================================================================
 *
 * Naive O(n^2) computation for verification.
 * Matches the Python compute_tmvp_reference() and MATLAB toeplitz(Col,Row)*Vec.
 *
 *   result[i] = sum_j( f[(i-j) mod n] * g[j] ) mod 256
 */
void reference_tmvp(const uint8_t *f, const uint8_t *g, uint8_t *result, int n)
{
    int i, j;
    for (i = 0; i < n; i++) {
        int acc = 0;
        for (j = 0; j < n; j++) {
            int idx = (i >= j) ? (i - j) : (n - (j - i));
            acc += (int)(int8_t)f[idx] * (int)(int8_t)g[j];
        }
        result[i] = trunc8(acc);
    }
}

/* ============================================================================
 * SIMPLE PRNG (xorshift32, for reproducible tests)
 * ============================================================================ */
static uint32_t rng_state = 12345;

static uint32_t xorshift32(void)
{
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

/* ============================================================================
 * VERIFICATION TESTS
 * ============================================================================ */
static int run_tests(void)
{
    int t, i;
    int all_passed = 1;
    int num_random_tests = 5;

    uint8_t *f          = (uint8_t *)malloc(TMVP_REAL_N);
    uint8_t *g          = (uint8_t *)malloc(TMVP_REAL_N);
    uint8_t *result_mt  = (uint8_t *)malloc(TMVP_REAL_N);
    uint8_t *result_ref = (uint8_t *)malloc(TMVP_REAL_N);

    printf("============================================================\n");
    printf("MT-TMVP C Implementation - Verification Tests\n");
    printf("============================================================\n");
    printf("Parameters:\n");
    printf("  N (padded)  = %d\n", TMVP_N);
    printf("  REAL_N      = %d\n", TMVP_REAL_N);
    printf("  TILE_SIZE   = %d\n", TMVP_TILE_SIZE);
    printf("  DATA_WIDTH  = %d bits\n", TMVP_DATA_WIDTH);
    printf("  q (modulus) = %d\n", TMVP_MOD_Q);
    printf("  Recursion   : %d -> %d -> %d -> %d -> %d -> %d (schoolbook)\n",
           TMVP_N, TMVP_N/2, TMVP_N/4, TMVP_N/8, TMVP_N/16, TMVP_TILE_SIZE);
    printf("  Base mults  : 3^5 = 243 (vs. naive 1024)\n\n");

    /* --- Random polynomial tests --- */
    for (t = 0; t < num_random_tests; t++) {
        int mismatches = 0;

        rng_state = 42 + t * 1000;
        for (i = 0; i < TMVP_REAL_N; i++) {
            f[i] = (uint8_t)(xorshift32() & 0x7F);  /* 0..127, matching MATLAB randi */
            g[i] = (uint8_t)(xorshift32() & 0x7F);
        }

        printf("Test %d/%d: Random 7-bit polynomials (seed=%u)\n",
               t + 1, num_random_tests + 1, 42 + t * 1000);
        printf("  f[0:5] = [%3d, %3d, %3d, %3d, %3d]\n",
               f[0], f[1], f[2], f[3], f[4]);
        printf("  g[0:5] = [%3d, %3d, %3d, %3d, %3d]\n",
               g[0], g[1], g[2], g[3], g[4]);

        mt_tmvp_multiply(f, g, result_mt);
        reference_tmvp(f, g, result_ref, TMVP_REAL_N);

        for (i = 0; i < TMVP_REAL_N; i++) {
            if (result_mt[i] != result_ref[i]) {
                mismatches++;
                if (mismatches <= 3) {
                    printf("  MISMATCH [%d]: MT-TMVP=%u, Ref=%u\n",
                           i, result_mt[i], result_ref[i]);
                }
            }
        }

        if (mismatches == 0) {
            printf("  PASSED  (all %d outputs match)\n", TMVP_REAL_N);
            printf("  result[0:5] = [%3d, %3d, %3d, %3d, %3d]\n\n",
                   result_mt[0], result_mt[1], result_mt[2],
                   result_mt[3], result_mt[4]);
        } else {
            printf("  FAILED  (%d / %d mismatches)\n\n", mismatches, TMVP_REAL_N);
            all_passed = 0;
        }
    }

    /* --- Ternary coefficient test (SCA dataset encoding) --- */
    {
        int mismatches = 0;

        printf("Test %d/%d: Ternary coefficients {-1,0,1} (SCA dataset encoding)\n",
               num_random_tests + 1, num_random_tests + 1);
        printf("  Encoding: -1 -> 0xFF (255),  0 -> 0x00,  1 -> 0x01\n");

        rng_state = 999;
        for (i = 0; i < TMVP_REAL_N; i++) {
            int r = xorshift32() % 3;
            f[i] = (r == 0) ? 0xFF : (r == 1) ? 0x00 : 0x01;
            r = xorshift32() % 3;
            g[i] = (r == 0) ? 0xFF : (r == 1) ? 0x00 : 0x01;
        }

        printf("  f[0:10] = [");
        for (i = 0; i < 10; i++)
            printf("%3d%s", f[i], (i < 9) ? ", " : "");
        printf("]\n");
        printf("  g[0:10] = [");
        for (i = 0; i < 10; i++)
            printf("%3d%s", g[i], (i < 9) ? ", " : "");
        printf("]\n");

        mt_tmvp_multiply(f, g, result_mt);
        reference_tmvp(f, g, result_ref, TMVP_REAL_N);

        for (i = 0; i < TMVP_REAL_N; i++) {
            if (result_mt[i] != result_ref[i]) {
                mismatches++;
                if (mismatches <= 3) {
                    printf("  MISMATCH [%d]: MT-TMVP=%u, Ref=%u\n",
                           i, result_mt[i], result_ref[i]);
                }
            }
        }

        if (mismatches == 0) {
            printf("  PASSED  (all %d outputs match)\n", TMVP_REAL_N);
            printf("  result[0:10] = [");
            for (i = 0; i < 10; i++)
                printf("%3d%s", result_mt[i], (i < 9) ? ", " : "");
            printf("]\n");
        } else {
            printf("  FAILED  (%d / %d mismatches)\n", mismatches, TMVP_REAL_N);
            all_passed = 0;
        }
    }

    /* --- Edge case: all zeros --- */
    {
        int mismatches = 0;
        printf("\nEdge case: f = all zeros, g = all ones\n");
        memset(f, 0, TMVP_REAL_N);
        memset(g, 1, TMVP_REAL_N);
        mt_tmvp_multiply(f, g, result_mt);
        reference_tmvp(f, g, result_ref, TMVP_REAL_N);
        for (i = 0; i < TMVP_REAL_N; i++) {
            if (result_mt[i] != result_ref[i]) mismatches++;
        }
        if (mismatches == 0 && result_mt[0] == 0) {
            printf("  PASSED  (all zeros as expected)\n");
        } else if (mismatches == 0) {
            printf("  PASSED  (outputs match)\n");
        } else {
            printf("  FAILED  (%d mismatches)\n", mismatches);
            all_passed = 0;
        }
    }

    /* --- Edge case: identity-like --- */
    {
        int mismatches = 0;
        printf("\nEdge case: f = [1, 0, 0, ...], g = random\n");
        memset(f, 0, TMVP_REAL_N);
        f[0] = 1;
        rng_state = 777;
        for (i = 0; i < TMVP_REAL_N; i++)
            g[i] = (uint8_t)(xorshift32() & 0xFF);
        mt_tmvp_multiply(f, g, result_mt);
        reference_tmvp(f, g, result_ref, TMVP_REAL_N);
        for (i = 0; i < TMVP_REAL_N; i++) {
            if (result_mt[i] != result_ref[i]) mismatches++;
        }
        if (mismatches == 0) {
            /* When f = [1,0,...,0], T(f) = Identity, so result should equal g */
            int identity_check = 1;
            for (i = 0; i < TMVP_REAL_N; i++) {
                if (result_mt[i] != g[i]) { identity_check = 0; break; }
            }
            printf("  PASSED  (result == g: %s)\n",
                   identity_check ? "yes, identity confirmed" : "no, but ref matches");
        } else {
            printf("  FAILED  (%d mismatches)\n", mismatches);
            all_passed = 0;
        }
    }

    printf("\n============================================================\n");
    if (all_passed) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("SOME TESTS FAILED\n");
    }
    printf("============================================================\n");

    free(f);
    free(g);
    free(result_mt);
    free(result_ref);

    return all_passed ? 0 : 1;
}

/* ============================================================================
 * PERFORMANCE BENCHMARK
 * ============================================================================ */
static void run_benchmark(void)
{
    int i, iter;
    int iterations = 1000;
    clock_t start, end;
    double mt_time, ref_time;

    uint8_t *f      = (uint8_t *)malloc(TMVP_REAL_N);
    uint8_t *g      = (uint8_t *)malloc(TMVP_REAL_N);
    uint8_t *result = (uint8_t *)malloc(TMVP_REAL_N);

    printf("\n============================================================\n");
    printf("MT-TMVP Performance Benchmark (%d iterations)\n", iterations);
    printf("============================================================\n");

    rng_state = 42;
    for (i = 0; i < TMVP_REAL_N; i++) {
        f[i] = (uint8_t)(xorshift32() & 0x7F);
        g[i] = (uint8_t)(xorshift32() & 0x7F);
    }

    /* Benchmark MT-TMVP */
    start = clock();
    for (iter = 0; iter < iterations; iter++) {
        mt_tmvp_multiply(f, g, result);
    }
    end = clock();
    mt_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("MT-TMVP (recursive):  %.3f s total,  %.1f us/multiply\n",
           mt_time, mt_time / iterations * 1e6);

    /* Benchmark reference schoolbook */
    start = clock();
    for (iter = 0; iter < iterations; iter++) {
        reference_tmvp(f, g, result, TMVP_REAL_N);
    }
    end = clock();
    ref_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Reference (schoolbook): %.3f s total,  %.1f us/multiply\n",
           ref_time, ref_time / iterations * 1e6);

    if (mt_time > 0) {
        printf("Speedup: %.2fx\n", ref_time / mt_time);
    }

    free(f);
    free(g);
    free(result);
}

/* ============================================================================
 * FILE-BASED COMPUTATION
 *
 * Reads f and g from hex files (one coefficient per line, as generated by
 * the MATLAB verification script or generate_inputs.py).
 * ============================================================================ */
static int run_from_files(const char *f_path, const char *g_path)
{
    FILE *fp;
    uint8_t f[TMVP_REAL_N];
    uint8_t g[TMVP_REAL_N];
    uint8_t result[TMVP_REAL_N];
    int count, i;
    unsigned int val;

    memset(f, 0, sizeof(f));
    memset(g, 0, sizeof(g));

    /* Read f */
    fp = fopen(f_path, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s'\n", f_path);
        return 1;
    }
    count = 0;
    while (count < TMVP_REAL_N && fscanf(fp, "%x", &val) == 1) {
        f[count++] = (uint8_t)(val & 0xFF);
    }
    fclose(fp);
    printf("Read %d coefficients from %s\n", count, f_path);

    /* Read g */
    fp = fopen(g_path, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s'\n", g_path);
        return 1;
    }
    count = 0;
    while (count < TMVP_REAL_N && fscanf(fp, "%x", &val) == 1) {
        g[count++] = (uint8_t)(val & 0xFF);
    }
    fclose(fp);
    printf("Read %d coefficients from %s\n", count, g_path);

    /* Compute */
    printf("\nComputing MT-TMVP product...\n");
    mt_tmvp_multiply(f, g, result);

    /* Print result */
    printf("\nResult (%d coefficients):\n", TMVP_REAL_N);
    for (i = 0; i < TMVP_REAL_N; i++) {
        printf("%d", result[i]);
        if (i < TMVP_REAL_N - 1) printf("\n");
    }
    printf("\n");

    return 0;
}

/* ============================================================================
 * MAIN
 * ============================================================================ */
int main(int argc, char *argv[])
{
    int i;
    int do_test = 0;
    int do_bench = 0;
    const char *f_file = NULL;
    const char *g_file = NULL;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test") == 0) {
            do_test = 1;
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            do_bench = 1;
        } else if (strcmp(argv[i], "--file") == 0 && i + 2 < argc) {
            f_file = argv[i + 1];
            g_file = argv[i + 2];
            i += 2;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("MT-TMVP Polynomial Multiplier (C Implementation)\n\n");
            printf("Usage: %s [options]\n\n", argv[0]);
            printf("Options:\n");
            printf("  --test              Run verification tests (default)\n");
            printf("  --benchmark         Run performance benchmark\n");
            printf("  --file f.hex g.hex  Compute product from hex input files\n");
            printf("  --help              Show this help\n\n");
            printf("Parameters: N=%d, REAL_N=%d, TILE_SIZE=%d, q=%d\n",
                   TMVP_N, TMVP_REAL_N, TMVP_TILE_SIZE, TMVP_MOD_Q);
            return 0;
        }
    }

    if (f_file && g_file) {
        return run_from_files(f_file, g_file);
    }

    if (!do_test && !do_bench) {
        do_test = 1;  /* Default: run tests */
    }

    if (do_test) {
        int rc = run_tests();
        if (rc != 0) return rc;
    }

    if (do_bench) {
        run_benchmark();
    }

    return 0;
}
