// ============================================================================
// LFSR-based Pseudo-Random Number Generator for SCA Countermeasures
// ============================================================================
// Generates pseudo-random bits used by the countermeasure modules:
//   - Random masks for arithmetic masking
//   - Random delay values for random delay insertion
//   - Random data for dummy operations
//
// Uses a 32-bit maximal-length LFSR (taps at bits 31, 21, 1, 0)
// which produces a period of 2^32 - 1 before repeating.
//
// The seed input should be a non-zero value provided externally
// (e.g., from a hardware RNG, or loaded via USB before each run).
// ============================================================================

module lfsr_prng #(
    parameter WIDTH = 32,
    parameter [WIDTH-1:0] SEED_INIT = 32'hDEAD_BEEF
)(
    input  wire             clk,
    input  wire             reset,    // active-low reset
    input  wire             seed_en,  // pulse high for one cycle to load seed
    input  wire [WIDTH-1:0] seed,     // seed value (must be non-zero)
    output wire [WIDTH-1:0] rng_out,  // current random output
    output wire [7:0]       rng_byte  // convenience: lower 8 bits
);

    reg [WIDTH-1:0] lfsr;

    // Feedback polynomial: x^32 + x^22 + x^2 + x + 1
    // Taps at bit positions 31, 21, 1, 0 (zero-indexed)
    wire feedback = lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0];

    always @(posedge clk) begin
        if (!reset) begin
            lfsr <= SEED_INIT; // default non-zero value
        end
        else if (seed_en) begin
            lfsr <= (seed == 0) ? SEED_INIT : seed; // prevent all-zero state
        end
        else begin
            lfsr <= {lfsr[WIDTH-2:0], feedback};
        end
    end

    assign rng_out  = lfsr;
    assign rng_byte = lfsr[7:0];

endmodule
