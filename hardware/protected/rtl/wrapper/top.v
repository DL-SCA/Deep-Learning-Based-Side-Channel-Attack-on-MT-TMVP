`include "params.vh"
`timescale 1ns / 1ps

module top (
   // USB Interface
   input wire usb_clk,
   inout wire [7:0] usb_data,
   input wire [`pADDR_WIDTH-1:0] usb_addr,
   input wire usb_rdn,
   input wire usb_wrn,
   input wire usb_cen,
   input wire usb_trigger,

   // PLL
   input wire pll_clk1,

   // 20-Pin Connector Stuff
   output wire tio_trigger,
   output wire tio_clkout,
   input wire tio_clkin
);

// Clocking
//
wire usb_clk_buf;
wire dut_clk_buf;
//
clock_config U_clock_config (
   .usb_clk (usb_clk),
   .usb_clk_buf (usb_clk_buf),
   .cw_clkin (tio_clkin),
   .pll_clk1 (pll_clk1),
   .cw_clkout (tio_clkout),
   .dut_clk_buf (dut_clk_buf)
);


// USB Adapter
//
wire [7:0] usb_din;
wire [7:0] usb_dout;
wire isout;
wire [`pADDR_WIDTH-`pBYTECNT_SIZE-1:0] reg_address;
wire [`pBYTECNT_SIZE-1:0] reg_bytecnt;
wire reg_addrvalid;
wire [7:0] write_data;
wire [7:0] read_data;
wire reg_read;
wire reg_write;
//
usb_reg_adapter U_usb_reg_adapter (
   .usb_clk (usb_clk_buf),
   .usb_din (usb_din),
   .usb_dout (usb_dout),
   .usb_rdn (usb_rdn),
   .usb_wrn (usb_wrn),
   .usb_cen (usb_cen),
   .usb_alen (1'b0), // unused
   .usb_addr (usb_addr),
   .usb_isout (isout),
   .reg_address (reg_address),
   .reg_bytecnt (reg_bytecnt),
   .reg_datao (write_data),
   .reg_datai (read_data),
   .reg_read (reg_read),
   .reg_write (reg_write),
   .reg_addrvalid (reg_addrvalid)
);
//
genvar i;
generate
   for (i=0; i<8; i=i+1) begin
      IOBUF #(
         .DRIVE(12),
         .IOSTANDARD("LVCMOS33")
      ) IOBUF_inst (
         .O(usb_din[i]),
         .IO(usb_data[i]),
         .I(usb_dout[i]),
         .T(~isout)
      );
   end
endgenerate


// DUT
//
wire dut_rst;
/*(* mark_debug = "true" *)*/ wire dut_start;
/*(* mark_debug = "true" *)*/ wire [(`TMVP_DATA_WIDTH-1):0] dut_bram_f_data_out_a;
/*(* mark_debug = "true" *)*/ wire [(`TMVP_DATA_WIDTH-1):0] dut_bram_f_data_out_b;
/*(* mark_debug = "true" *)*/ wire [(`TMVP_DATA_WIDTH-1):0] dut_bram_g_data_out_a;
/*(* mark_debug = "true" *)*/ wire [(`TMVP_DATA_WIDTH-1):0] dut_bram_g_data_out_b;
//
wire [`TMVP_DATA_WIDTH-1:0] dut_m_axis_data;
/*(* mark_debug = "true" *)*/ wire [$clog2(`TMVP_N)-1:0] dut_bram_f_address_a;
/*(* mark_debug = "true" *)*/ wire [$clog2(`TMVP_N)-1:0] dut_bram_f_address_b; // unused by design
/*(* mark_debug = "true" *)*/ wire [$clog2(`TMVP_N)-1:0] dut_bram_g_address_a;
/*(* mark_debug = "true" *)*/ wire [$clog2(`TMVP_N)-1:0] dut_bram_g_address_b; // unused by design
wire dut_done; // unused, use of valid and busy instead
wire dut_ready;
wire dut_m_axis_data_valid;
// SCA Countermeasure: dut_busy now reflects the entire dummy+real sequence.
// dut_tmvp_busy is the raw TMVP module busy, used internally by the
// dummy rounds controller. dut_busy is what the host sees via usb_reg.
wire dut_tmvp_busy = !dut_ready;
// dut_busy is declared as a wire here; it will be assigned after dr_trigger
// is defined (see below). Forward-declared for usb_reg connection.
wire dut_busy;
//
usb_reg U_usb_reg (
   .usb_clk(usb_clk_buf),
   .dut_clk(dut_clk_buf),
   .reg_address(reg_address[`pADDR_WIDTH-`pBYTECNT_SIZE-1:0]),
   .reg_bytecnt(reg_bytecnt),
   .read_data(read_data),
   .write_data(write_data),
   .reg_read(reg_read),
   .reg_write(reg_write),
   .reg_addrvalid(reg_addrvalid),
   .exttrigger_in(usb_trigger),
   //
   .dut_bram_f_address_a(dut_bram_f_address_a),
   .dut_bram_f_address_b(dut_bram_f_address_b),
   .dut_bram_g_address_a(dut_bram_g_address_a),
   .dut_bram_g_address_b(dut_bram_g_address_b),
   .dut_m_axis_data(dut_m_axis_data),
   .dut_m_axis_data_valid(dut_m_axis_data_valid),
   .dut_busy(dut_busy),
   //
   .dut_rst(dut_rst),
   .dut_start(dut_start),
   .dut_bram_f_data_out_a(dut_bram_f_data_out_a),
   .dut_bram_f_data_out_b(dut_bram_f_data_out_b),
   .dut_bram_g_data_out_a(dut_bram_g_data_out_a),
   .dut_bram_g_data_out_b(dut_bram_g_data_out_b)
);
//

// Debug RAM
/*
reg [$clog2(`TMVP_N)-1:0] bram_addr = 0;
//
always @(posedge dut_clk_buf) begin
   if (dut_start) begin
      bram_addr <= 0;
   end else if (bram_addr == `TMVP_REAL_N) begin
      bram_addr <= 0;
   end else begin
      bram_addr <= bram_addr+1;
   end
end
//
assign dut_bram_f_address_a = bram_addr;
assign dut_bram_f_address_b = 0;
assign dut_bram_g_address_a = bram_addr;
assign dut_bram_g_address_b = 0;
assign dut_ready = (bram_addr == 0);
assign dut_done = 1'b0;
assign dut_m_axis_data = 0;
assign dut_m_axis_data_valid = 0;
*/

/*
(* mark_debug = "true" *) reg [31:0] runtime_cnt;
always @(posedge dut_clk_buf) begin
   if (dut_start) begin
      runtime_cnt <= 0;
   end else if (dut_busy) begin
      runtime_cnt <= runtime_cnt+1;
   end
end*/

// Drive reset/start
//
reg dut_reset_n;
//
always @(posedge dut_clk_buf) begin
   if (dut_rst) begin
      dut_reset_n <= 1'b0;
   end else begin
      dut_reset_n <= 1'b1;
   end
end

// ============================================================================
// SCA Countermeasure: Dummy Rounds Controller
// ============================================================================
// When the host triggers a computation, this controller first runs a "dummy"
// TMVP multiplication using random data from an LFSR (instead of the real f/g).
// Only after the dummy round completes does it run the real computation with
// the actual f/g data. The trigger output (tio_trigger) stays asserted during
// BOTH rounds, so the attacker capturing from the trigger edge sees a combined
// trace that includes a decoy computation before the real one. The number of
// dummy rounds is configurable (currently 1) and the random data changes every
// execution due to the free-running LFSR.
// ============================================================================

// Dummy round FSM states
localparam DR_IDLE       = 3'd0;
localparam DR_DUMMY_RUN  = 3'd1;
localparam DR_DUMMY_WAIT = 3'd2;
localparam DR_REAL_START = 3'd3;
localparam DR_REAL_WAIT  = 3'd4;

reg [2:0] dr_state;
reg       tmvp_start_pulse;  // one-cycle start pulse to Top_TMVP
reg       use_dummy_data;    // mux select: 1=random data, 0=real BRAM data
reg       dr_trigger;        // trigger output covering both rounds

// LFSR for dummy round random data
wire [31:0] dummy_rng_out;
wire [7:0]  dummy_rng_byte;

lfsr_prng #(.WIDTH(32), .SEED_INIT(32'hDEAD_BEEF)) u_dummy_prng (
   .clk(dut_clk_buf),
   .reset(dut_reset_n),
   .seed_en(1'b0),
   .seed(32'h0),
   .rng_out(dummy_rng_out),
   .rng_byte(dummy_rng_byte)
);

always @(posedge dut_clk_buf) begin
   if (!dut_reset_n) begin
      dr_state         <= DR_IDLE;
      tmvp_start_pulse <= 1'b0;
      use_dummy_data   <= 1'b0;
      dr_trigger       <= 1'b0;
   end
   else begin
      case (dr_state)
         DR_IDLE: begin
            tmvp_start_pulse <= 1'b0;
            dr_trigger       <= 1'b0;
            use_dummy_data   <= 1'b0;
            if (dut_start) begin
               // Start dummy round with random data
               dr_state         <= DR_DUMMY_RUN;
               use_dummy_data   <= 1'b1;
               tmvp_start_pulse <= 1'b1;
               dr_trigger       <= 1'b1;  // assert trigger for attacker
            end
         end
         DR_DUMMY_RUN: begin
            tmvp_start_pulse <= 1'b0;
            // Wait for TMVP to go busy (use raw TMVP busy, not combined)
            if (dut_tmvp_busy) begin
               dr_state <= DR_DUMMY_WAIT;
            end
         end
         DR_DUMMY_WAIT: begin
            // Wait for dummy computation to finish (TMVP returns to ready)
            if (dut_ready) begin
               dr_state         <= DR_REAL_START;
               use_dummy_data   <= 1'b0;  // switch to real data
               tmvp_start_pulse <= 1'b1;  // start real computation
            end
         end
         DR_REAL_START: begin
            tmvp_start_pulse <= 1'b0;
            if (dut_tmvp_busy) begin
               dr_state <= DR_REAL_WAIT;
            end
         end
         DR_REAL_WAIT: begin
            // Wait for real computation to finish
            if (dut_ready) begin
               dr_state   <= DR_IDLE;
               dr_trigger <= 1'b0;
            end
         end
         default: begin
            dr_state <= DR_IDLE;
         end
      endcase
   end
end

// Trigger output: asserted during entire dummy+real sequence
assign tio_trigger = dr_trigger;
// SCA Countermeasure: Host sees busy for the entire dummy+real sequence
assign dut_busy = dr_trigger;

// Mux: during dummy round, feed random data instead of real f/g BRAM contents
wire [`TMVP_DATA_WIDTH-1:0] muxed_f_data_a = use_dummy_data ? dummy_rng_out[`TMVP_DATA_WIDTH-1:0]   : dut_bram_f_data_out_a;
wire [`TMVP_DATA_WIDTH-1:0] muxed_f_data_b = use_dummy_data ? dummy_rng_out[`TMVP_DATA_WIDTH+7:8]   : dut_bram_f_data_out_b;
wire [`TMVP_DATA_WIDTH-1:0] muxed_g_data_a = use_dummy_data ? dummy_rng_out[`TMVP_DATA_WIDTH+15:16] : dut_bram_g_data_out_a;
wire [`TMVP_DATA_WIDTH-1:0] muxed_g_data_b = use_dummy_data ? dummy_rng_out[`TMVP_DATA_WIDTH+23:24] : dut_bram_g_data_out_b;

//
(* DONT_TOUCH = "yes" *) Top_TMVP #(
   .N(`TMVP_N),
   .DATA_WIDTH(`TMVP_DATA_WIDTH),
   .REAL_N(`TMVP_REAL_N),
   .TILE_SIZE(`TMVP_TILE_SIZE)
) u_tmvp (
   .clk(dut_clk_buf),
   .reset(dut_reset_n),
   .start(tmvp_start_pulse),
   .ready(dut_ready),
   .done(dut_done),
   .bram_f_address_a(dut_bram_f_address_a),
   .bram_f_address_b(dut_bram_f_address_b),
   .bram_f_data_out_a(muxed_f_data_a),
   .bram_f_data_out_b(muxed_f_data_b),
   .bram_g_address_a(dut_bram_g_address_a),
   .bram_g_address_b(dut_bram_g_address_b),
   .bram_g_data_out_a(muxed_g_data_a),
   .bram_g_data_out_b(muxed_g_data_b),
   .m_axis_tdata(dut_m_axis_data),
   .m_axis_tvalid(dut_m_axis_data_valid)
);

endmodule
