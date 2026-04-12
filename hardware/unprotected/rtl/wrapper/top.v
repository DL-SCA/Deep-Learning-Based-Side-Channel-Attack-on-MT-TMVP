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
/*(* mark_debug = "true" *)*/ wire dut_busy = !dut_ready;
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
reg dut_start_q;
reg dut_reset_n;
//
always @(posedge dut_clk_buf) begin
   dut_start_q <= dut_start;
   if (dut_rst) begin
      dut_reset_n <= 1'b0;
   end else begin
      dut_reset_n <= 1'b1;
   end
end
//
assign tio_trigger = dut_busy;
//
(* DONT_TOUCH = "yes" *) Top_TMVP #(
   .N(`TMVP_N),
   .DATA_WIDTH(`TMVP_DATA_WIDTH),
   .REAL_N(`TMVP_REAL_N),
   .TILE_SIZE(`TMVP_TILE_SIZE)
) u_tmvp (
   .clk(dut_clk_buf),
   .reset(dut_reset_n),
   .start(dut_start_q),
   .ready(dut_ready),
   .done(dut_done),
   .bram_f_address_a(dut_bram_f_address_a),
   .bram_f_address_b(dut_bram_f_address_b),
   .bram_f_data_out_a(dut_bram_f_data_out_a),
   .bram_f_data_out_b(dut_bram_f_data_out_b),
   .bram_g_address_a(dut_bram_g_address_a),
   .bram_g_address_b(dut_bram_g_address_b),
   .bram_g_data_out_a(dut_bram_g_data_out_a),
   .bram_g_data_out_b(dut_bram_g_data_out_b),
   .m_axis_tdata(dut_m_axis_data),
   .m_axis_tvalid(dut_m_axis_data_valid)
);

endmodule
