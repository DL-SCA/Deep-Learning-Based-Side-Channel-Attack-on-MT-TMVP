`include "params.vh"

`timescale 1ns / 1ps

module usb_reg(
   // Interface to usb_reg_adapter:
   input wire usb_clk,
   input wire dut_clk,
   input wire [`pADDR_WIDTH-`pBYTECNT_SIZE-1:0] reg_address, // Address of register
   input wire [`pBYTECNT_SIZE-1:0] reg_bytecnt, // Current byte count
   /*(* mark_debug = "true" *)*/ output reg [7:0] read_data, //
   input wire [7:0] write_data, //
   input wire reg_read, // Read flag. One clock cycle AFTER this flag is high
   // valid data must be present on the read_data bus
   input wire reg_write, // Write flag. When high on rising edge valid data is
   // present on write_data
   input wire reg_addrvalid, // Address valid flag

   // from top:
   input wire exttrigger_in,

   // register inputs:
   input wire [$clog2(`TMVP_N)-1:0] dut_bram_f_address_a,
   input wire [$clog2(`TMVP_N)-1:0] dut_bram_f_address_b,
   input wire [$clog2(`TMVP_N)-1:0] dut_bram_g_address_a,
   input wire [$clog2(`TMVP_N)-1:0] dut_bram_g_address_b,
   /*(* mark_debug = "true" *)*/ input wire [`TMVP_DATA_WIDTH-1:0] dut_m_axis_data,
   /*(* mark_debug = "true" *)*/ input wire dut_m_axis_data_valid,
   /*(* mark_debug = "true" *)*/ input wire dut_busy,

   // register outputs:
   output wire dut_rst,
   /*(* mark_debug = "true" *)*/ output wire dut_start,
   output wire [`TMVP_DATA_WIDTH-1:0] dut_bram_f_data_out_a,
   output wire [`TMVP_DATA_WIDTH-1:0] dut_bram_f_data_out_b,
   output wire [`TMVP_DATA_WIDTH-1:0] dut_bram_g_data_out_a,
   output wire [`TMVP_DATA_WIDTH-1:0] dut_bram_g_data_out_b
);

// Paramters
//
localparam integer FG_FIFO_DATA_WIDTH = 1+8+`pBYTECNT_SIZE;
localparam integer PACKETS_IN_WRITE_FIFO = 2*`TMVP_REAL_N; // The expected amount of packets that should be read from the USB_WRITE fifo
//
localparam integer PACKETS_IN_OUTPUT_FIFO = `TMVP_REAL_N; // The expected amount of 8-bit packets to be in the output fifo

// Reset signal
//
reg reg_dut_rst_usbclk;

// USB-RAM FIFO
//
wire write_usb_fifo = reg_addrvalid && reg_write && (reg_address == `REG_DUT_RAM_F || reg_address == `REG_DUT_RAM_G);
/*(* mark_debug = "true" *)*/ wire flush_fg_fifo_req;
/*(* mark_debug = "true" *)*/ reg read_fg_fifo = 0;
/*(* mark_debug = "true" *)*/ reg [$clog2(PACKETS_IN_WRITE_FIFO)-1:0] read_fg_fifo_cnt = 0;
//
wire [FG_FIFO_DATA_WIDTH-1:0] fg_fifo_in = {reg_address == `REG_DUT_RAM_F, write_data, reg_bytecnt};
//
wire [FG_FIFO_DATA_WIDTH-1:0] fg_fifo_out;
/*(* mark_debug = "true" *)*/ wire write_f_RAM = fg_fifo_out[8+`pBYTECNT_SIZE] && read_fg_fifo;
/*(* mark_debug = "true" *)*/ wire write_g_RAM = !fg_fifo_out[8+`pBYTECNT_SIZE] && read_fg_fifo;
/*(* mark_debug = "true" *)*/ wire [7:0] write_data_RAM = fg_fifo_out[8+`pBYTECNT_SIZE-1:`pBYTECNT_SIZE];
/*(* mark_debug = "true" *)*/ wire [$clog2(`TMVP_N)-1:0] write_addr_RAM = fg_fifo_out[`pBYTECNT_SIZE-1:0];
//
always @(posedge dut_clk) begin
   if (dut_rst) begin
      read_fg_fifo <= 1'b0;
      read_fg_fifo_cnt <= 0;
   end else if (flush_fg_fifo_req) begin
      read_fg_fifo <= 1'b1;
      read_fg_fifo_cnt <= 0;
   end else if (read_fg_fifo) begin
      if (read_fg_fifo_cnt == PACKETS_IN_WRITE_FIFO-1) begin
         read_fg_fifo_cnt <= 0;
         read_fg_fifo <= 0;
      end else begin
         read_fg_fifo_cnt <= read_fg_fifo_cnt+1;
      end
   end
end
//
cdc_fifo #(
   .DEPTH(PACKETS_IN_WRITE_FIFO),
   .DATA_WIDTH(FG_FIFO_DATA_WIDTH)
) fg_RAM_fifo (
   .wr_clk(usb_clk),
   .data_in(fg_fifo_in),
   .wr_en(write_usb_fifo),
   .wr_rst(reg_dut_rst_usbclk),
   .rd_clk(dut_clk),
   .rd_en(read_fg_fifo),
   .rd_rst(dut_rst),
   .data_out(fg_fifo_out)
);

// f and g RAM
//
dual_port_ram_TMVP #(
   .DATA_WIDTH(`TMVP_DATA_WIDTH),
   .ADDR_WIDTH($clog2(`TMVP_N))
) f_RAM (
   .clk(dut_clk),
   .data_a(write_data_RAM),
   .addr_a(read_fg_fifo ? write_addr_RAM : dut_bram_f_address_a),
   .we_a(write_f_RAM), // Only written to via USB, single channel
   .q_a(dut_bram_f_data_out_a),
   .data_b(0),
   .addr_b(dut_bram_f_address_b),
   .we_b(0),
   .q_b(dut_bram_f_data_out_b)
);
//
dual_port_ram_TMVP #(
   .DATA_WIDTH(`TMVP_DATA_WIDTH),
   .ADDR_WIDTH($clog2(`TMVP_N))
) g_RAM (
   .clk(dut_clk),
   .data_a(write_data_RAM),
   .addr_a(read_fg_fifo ? write_addr_RAM : dut_bram_g_address_a),
   .we_a(write_g_RAM), // Only written to via USB, single channel
   .q_a(dut_bram_g_data_out_a),
   .data_b(0),
   .addr_b(dut_bram_g_address_b),
   .we_b(0),
   .q_b(dut_bram_g_data_out_b)
);


// DUT Output M Axis FIFO
//
wire out_rd_en;
reg out_rd_en_d = 0;
//
wire [7:0] dut_output_fifo_data_out;
//
cdc_fifo #(
   .DEPTH(PACKETS_IN_OUTPUT_FIFO),
   .DATA_WIDTH(`TMVP_DATA_WIDTH)
) dut_output_fifo (
   .wr_clk(dut_clk),
   .data_in(dut_m_axis_data),
   .wr_en(dut_m_axis_data_valid),
   .wr_rst(dut_rst),
   .rd_clk(usb_clk),
   .rd_en(out_rd_en && !out_rd_en_d),
   .rd_rst(reg_dut_rst_usbclk),
   .data_out(dut_output_fifo_data_out)
);
//
// CW keeps read signal over multiple clock cycles, but does not actually read.
assign out_rd_en = reg_addrvalid && reg_read && reg_address == `REG_DUT_DATA_OUT;
//
always @(posedge usb_clk) begin
   out_rd_en_d <= out_rd_en;
end


// Go Signal
//
reg go_ext, reg_go_ext;
/*(* mark_debug = "true" *)*/ wire dut_go_ext;
reg reg_dut_go_usbclk;
/*(* mark_debug = "true" *)*/ wire dut_go_usb; // unused
//
(* ASYNC_REG = "TRUE" *) reg [1:0] go_ext_buf;
//
assign dut_start = dut_go_ext || dut_go_usb;
assign dut_go_ext = go_ext & !reg_go_ext;
//
always @(posedge dut_clk) begin
   {reg_go_ext, go_ext, go_ext_buf} <= {go_ext, go_ext_buf, exttrigger_in};
end
//
cdc_pulse U_go_pulse (
   .src_clk (usb_clk),
   .src_pulse (reg_dut_go_usbclk),
   .dst_clk (dut_clk),
   .dst_pulse (dut_go_usb)
);

// Reset signal
//
cdc_pulse U_rst_pulse (
   .src_clk (usb_clk),
   .src_pulse (reg_dut_rst_usbclk),
   .dst_clk (dut_clk),
   .dst_pulse (dut_rst)
);

// Flush fifo signal
//
/*(* mark_debug = "true" *)*/ reg reg_flush_fifo_usbclk;
//
cdc_pulse U_flush_pulse (
   .src_clk (usb_clk),
   .src_pulse (reg_flush_fifo_usbclk),
   .dst_clk (dut_clk),
   .dst_pulse (flush_fg_fifo_req)
);


// Busy signal
//
reg reg_d1_dut_busy_usbclk;
(* ASYNC_REG = "TRUE" *) reg [1:0] reg_dut_busy_usbclk;
//
always @(posedge usb_clk) begin
   {reg_d1_dut_busy_usbclk, reg_dut_busy_usbclk} <= {reg_dut_busy_usbclk, dut_busy};
end


// USB Interface
//
reg [7:0] reg_read_data;
//
//////////////////////////////////
// read logic:
//////////////////////////////////
always @(*) begin
   if (reg_addrvalid && reg_read) begin
      case (reg_address)
         // TODO: Implement read logic
         `REG_DUT_GO: reg_read_data = reg_d1_dut_busy_usbclk;
         `REG_DUT_DATA_OUT: reg_read_data = dut_output_fifo_data_out;
         default: reg_read_data = 0;
      endcase
   end else begin
      reg_read_data = 0;
   end
end
//
// Register output read data to ease timing. If you need read data one clock
// cycle earlier, simply remove this stage:
always @(posedge usb_clk) begin
   read_data <= reg_read_data;
end
//
//////////////////////////////////
// write logic (USB clock domain):
//////////////////////////////////
always @(posedge usb_clk) begin
   if (reg_addrvalid && reg_write) begin
      reg_dut_go_usbclk <= (reg_address == `REG_DUT_GO); // Create pulse
      reg_dut_rst_usbclk <= (reg_address == `REG_DUT_RESET); // Create pulse
      reg_flush_fifo_usbclk <= (reg_address == `REG_DUT_FLUSHFIFO);
   end else begin
      reg_dut_go_usbclk <= 1'b0;
      reg_dut_rst_usbclk <= 1'b0;
      reg_flush_fifo_usbclk <= 1'b0;
   end
end

endmodule
