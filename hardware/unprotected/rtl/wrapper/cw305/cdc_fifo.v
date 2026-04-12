module cdc_fifo#(
    parameter DEPTH = 8,
    parameter DATA_WIDTH = 8
) (
    // Write IO
    input wire wr_clk,
    input wire [DATA_WIDTH-1:0] data_in,
    input wire wr_en,
    input wire wr_rst,

    // Read IO
    input wire rd_clk,
    input wire rd_en,
    input wire rd_rst,
    output wire [DATA_WIDTH-1:0] data_out
);


localparam PTR_WIDTH = $clog2(DEPTH);

// Internal wires
//
reg [PTR_WIDTH:0] b_wr_ptr = 0;
wire [PTR_WIDTH:0] b_wr_ptr_ns;
//
reg [PTR_WIDTH:0] b_rd_ptr = 0;
wire [PTR_WIDTH:0] b_rd_ptr_ns;
//
reg [DATA_WIDTH-1:0] fifo_mem [0:DEPTH-1];


// Write pointer logic
//
always @(posedge wr_clk) begin
    if (wr_rst) begin
        b_wr_ptr <= 0;
    end else begin
        b_wr_ptr <= b_wr_ptr_ns;
    end
end
//
assign b_wr_ptr_ns = b_wr_ptr+wr_en;


// Read pointer logic
//
always @(posedge rd_clk) begin
    if (rd_rst) begin
        b_rd_ptr <= 0;
    end else begin
        b_rd_ptr <= b_rd_ptr_ns;
    end
end
//
assign b_rd_ptr_ns = b_rd_ptr+rd_en;


// Memory Logic
//
always @(posedge wr_clk) begin
    if (wr_en) begin
        fifo_mem[b_wr_ptr[PTR_WIDTH-1:0]] <= data_in;
    end
end
//
assign data_out = fifo_mem[b_rd_ptr[PTR_WIDTH-1:0]];


endmodule