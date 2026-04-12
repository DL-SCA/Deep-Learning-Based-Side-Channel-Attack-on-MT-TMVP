module dual_port_ram_TMVP #(   // read first
    parameter DATA_WIDTH = 8,  // Data bit width
    parameter ADDR_WIDTH = 6,   // Address bit width
    parameter INITIAL_FILE = "VOID"
)(
    input wire clk,  // Clock signal

    // Port A
    input wire [(DATA_WIDTH-1):0] data_a, // Data input for writing
    input wire [(ADDR_WIDTH-1):0] addr_a, // Access address
    input wire we_a,  // Write enable for port A
    output reg [(DATA_WIDTH-1):0] q_a, // Data output for port A

    // Port B
    input wire [(DATA_WIDTH-1):0] data_b, // Data input for writing
    input wire [(ADDR_WIDTH-1):0] addr_b, // Access address
    input wire we_b,  // Write enable for port B
    output reg [(DATA_WIDTH-1):0] q_b // Data output for port A
);

// Define the memory with appropriate size for address and data width
// See UG473 for BRAM configuration constraints (DATA_WIDTH<=36 bits for independent dual port access)
localparam N = (2**ADDR_WIDTH);
(* ram_style = "block" *) reg [DATA_WIDTH-1:0] ram [N-1:0];
integer i;
initial begin
    if (INITIAL_FILE != "VOID") begin
        $readmemh(INITIAL_FILE, ram);
    end
    else begin
        for (i = 0; i < N; i=i+1) begin
            ram[i]      =   0;
        end
    end
end

// Read and write operations for port A
always @(posedge clk) begin
    if (we_a) begin
        ram[addr_a]     <=  data_a; // Write operation
    end
    q_a                 <=  ram[addr_a]; // Read operation
end

// Read and write operations for port B
always @(posedge clk) begin
    if (we_b) begin
        ram[addr_b]     <=  data_b; // Write operation
    end
    q_b                 <=  ram[addr_b]; // Read operation
end


endmodule
