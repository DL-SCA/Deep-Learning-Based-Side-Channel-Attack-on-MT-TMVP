module TMVP2 
    #(parameter N = 32, DATA_WIDTH = 4)
        (clk, reset, 
        start, ready,
		data_row_data_1, data_row_data_2, data_row_valid,
		data_vec_data_1, data_vec_data_2, data_vec_valid,
		address_1, address_1_isRow,
		address_2, address_2_isRow,
		address_row_valid,
		address_vec_1, address_vec_2, address_vec_valid,
        m_axis_tdata, m_axis_tvalid);
    
    //____________________IO____________________
    // clk and reset
    input   wire                        		clk;
    input   wire                        		reset;
    // saxis
    input   wire                        		start;
	output	wire								ready;
    input	wire    signed	[DATA_WIDTH-1:0]    data_row_data_1;
    input	wire    signed 	[DATA_WIDTH-1:0]    data_row_data_2;
	input	wire                        		data_row_valid;  

    input	wire    signed	[DATA_WIDTH-1:0]    data_vec_data_1;
    input	wire    signed	[DATA_WIDTH-1:0]    data_vec_data_2;
    input	wire                        		data_vec_valid;  

    output  reg     [$clog2(N)-1:0]				address_1;
    output  reg         						address_1_isRow;
    output  reg     [$clog2(N)-1:0]     		address_2;
    output  reg                         		address_2_isRow;
    output  reg                         		address_row_valid;  
    output  reg     [$clog2(N)-1:0]     		address_vec_1;
    output  reg     [$clog2(N)-1:0]     		address_vec_2;
    output  reg                         		address_vec_valid;  
    // maxis
    output  reg     signed [DATA_WIDTH-1:0]  	m_axis_tdata;
    output  reg                         		m_axis_tvalid;
    //____________________IO____________________

    //____________________Regs_and_Wires____________________
	// FSM
	localparam IDLE = 3'b000;
	localparam ROW_VEC_1 = 3'b001;
	localparam COL_1 = 3'b011;
	localparam ROW_VEC_2 = 3'b010;
	localparam COL_2 = 3'b110;
	localparam ROW_VEC_3 = 3'b100;
	localparam COL_3 = 3'b101;
	localparam RESULT = 3'b111;
	reg				[2:0]				state_add;
	reg 			[$clog2(N/2)-1:0] 	counter_add;
	reg 			[$clog2(N):0] 		counter_data_out;
	reg 			[$clog2(N/2)-1:0] 	counter_data_out_each_part;
	reg 			[$clog2(3*N):0] 	counter_data_in;
	// Multiplier
    reg     signed  [DATA_WIDTH-1:0]    s_axis_tdata_vec;
    reg                                 s_axis_tvalid;
    reg     signed	[DATA_WIDTH-1:0]    s_axis_tdata_row;
    wire    signed	[DATA_WIDTH-1:0]  	m_axis_tdata_multiplier;
    wire                                m_axis_tvalid_multiplier;
    reg    	signed	[DATA_WIDTH-1:0]  	m_axis_tdata_multiplier_D;
    reg    	                            m_axis_tvalid_multiplier_D;
	// Result
	reg 								RAM_wr_en;
	// reg 	signed 	[DATA_WIDTH-1:0] 	RAM_data_in;
	wire 	signed 	[DATA_WIDTH-1:0] 	RAM_data_out;
	// reg 	 		[$clog2(N)-1:0] 	RAM_address[1:0];
	reg									common_part_done;
    //____________________Regs_and_Wires____________________
    
    //____________________Multiplier____________________
	MatrixVectorMultiplier #(.N(N/2), .DATA_WIDTH(DATA_WIDTH)) Multiplier
	(   .clk(clk), .reset(reset), 
		.s_axis_tdata_vec(s_axis_tdata_vec),
		.s_axis_tdata_row(s_axis_tdata_row), 
		.s_axis_tvalid(s_axis_tvalid),
		.m_axis_tdata(m_axis_tdata_multiplier), .m_axis_tvalid(m_axis_tvalid_multiplier)
	);
    //____________________Multiplier____________________

    //____________________Control_Address____________________
	always @(posedge clk) begin
        if (!reset) begin
			state_add	<=	IDLE;
		end
		else begin
			case (state_add)
				IDLE: begin
					if (start) begin
						state_add	<=	ROW_VEC_1;
					end
				end
				ROW_VEC_1: begin
					if (counter_add == (N/2-1)) begin
						state_add	<=	COL_1;
					end
				end 
				COL_1: begin
					if (counter_add == (N/2-1)) begin
						state_add	<=	ROW_VEC_2;
					end
				end 
				ROW_VEC_2: begin
					if (counter_add == (N/2-1)) begin
						state_add	<=	COL_2;
					end
				end 
				COL_2: begin
					if (counter_add == (N/2-1)) begin
						state_add	<=	ROW_VEC_3;
					end
				end 
				ROW_VEC_3: begin
					if (counter_add == (N/2-1)) begin
						state_add	<=	COL_3;
					end
				end 
				COL_3: begin
					if (counter_add == (N/2-1)) begin
						state_add	<=	RESULT;
					end
				end 
				RESULT: begin
					if (counter_data_out == (3*N/2-1)) begin
						state_add	<=	IDLE;
					end
				end 
				default: begin
						state_add	<=	IDLE;
				end
			endcase
		end
    end
	always @(posedge clk) begin
		if (state_add == ROW_VEC_1	||	state_add == ROW_VEC_2	||	state_add == ROW_VEC_3) begin
			if (counter_add	==	N/2-1) begin
				counter_add			<=	1;				
			end
			else begin
				counter_add			<=	counter_add	+ 1;				
			end
		end
		else if (state_add == COL_1	||	state_add == COL_2	||	state_add == COL_3) begin
			if (counter_add	==	N/2-1) begin
				counter_add			<=	0;				
			end
			else begin
				counter_add			<=	counter_add	+ 1;				
			end				
		end
		else begin
			counter_add				<=	0;
		end		
	end
	assign	ready	=	(state_add	==	IDLE);
    //____________________Control_Address____________________

	//____________________Address_Assignment____________________
    always @(posedge clk) begin
		case (state_add)
			ROW_VEC_1: begin
				address_1		<=	(N/2-1) - counter_add;
				address_1_isRow	<=	1'b1;
				address_vec_1	<=	(N/2-1) - counter_add;
				address_vec_2	<=	(N-1) - counter_add;
			end 
			COL_1: begin
				address_1		<=	counter_add;
				address_1_isRow	<=	1'b0;
			end
			ROW_VEC_2: begin
				address_1		<=	(N/2-1) - counter_add;
				address_1_isRow	<=	1'b1;
				address_2		<=	(N-1) - counter_add;
				address_2_isRow	<=	1'b1;
				address_vec_1	<=	(N-1) - counter_add;
			end 
			COL_2: begin
				address_1		<=	counter_add;
				address_1_isRow	<=	1'b0;
				address_2		<=	N/2 - counter_add;
				address_2_isRow	<=	1'b1;
			end
			ROW_VEC_3: begin
				address_1		<=	(N/2-1) - counter_add;
				address_1_isRow	<=	1'b1;
				address_2		<=	counter_add + 1;
				address_2_isRow	<=	1'b0;
				address_vec_1	<=	(N/2-1) - counter_add;
			end 
			COL_3: begin
				address_1		<=	counter_add;
				address_1_isRow	<=	1'b0;
				address_2		<=	N/2 + counter_add;
				address_2_isRow	<=	1'b0;
			end
			default: begin
				address_1		<=	0;
				address_1_isRow	<=	1'b0;
				address_2		<=	0;
				address_2_isRow	<=	1'b0;
			end
		endcase
	end
	always @(posedge clk) begin
		address_row_valid		<=	(state_add == ROW_VEC_1)	||	(state_add == ROW_VEC_2)	|| 	(state_add == ROW_VEC_3)	||
									(state_add == COL_1)		||	(state_add == COL_2)		||	(state_add == COL_3);
	end
	//____________________Address_Assignment____________________
	
	//____________________Multiplier_Input____________________
	always @(posedge clk) begin
		if (state_add	==	IDLE) begin
			counter_data_in		<=	0;
		end
		else begin
			if (data_row_valid) begin
				counter_data_in	<=	counter_data_in + 1;
			end
		end
	end
	always @(posedge clk) begin
		if (counter_data_in < N-1) begin
			s_axis_tdata_row		<=	data_row_data_1;
			s_axis_tdata_vec		<=	data_vec_data_1 + data_vec_data_2; 
			s_axis_tvalid			<=	data_row_valid;
		end 
		else begin
			s_axis_tdata_row		<=	data_row_data_2 - data_row_data_1;
			s_axis_tvalid			<=	data_row_valid;
			s_axis_tdata_vec		<=	data_vec_data_1; 
		end 
	end
    //____________________Multiplier_Input____________________

    //____________________Output____________________
	always @(posedge clk) begin
		if (state_add	==	IDLE) begin
			counter_data_out			<=	0;
		end
		else begin
			if (m_axis_tvalid_multiplier) begin
				if (counter_data_out	==	(3*N/2-1)) begin
					counter_data_out	<=	counter_data_out;
				end
				else begin
					counter_data_out	<=	counter_data_out + 1;
				end
			end
		end
	end
	always @(posedge clk) begin
		if (state_add	==	IDLE) begin
			counter_data_out_each_part			<=	0;
		end
		else begin
			if (m_axis_tvalid_multiplier) begin
				if (counter_data_out_each_part	==	(N/2-1)) begin
					counter_data_out_each_part	<=	0;
				end
				else begin
					counter_data_out_each_part	<=	counter_data_out_each_part + 1;
				end
			end
		end
	end
	always @(posedge clk) begin
		if (state_add	==	IDLE) begin
			RAM_wr_en					<=	1'b1;
		end
		else begin
			if (counter_data_out	==	N/2 - 1) begin
				RAM_wr_en				<=	1'b0;
			end
		end
	end
	always @(posedge clk ) begin
		common_part_done				<=	counter_data_out >= (N/2);
	end
	always @(posedge clk ) begin
		m_axis_tdata_multiplier_D		<=	m_axis_tdata_multiplier;
		m_axis_tvalid_multiplier_D		<=	m_axis_tvalid_multiplier;
	end
	always @(posedge clk) begin
		m_axis_tdata					<=	m_axis_tdata_multiplier_D	+	RAM_data_out;
		if (common_part_done) begin
			m_axis_tvalid				<=	m_axis_tvalid_multiplier_D;
		end
		else begin
			m_axis_tvalid				<=	1'b0;
		end
	end
    //____________________Output____________________
	
    //____________________RAM____________________
    dual_port_ram_TMVP #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH($clog2(N/2))) RESULT_RAM (
        .clk(clk),
        .data_a(m_axis_tdata_multiplier), .addr_a(counter_data_out[$clog2(N/2)-1:0]), .we_a(RAM_wr_en),
		.addr_b(counter_data_out_each_part), .q_b(RAM_data_out)
    );
    //____________________RAM____________________
endmodule