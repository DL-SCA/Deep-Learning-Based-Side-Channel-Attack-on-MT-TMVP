module Top_TMVP
#(parameter N = 512, TILE_SIZE = 16, DATA_WIDTH = 8, REAL_N = 509)
        (clk, reset,
        start,
        ready,
        done,
        bram_f_address_a, bram_f_address_b,
        bram_f_data_out_a, bram_f_data_out_b,
        bram_g_address_a, bram_g_address_b,
        bram_g_data_out_a, bram_g_data_out_b,
        m_axis_tdata, m_axis_tvalid);

    //____________________IO____________________
    // clk and reset
    input   wire                        		clk;
    input   wire                        		reset;
    // saxis
    input   wire                       	 		start;
    output  wire                                ready;
    output  reg     [$clog2(N)-1:0]             bram_f_address_a, bram_f_address_b;
    output  reg     [$clog2(N)-1:0]             bram_g_address_a, bram_g_address_b;
    input   wire    [DATA_WIDTH-1:0]            bram_f_data_out_a, bram_f_data_out_b;
    input   wire    [DATA_WIDTH-1:0]            bram_g_data_out_a, bram_g_data_out_b;
    // maxis
    output  reg     [DATA_WIDTH-1:0] 	        m_axis_tdata;
    output  reg                     			m_axis_tvalid;
    output  reg                                 done;
    //____________________IO____________________

    //____________________Regs_and_Wires____________________
    // FSM - SCA Countermeasure: Added RANDOM_DELAY state
    localparam IDLE = 2'b00;
    localparam LOADING = 2'b01;
    localparam RANDOM_DELAY = 2'b10;  // SCA Countermeasure: random delay state
    localparam BUSY = 2'b11;
    reg     [1:0]               state;
    reg     [$clog2(N)-1:0]     counter_out;
    reg     [$clog2(N):0]       counter_in;
    reg                         loading_done;
	// Main Multiplier
    reg                         start_TMVP;
	reg     [DATA_WIDTH-1:0]   	data_row_1, data_row_2;
	reg     [DATA_WIDTH-1:0]   	data_col_1, data_col_2;
    reg             			data_valid_in;
    reg     [DATA_WIDTH-1:0]   	data_vec_data_1, data_vec_data_2;
    reg             			data_vec_valid;
    wire    [$clog2(N)-1:0]   	address_row_1, address_row_2;
    wire    [$clog2(N)-1:0]   	address_col_1, address_col_2;
    wire            			address_valid;
    wire    [$clog2(N)-1:0]   	address_vec_1, address_vec_2;
    wire            			address_vec_valid;
    wire    [DATA_WIDTH-1:0] 	m_axis_tdata_multiplier;
    wire                        m_axis_tvalid_multiplier;
	// RAMS
	wire    [DATA_WIDTH-1:0]   	data_out_row_A, data_out_row_B;
	wire    [DATA_WIDTH-1:0]   	data_out_col_A, data_out_col_B;
	wire    [DATA_WIDTH-1:0]   	data_out_vec_A, data_out_vec_B;
    reg     [$clog2(N)-1:0]   	address_row_A, address_row_B;
    reg     [$clog2(N)-1:0]   	address_col_A, address_col_B;
    reg     [$clog2(N)-1:0]   	address_vec_A, address_vec_B;
	reg     [DATA_WIDTH-1:0]   	data_in_row_A, data_in_row_B;
	reg     [DATA_WIDTH-1:0]   	data_in_col_A, data_in_col_B;
	reg     [DATA_WIDTH-1:0]   	data_in_vec_A, data_in_vec_B;
	reg   	                    wr_en_row_A, wr_en_row_B;
	reg   	                    wr_en_col_A, wr_en_col_B;
	reg   	                    wr_en_vec_A, wr_en_vec_B;
    reg                         f_g_address_valid, f_g_data_valid;
	// DATA INPUT
	reg 						address_valid_D, address_valid_D2;
	reg 						address_vec_valid_D, address_vec_valid_D2;
    // SCA Countermeasure: LFSR PRNG and random delay
    wire    [31:0]              rng_out;
    reg     [5:0]               delay_counter;   // counts random delay cycles
    reg     [5:0]               delay_target;    // latched from LFSR: 0-63 extra cycles
    //____________________Regs_and_Wires____________________

    //____________________FSM____________________
    // SCA Countermeasure: RANDOM_DELAY state inserted between LOADING and BUSY.
    // After loading f and g data, the FSM waits for a random number of clock
    // cycles (0 to 63) before starting the actual multiplication. This shifts
    // the computation in time by a random offset on every execution, making
    // it much harder for the DL model to align traces and learn patterns.
    always @(posedge clk) begin
        if (!reset) begin
            state                   <=  IDLE;
            start_TMVP              <=  1'b0;
            done                    <=  1'b0;
        end
        else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        state       <=  LOADING;
                    end
                    start_TMVP      <=  1'b0;
                    done            <=  1'b0;
                end
                LOADING: begin
                    if (loading_done) begin
                        // SCA Countermeasure: go to random delay instead of BUSY
                        state       <=  RANDOM_DELAY;
                    end
                    start_TMVP      <=  1'b0;
                    done            <=  1'b0;
                end
                RANDOM_DELAY: begin
                    // SCA Countermeasure: wait for random number of cycles
                    if (delay_counter >= delay_target) begin
                        state       <=  BUSY;
                        start_TMVP  <=  1'b1;
                    end
                    else begin
                        start_TMVP  <=  1'b0;
                    end
                    done            <=  1'b0;
                end
                BUSY: begin
                    if (counter_in  == N) begin
                        state       <=  IDLE;
                        done        <=  1'b1;
                    end
                    start_TMVP      <=  1'b0;
                end
                default: begin
                    start_TMVP      <=  1'b0;
                    state           <=  IDLE;
                    done            <=  1'b0;
                end
            endcase
        end
    end
    // SCA Countermeasure: Random delay counter and target latch
    always @(posedge clk) begin
        if (!reset) begin
            delay_counter           <=  0;
            delay_target            <=  0;
        end
        else if (state == LOADING && loading_done) begin
            // Latch the delay target from LFSR output when entering RANDOM_DELAY
            delay_target            <=  rng_out[5:0];
            delay_counter           <=  0;
        end
        else if (state == RANDOM_DELAY) begin
            delay_counter           <=  delay_counter + 1;
        end
        else begin
            delay_counter           <=  0;
        end
    end
    always @(posedge clk) begin
        case (state)
            IDLE: begin
                counter_out         <=  0;
            end
            LOADING: begin
                if (counter_out == REAL_N) begin
                    if (loading_done) begin
                        counter_out <=  0;
                    end
                end
                else begin
                    counter_out     <=  counter_out + 1;
                end
            end
            BUSY: begin
                if (m_axis_tvalid_multiplier) begin
                    if (counter_out == REAL_N) begin
                        counter_out <=  counter_out;
                    end
                    else begin
                        counter_out <=  counter_out + 1;
                    end
                end
            end
            default: begin
                counter_out         <=  0;
            end
        endcase
    end
    always @(posedge clk) begin
        case (state)
            IDLE: begin
                counter_in          <=  0;
            end
            LOADING: begin
                if (loading_done) begin
                    counter_in      <=  0;
                end
                else if (f_g_data_valid) begin
                    counter_in      <=  counter_in + 1;
                end
            end
            BUSY: begin
                if (m_axis_tvalid_multiplier) begin
                    counter_in      <=  counter_in + 1;
                end
            end
            default: begin
                counter_in          <=  0;
            end
        endcase
    end
    always @(posedge clk) begin
        if (state == LOADING) begin
            if (counter_in == REAL_N) begin
                loading_done        <=  1'b1;
            end
        end
        else begin
            loading_done            <=  1'b0;
        end
    end
    assign  ready   =   (state == IDLE);
    //____________________FSM____________________

    //____________________LFSR_PRNG____________________
    // SCA Countermeasure: LFSR generates random values used for:
    //   1. Arithmetic masking in MatrixVectorMultiplier (rng_out passed down)
    //   2. Random delay target (rng_out[5:0] latched at start of RANDOM_DELAY)
    lfsr_prng #(.WIDTH(32), .SEED_INIT(32'hA5A5_3C3C)) u_prng (
        .clk(clk),
        .reset(reset),
        .seed_en(1'b0),
        .seed(32'h0),
        .rng_out(rng_out),
        .rng_byte()
    );
    //____________________LFSR_PRNG____________________

    //____________________Outside_RAM_Interface____________________
    always @(posedge clk) begin
        bram_f_address_a            <=  counter_out;
        bram_g_address_a            <=  counter_out;
        if (!reset) begin
            f_g_address_valid       <=  1'b0;
            f_g_data_valid          <=  1'b0;
        end
        else if (state == LOADING) begin
            if (counter_out   ==  REAL_N) begin
                f_g_address_valid   <=  1'b0;
            end
            else begin
                f_g_address_valid   <=  1'b1;
            end
            f_g_data_valid          <=  f_g_address_valid;
        end
        else begin
            f_g_address_valid       <=  1'b0;
            f_g_data_valid          <=  1'b0;
        end
    end
    always @(posedge clk) begin
        case (state)
            LOADING: begin
                if (counter_in == 0) begin
                    address_row_A   <=  counter_in;
                end
                else begin
                    address_row_A   <=  REAL_N - counter_in;
                end
                address_col_A       <=  counter_in;
                address_vec_A       <=  counter_in;
                wr_en_row_A         <=  f_g_data_valid;
                wr_en_col_A         <=  f_g_data_valid;
                wr_en_vec_A         <=  f_g_data_valid;
            end
            default: begin
                address_row_A       <=  address_row_1;
                address_col_A       <=  address_col_1;
                address_vec_A       <=  address_vec_1;
                wr_en_row_A         <=  1'b0;
                wr_en_col_A         <=  1'b0;
                wr_en_vec_A         <=  1'b0;
            end
        endcase
        data_in_row_A               <=  bram_f_data_out_a;
        data_in_col_A               <=  bram_f_data_out_a;
        data_in_vec_A               <=  bram_g_data_out_a;
        address_row_B               <=  address_row_2;
        address_col_B               <=  address_col_2;
        address_vec_B               <=  address_vec_2;
        wr_en_row_B                 <=  1'b0;
        wr_en_col_B                 <=  1'b0;
        wr_en_vec_B                 <=  1'b0;
    end
    //____________________Outside_RAM_Interface____________________

    //____________________Main_Multiplier____________________
    if (N == 720  ||  N == 864) begin
        TMVP3_main #(.N(N/3), .DATA_WIDTH(DATA_WIDTH), .TILE_SIZE(TILE_SIZE)) TMVP_multiplier
            (.clk(clk), .reset(reset),
            .start(start_TMVP),
            .address_row_1(address_row_1), .address_row_2(address_row_2), .address_col_1(address_col_1), .address_col_2(address_col_2), .address_valid(address_valid),
            .data_row_1(data_row_1), .data_row_2(data_row_2), .data_col_1(data_col_1), .data_col_2(data_col_2), .data_valid(data_valid_in),
            .address_vec_1(address_vec_1), .address_vec_2(address_vec_2), .address_vec_valid(address_vec_valid),
            .data_vec_data_1(data_vec_data_1), .data_vec_data_2(data_vec_data_2), .data_vec_valid(data_vec_valid),
            .m_axis_tdata(m_axis_tdata_multiplier), .m_axis_tvalid(m_axis_tvalid_multiplier),
            .rng_mask(rng_out)
            );
    end
    else begin
        TMVP2_main #(.N(N/2), .DATA_WIDTH(DATA_WIDTH), .TILE_SIZE(TILE_SIZE)) TMVP_multiplier
            (.clk(clk), .reset(reset),
            .start(start_TMVP),
            .address_row_1(address_row_1), .address_row_2(address_row_2), .address_col_1(address_col_1), .address_col_2(address_col_2), .address_valid(address_valid),
            .data_row_1(data_row_1), .data_row_2(data_row_2), .data_col_1(data_col_1), .data_col_2(data_col_2), .data_valid(data_valid_in),
            .address_vec_1(address_vec_1), .address_vec_2(address_vec_2), .address_vec_valid(address_vec_valid),
            .data_vec_data_1(data_vec_data_1), .data_vec_data_2(data_vec_data_2), .data_vec_valid(data_vec_valid),
            .m_axis_tdata(m_axis_tdata_multiplier), .m_axis_tvalid(m_axis_tvalid_multiplier),
            .rng_mask(rng_out)
            );
    end
    //____________________Main_Multiplier____________________

    //____________________RAMS____________________
	dual_port_ram_TMVP #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH($clog2(N)))	ROW_RAM	(
    .clk(clk),
    .data_a(data_in_row_A), .addr_a(address_row_A), .we_a(wr_en_row_A), .q_a(data_out_row_A),
    .data_b(0), .addr_b(address_row_B), .we_b(1'b0), .q_b(data_out_row_B)
	);
	dual_port_ram_TMVP #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH($clog2(N)))	COL_RAM	(
    .clk(clk),
    .data_a(data_in_col_A), .addr_a(address_col_A), .we_a(wr_en_col_A), .q_a(data_out_col_A),
    .data_b(0), .addr_b(address_col_B), .we_b(1'b0), .q_b(data_out_col_B)
	);
	dual_port_ram_TMVP #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH($clog2(N)))	VEC_RAM	(
    .clk(clk),
    .data_a(data_in_vec_A), .addr_a(address_vec_A), .we_a(wr_en_vec_A), .q_a(data_out_vec_A),
    .data_b(0), .addr_b(address_vec_B), .we_b(1'b0), .q_b(data_out_vec_B)
	);
    //____________________RAMS____________________

    //____________________DATA_INPUT____________________
	always @(posedge clk) begin
        if (state == BUSY) begin
            address_valid_D			<=	address_valid;
            address_vec_valid_D		<=	address_vec_valid;
        end
        else begin
            address_valid_D			<=	1'b0;
            address_vec_valid_D		<=	1'b0;
        end
        address_valid_D2		    <=	address_valid_D;
        address_vec_valid_D2	    <=	address_vec_valid_D;
	end
	always @(posedge clk) begin
        data_row_1                  <=  data_out_row_A;
        data_row_2                  <=  data_out_row_B;
        data_col_1                  <=  data_out_col_A;
        data_col_2                  <=  data_out_col_B;
		data_vec_data_1			    <=	data_out_vec_A;
		data_vec_data_2			    <=	data_out_vec_B;
		data_vec_valid			    <=	address_vec_valid_D2;
		data_valid_in		        <=	address_valid_D2;
	end
    //____________________DATA_INPUT____________________

    //____________________Ignoring____________________
    always @(posedge clk) begin
        if (counter_out == REAL_N) begin
            m_axis_tvalid           <=  1'b0;
        end
        else begin
            m_axis_tvalid           <=  m_axis_tvalid_multiplier;
        end
        m_axis_tdata                <=  m_axis_tdata_multiplier;
    end
    //____________________Ignoring____________________


endmodule
