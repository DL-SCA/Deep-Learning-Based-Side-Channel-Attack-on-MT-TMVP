module TMVP2_main    
    #(parameter N = 256, DATA_WIDTH = 7, TILE_SIZE = 16)
        (clk, reset, 
        start, 
        address_row_1, address_row_2, address_col_1, address_col_2, address_valid,
		data_row_1, data_row_2, data_col_1, data_col_2, data_valid,
        address_vec_1, address_vec_2, address_vec_valid,
		data_vec_data_1, data_vec_data_2, data_vec_valid,
        m_axis_tdata, m_axis_tvalid
        );

    //____________________IO____________________
    // clk and reset
    input   wire                                clk;
    input   wire                                reset;
    // saxis
    input   wire                       	 		start;
    input	wire	signed	[DATA_WIDTH-1:0]   	data_row_1, data_row_2, data_col_1, data_col_2;
	input	wire                               	data_valid;  
    input	wire    signed  [DATA_WIDTH-1:0]   	data_vec_data_1, data_vec_data_2;
    input	wire                               	data_vec_valid;  
    output  reg     [$clog2(N):0]     		    address_row_1, address_row_2, address_col_1, address_col_2;
    output  reg                         		address_valid;
    output  reg     [$clog2(N):0]     		    address_vec_1, address_vec_2;
    output  reg          						address_vec_valid;
    // maxis
    output  reg     [DATA_WIDTH-1:0]            m_axis_tdata;
    output  reg                         		m_axis_tvalid;    
    //____________________IO____________________

    //____________________Regs_and_Wires____________________
    // FSM
    localparam IDLE = 3'b000;
    localparam FIRST_LOAD = 3'b001;
    localparam STEP_1 = 3'b011;
    localparam STEP_2 = 3'b010;
    localparam LAST_LOAD = 3'b110;
    localparam STEP_3 = 3'b100;
    reg             [2:0]                       state;
    reg                                         load_done;
    reg             [$clog2(N):0]               counter_in;
    reg             [$clog2(N):0]               counter_out;
    // RAMs
    wire            [DATA_WIDTH-1:0]            ROW_data_out[1:0];
    wire            [DATA_WIDTH-1:0]            COL_data_out[1:0];
    wire            [DATA_WIDTH-1:0]            VEC_data_out[1:0];
    wire            [DATA_WIDTH-1:0]            RES_data_out[1:0];
    reg             [DATA_WIDTH-1:0]            ROW_data_in[1:0];
    reg             [DATA_WIDTH-1:0]            COL_data_in[1:0];
    reg             [DATA_WIDTH-1:0]            VEC_data_in[1:0];
    reg             [DATA_WIDTH-1:0]            RES_data_in[1:0];
    reg             [$clog2(N):0]               ROW_address[1:0];
    reg             [$clog2(N):0]               COL_address[1:0];
    reg             [$clog2(N):0]               VEC_address[1:0];
    reg             [$clog2(N)-1:0]             RES_address[1:0];
    reg                                         ROW_wr_en[1:0];
    reg                                         COL_wr_en[1:0];
    reg                                         VEC_wr_en[1:0];
    reg                                         RES_wr_en[1:0];
    // TMVP common
    reg                       	 		        start_TMVP;
    wire            [DATA_WIDTH-1:0]            m_axis_tdata_TMVP;
    wire                         		        m_axis_tvalid_TMVP; 
    reg             [DATA_WIDTH-1:0]            m_axis_tdata_TMVP_D, m_axis_tdata_TMVP_D2;
    reg                          		        m_axis_tvalid_TMVP_D, m_axis_tvalid_TMVP_D2; 
    reg     signed  [DATA_WIDTH-1:0]   	        data_vec_data_1_TMVP, data_vec_data_2_TMVP;
    reg                                   	    data_vec_valid_TMVP;  
    wire            [$clog2(N)-1:0]    	        address_vec_1_TMVP, address_vec_2_TMVP;
    wire                         				address_vec_valid_TMVP;
    reg                         				address_vec_valid_TMVP_D, address_vec_valid_TMVP_D2;
    wire                                        address_valid_TMVP;
    reg                                         address_valid_TMVP_D, address_valid_TMVP_D2;
    reg                                   	    data_valid_TMVP;  
    reg     signed	[DATA_WIDTH-1:0]   	        data_row_1_TMVP, data_row_2_TMVP;
    // TMVP_main
    reg     signed  [DATA_WIDTH-1:0]            data_col_1_TMVP, data_col_2_TMVP;
    wire            [$clog2(N)-1:0]  	        address_row_1_TMVP, address_row_2_TMVP, address_col_1_TMVP, address_col_2_TMVP;
    // TMVP Final Step
    wire            [$clog2(N)-1:0]     	    address_1_TMVP, address_2_TMVP;
    wire                                        address_1_isRow_TMVP, address_2_isRow_TMVP;
    reg                                         address_1_isRow_TMVP_D, address_2_isRow_TMVP_D;
    reg                                         address_1_isRow_TMVP_D2, address_2_isRow_TMVP_D2;
    //____________________Regs_and_Wires____________________

    //____________________FSM____________________
    /*
    FIRST_LOAD: 
    ROW:     1: A0, 2: A2
    COL:     1: A0(Reverse) for A2 use A0 Row data
    VEC:     1: B0, 2: B1 
    LAST_LOAD: 
    ROW:     1: A0(Reverse) for A1 use A0 Col data
    ROW:     1: A0, 2: A1
    VEC:     1: B0, 2: B1
    STEP_1: A0*(B0 + B1)        upper half 
    STEP_2: (A2 - A0)*B1        lower half
    STEP_3: (A1 - A0)*B0        upper half
    */
    always @(posedge clk) begin
        if (!reset) begin
            state                           <=  IDLE;
            start_TMVP                      <=  1'b0;
        end
        else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        state               <=  FIRST_LOAD;
                    end
                    start_TMVP              <=  1'b0;
                end 
                FIRST_LOAD: begin
                    if (load_done) begin
                        state               <=  STEP_1;
                        start_TMVP          <=  1'b1;
                    end
                    else begin
                        start_TMVP          <=  1'b0;
                    end
                end 
                STEP_1: begin
                    if (counter_in == N-1) begin
                        state               <=  STEP_2;
                        start_TMVP          <=  1'b1;
                    end
                    else begin
                        start_TMVP          <=  1'b0;
                    end
                end 
                STEP_2: begin
                    if (counter_out == N) begin
                        state               <=  LAST_LOAD;
                    end
                    start_TMVP              <=  1'b0;
                end 
                LAST_LOAD: begin
                    if (load_done) begin
                        state               <=  STEP_3;
                        start_TMVP          <=  1'b1;
                    end
                    else begin
                        start_TMVP          <=  1'b0;
                    end
                end
                STEP_3: begin
                    if (counter_out == N) begin
                        state               <=  IDLE;
                    end
                    start_TMVP              <=  1'b0;
                end
                default: begin
                    state                   <=  IDLE;
                end
            endcase
        end
    end
    always @(posedge clk) begin
        if (state == FIRST_LOAD ||  state == LAST_LOAD) begin
            if (counter_out ==  N) begin
                if (load_done) begin
                    counter_out             <=  0;
                end
                else begin
                    counter_out             <=  counter_out;
                end
            end
            else begin
                counter_out                 <=  counter_out + 1;
            end
        end
        else if (state == STEP_2 ||  state == STEP_3) begin
            if (counter_out ==  N) begin
                counter_out                 <=  0;
            end
            else if (m_axis_tvalid_TMVP_D2) begin
                counter_out                 <=  counter_out + 1;
            end
        end
        else begin
            counter_out                     <=  0;
        end 
    end
    always @(posedge clk) begin
        if (state == FIRST_LOAD ||  state == LAST_LOAD) begin
            if (data_valid) begin
                if (counter_in ==  N - 1) begin
                    counter_in              <=  0;
                end
                else begin
                    counter_in              <=  counter_in + 1;
                end
            end
        end
        else if (state == STEP_1 ||  state == STEP_2  ||  state == STEP_3) begin
            if (m_axis_tvalid_TMVP) begin
                if (counter_in ==  N - 1) begin
                    counter_in              <=  0;
                end
                else begin
                    counter_in              <=  counter_in + 1;
                end
            end
        end
        else begin
            counter_in                      <=  0;
        end
    end
    always @(posedge clk) begin
        if (state == FIRST_LOAD     ||      state == LAST_LOAD) begin
            load_done                       <=  (counter_in == (N-1)) && data_valid;
        end
        else begin
            load_done                       <=  1'b0;
        end 
    end
    //____________________FSM____________________

    //____________________LOADING____________________
    always @(posedge clk) begin
        case (state)
            FIRST_LOAD: begin
                address_row_1                   <=  counter_out;
                address_row_2                   <=  counter_out + N;
                address_col_1                   <=  N - counter_out;
                address_valid                   <=  !(counter_out == N);
                address_vec_valid               <=  !(counter_out == N);
            end
            LAST_LOAD: begin
                address_col_1                   <=  counter_out;
                address_col_2                   <=  counter_out + N;
                address_row_1                   <=  N - counter_out;
                address_valid                   <=  !(counter_out == N);
                address_vec_valid               <=  !(counter_out == N);
            end
            default begin
                address_valid                   <=  1'b0;
                address_vec_valid               <=  1'b0;
            end
        endcase
    end
    always @(posedge clk) begin
        address_vec_1                           <=  counter_out;
        address_vec_2                           <=  counter_out + N;
    end
    //____________________LOADING____________________

    //____________________RAM_ACCESS____________________
    generate
        if (N == (2 * TILE_SIZE)) begin
            //____________________LEAF____________________
            always @(posedge clk) begin // COL RAM ACCESS
                case (state)
                    FIRST_LOAD: begin
                        if (data_valid) begin
                            if (counter_in == 0) begin
                                // A0
                                COL_address[0]              <=  0;
                                COL_data_in[0]              <=  data_row_1;
                                // A2 - A0
                                COL_address[1]              <=  N;
                                COL_data_in[1]              <=  data_row_2 - data_row_1;
                            end
                            else begin
                                // A0
                                COL_address[0]              <=  N - counter_in;
                                COL_data_in[0]              <=  data_col_1;
                                // A2 - A0
                                COL_address[1]              <=  2*N - counter_in;
                                COL_data_in[1]              <=  data_row_1 - data_col_1;
                            end
                        end
                        COL_wr_en[0]                        <=  data_valid;
                        COL_wr_en[1]                        <=  data_valid;
                    end
                    LAST_LOAD: begin
                        if (data_valid) begin
                            // A1 - A0
                            COL_address[0]                  <=  counter_in;
                            COL_data_in[0]                  <=  data_col_2 - data_col_1;
                        end
                        COL_wr_en[0]                        <=  data_valid;
                        COL_wr_en[1]                        <=  1'b0;
                    end
                    STEP_2: begin
                        COL_wr_en[0]                        <=  1'b0;
                        COL_wr_en[1]                        <=  1'b0;
                        COL_address[0]                      <=  address_1_TMVP + N;
                        COL_address[1]                      <=  address_2_TMVP + N;                            
                    end
                    default: begin
                        COL_wr_en[0]                        <=  1'b0;
                        COL_wr_en[1]                        <=  1'b0;
                        COL_address[0]                      <=  {0, address_1_TMVP};
                        COL_address[1]                      <=  {0, address_2_TMVP};                            
                    end
                endcase
            end
            always @(posedge clk) begin // ROW RAM ACCESS
                case (state)
                    FIRST_LOAD: begin
                        if (data_valid) begin
                            // A0
                            ROW_address[0]                  <=  counter_in;
                            ROW_data_in[0]                  <=  data_row_1;
                            // A2 - A0
                            ROW_address[1]                  <=  N + counter_in;
                            ROW_data_in[1]                  <=  data_row_2 - data_row_1;
                        end
                        ROW_wr_en[0]                        <=  data_valid;
                        ROW_wr_en[1]                        <=  data_valid;
                    end
                    LAST_LOAD: begin
                        if (data_valid) begin
                            if (counter_in == 0) begin
                                // A1 - A0
                                ROW_address[0]              <=  0;
                                ROW_data_in[0]              <=  data_col_2 - data_col_1;
                            end
                            else begin
                                // A1 - A0
                                ROW_address[0]              <=  N - counter_in;
                                ROW_data_in[0]              <=  data_col_1 - data_row_1;
                            end
                        end
                        ROW_wr_en[0]                        <=  data_valid;
                        ROW_wr_en[1]                        <=  1'b0;
                    end
                    STEP_2: begin
                        ROW_wr_en[0]                        <=  1'b0;
                        ROW_wr_en[1]                        <=  1'b0;
                        ROW_address[1]                      <=  address_2_TMVP + N;
                        ROW_address[0]                      <=  address_1_TMVP + N;
                    end
                    default: begin
                        ROW_wr_en[0]                        <=  1'b0;
                        ROW_wr_en[1]                        <=  1'b0;
                        ROW_address[1]                      <=  {0, address_2_TMVP};
                        ROW_address[0]                      <=  {0, address_1_TMVP};
                    end
                endcase
            end 
            //____________________LEAF____________________
        end
        else begin
            //____________________TREE____________________
            always @(posedge clk) begin // COL RAM ACCESS
                case (state)
                    FIRST_LOAD: begin
                        if (data_valid) begin
                            if (counter_in == 0) begin
                                // A0
                                COL_address[0]              <=  0;
                                COL_data_in[0]              <=  data_row_1;
                                // A2 - A0
                                COL_address[1]              <=  N;
                                COL_data_in[1]              <=  data_row_2 - data_row_1;
                            end
                            else begin
                                // A0
                                COL_address[0]              <=  N - counter_in;
                                COL_data_in[0]              <=  data_col_1;
                                // A2 - A0
                                COL_address[1]              <=  2*N - counter_in;
                                COL_data_in[1]              <=  data_row_1 - data_col_1;
                            end
                        end
                        COL_wr_en[0]                        <=  data_valid;
                        COL_wr_en[1]                        <=  data_valid;
                    end
                    LAST_LOAD: begin
                        if (data_valid) begin
                            // A1 - A0
                            COL_address[0]                  <=  counter_in;
                            COL_data_in[0]                  <=  data_col_2 - data_col_1;
                        end
                        COL_wr_en[0]                        <=  data_valid;
                    end
                    STEP_2: begin
                        COL_wr_en[0]                        <=  1'b0;
                        COL_wr_en[1]                        <=  1'b0;
                        COL_address[0]                      <=  address_col_1_TMVP + N;
                        COL_address[1]                      <=  address_col_2_TMVP + N;
                    end
                    default: begin
                        COL_wr_en[0]                        <=  1'b0;
                        COL_wr_en[1]                        <=  1'b0;
                        COL_address[0]                      <=  {0, address_col_1_TMVP};
                        COL_address[1]                      <=  {0, address_col_2_TMVP};
                    end
                endcase
            end
            always @(posedge clk) begin // ROW RAM ACCESS
                case (state)
                    FIRST_LOAD: begin
                        if (data_valid) begin
                            // A0
                            ROW_address[0]                  <=  counter_in;
                            ROW_data_in[0]                  <=  data_row_1;
                            // A2 - A0
                            ROW_address[1]                  <=  N + counter_in;
                            ROW_data_in[1]                  <=  data_row_2 - data_row_1;
                        end
                        ROW_wr_en[0]                        <=  data_valid;
                        ROW_wr_en[1]                        <=  data_valid;
                    end
                    LAST_LOAD: begin
                        if (data_valid) begin
                            if (counter_in == 0) begin
                                // A1 - A0
                                ROW_address[0]              <=  0;
                                ROW_data_in[0]              <=  data_col_2 - data_col_1;
                            end
                            else begin
                                // A1 - A0
                                ROW_address[0]              <=  N - counter_in;
                                ROW_data_in[0]              <=  data_col_1 - data_row_1;
                            end
                        end
                        ROW_wr_en[0]                        <=  data_valid;
                    end
                    STEP_2: begin
                        ROW_wr_en[0]                        <=  1'b0;
                        ROW_wr_en[1]                        <=  1'b0;
                        ROW_address[0]                      <=  address_row_1_TMVP + N;
                        ROW_address[1]                      <=  address_row_2_TMVP + N;
                    end
                    default: begin
                        ROW_wr_en[0]                        <=  1'b0;
                        ROW_wr_en[1]                        <=  1'b0;
                        ROW_address[0]                      <=  {0, address_row_1_TMVP};
                        ROW_address[1]                      <=  {0, address_row_2_TMVP};
                    end
                endcase
            end
            //____________________TREE____________________
        end
    endgenerate
    always @(posedge clk) begin // VEC RAM ACCESS
        case (state)
            FIRST_LOAD: begin
                if (data_vec_valid) begin
                    // B0 + B1
                    VEC_address[0]                          <=  counter_in;
                    VEC_data_in[0]                          <=  data_vec_data_1 + data_vec_data_2;
                    // B1
                    VEC_address[1]                          <=  N + counter_in;
                    VEC_data_in[1]                          <=  data_vec_data_2;
                end
                VEC_wr_en[0]                                <=  data_vec_valid;
                VEC_wr_en[1]                                <=  data_vec_valid;
            end
            LAST_LOAD: begin
                if (data_vec_valid) begin
                    // B0
                    VEC_address[0]                          <=  counter_in;
                    VEC_data_in[0]                          <=  data_vec_data_1;
                end
                VEC_wr_en[0]                                <=  data_vec_valid;
            end
            STEP_2: begin
                VEC_wr_en[0]                                <=  1'b0;
                VEC_wr_en[1]                                <=  1'b0;
                VEC_address[0]                              <=  {0, address_vec_1_TMVP} + N;
                VEC_address[1]                              <=  {0, address_vec_2_TMVP} + N;
            end
            default: begin
                VEC_wr_en[0]                                <=  1'b0;
                VEC_wr_en[1]                                <=  1'b0;
                VEC_address[0]                              <=  {1'b0, address_vec_1_TMVP};
                VEC_address[1]                              <=  {1'b0, address_vec_2_TMVP};
            end
        endcase
    end
    //____________________RAM_ACCESS____________________

    //____________________WRITING_DATA____________________
    generate
        if (N == (2 * TILE_SIZE)) begin
            //____________________LEAF____________________
            always @(posedge clk) begin
                address_1_isRow_TMVP_D                      <=  address_1_isRow_TMVP;
                address_2_isRow_TMVP_D                      <=  address_2_isRow_TMVP;
                address_1_isRow_TMVP_D2                     <=  address_1_isRow_TMVP_D;
                address_2_isRow_TMVP_D2                     <=  address_2_isRow_TMVP_D;
            end
            always @(posedge clk) begin
                if (address_1_isRow_TMVP_D2) begin
                    data_row_1_TMVP                         <=  ROW_data_out[0];
                end
                else begin
                    data_row_1_TMVP                         <=  COL_data_out[0];
                end
                if (address_2_isRow_TMVP_D2) begin
                    data_row_2_TMVP                         <=  ROW_data_out[1];
                end
                else begin
                    data_row_2_TMVP                         <=  COL_data_out[1];
                end
            end
            //____________________LEAF____________________
        end
        else begin
            //____________________TREE____________________
            always @(posedge clk) begin
                data_row_1_TMVP                             <=  ROW_data_out[0];
                data_row_2_TMVP                             <=  ROW_data_out[1];
                data_col_1_TMVP                             <=  COL_data_out[0];
                data_col_2_TMVP                             <=  COL_data_out[1];
            end
            //____________________TREE____________________
        end
    endgenerate
    always @(posedge clk) begin
        address_vec_valid_TMVP_D                            <=  address_vec_valid_TMVP;
        address_vec_valid_TMVP_D2                           <=  address_vec_valid_TMVP_D;
        data_vec_data_1_TMVP                                <=  VEC_data_out[0];
        data_vec_data_2_TMVP                                <=  VEC_data_out[1];
        data_vec_valid_TMVP                                 <=  address_vec_valid_TMVP_D2;
    end
    always @(posedge clk) begin
        address_valid_TMVP_D                                <=  address_valid_TMVP;
        address_valid_TMVP_D2                               <=  address_valid_TMVP_D;
        data_valid_TMVP                                     <=  address_valid_TMVP_D2;
    end
    //____________________WRITING_DATA____________________

    //____________________NEXT_TMVP____________________
    generate
        if (N == (2 * TILE_SIZE)) begin
            //____________________LEAF____________________
            TMVP2 #(.N(N), .DATA_WIDTH(DATA_WIDTH)) TMVP_final_inst
            (.clk(clk), .reset(reset), 
            .start(start_TMVP),
            .data_row_data_1(data_row_1_TMVP), .data_row_data_2(data_row_2_TMVP), .data_row_valid(data_valid_TMVP),
            .data_vec_data_1(data_vec_data_1_TMVP), .data_vec_data_2(data_vec_data_2_TMVP), .data_vec_valid(data_vec_valid_TMVP),
            .address_1(address_1_TMVP), .address_1_isRow(address_1_isRow_TMVP),
            .address_2(address_2_TMVP), .address_2_isRow(address_2_isRow_TMVP),
            .address_row_valid(address_valid_TMVP),
            .address_vec_1(address_vec_1_TMVP), .address_vec_2(address_vec_2_TMVP), .address_vec_valid(address_vec_valid_TMVP),
            .m_axis_tdata(m_axis_tdata_TMVP), .m_axis_tvalid(m_axis_tvalid_TMVP));    
            //____________________LEAF____________________
        end
        else begin
            //____________________TREE____________________
            TMVP2_main  #(.N(N/2), .DATA_WIDTH(DATA_WIDTH), .TILE_SIZE(TILE_SIZE))  TMVP_inst
            (.clk(clk), .reset(reset), 
            .start(start_TMVP), 
            .address_row_1(address_row_1_TMVP), .address_row_2(address_row_2_TMVP), .address_col_1(address_col_1_TMVP), .address_col_2(address_col_2_TMVP), .address_valid(address_valid_TMVP),
            .data_row_1(data_row_1_TMVP), .data_row_2(data_row_2_TMVP), .data_col_1(data_col_1_TMVP), .data_col_2(data_col_2_TMVP), .data_valid(data_valid_TMVP),
            .address_vec_1(address_vec_1_TMVP), .address_vec_2(address_vec_2_TMVP), .address_vec_valid(address_vec_valid_TMVP),
            .data_vec_data_1(data_vec_data_1_TMVP), .data_vec_data_2(data_vec_data_2_TMVP), .data_vec_valid(data_vec_valid_TMVP),
            .m_axis_tdata(m_axis_tdata_TMVP), .m_axis_tvalid(m_axis_tvalid_TMVP)
            );            
            //____________________TREE____________________
        end
    endgenerate
    //____________________NEXT_TMVP____________________

    //____________________DATA_RAM____________________
    dual_port_ram_TMVP #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH($clog2(N)+1)) ROW_RAM (
        .clk(clk),
        .data_a(ROW_data_in[0]), .addr_a(ROW_address[0]), .we_a(ROW_wr_en[0]), .q_a(ROW_data_out[0]), 
        .data_b(ROW_data_in[1]), .addr_b(ROW_address[1]), .we_b(ROW_wr_en[1]), .q_b(ROW_data_out[1])
    );
    dual_port_ram_TMVP #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH($clog2(N)+1)) COL_RAM (
        .clk(clk),
        .data_a(COL_data_in[0]), .addr_a(COL_address[0]), .we_a(COL_wr_en[0]), .q_a(COL_data_out[0]), 
        .data_b(COL_data_in[1]), .addr_b(COL_address[1]), .we_b(COL_wr_en[1]), .q_b(COL_data_out[1])
    );
    dual_port_ram_TMVP #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH($clog2(N)+1)) VEC_RAM (
        .clk(clk),
        .data_a(VEC_data_in[0]), .addr_a(VEC_address[0]), .we_a(VEC_wr_en[0]), .q_a(VEC_data_out[0]), 
        .data_b(VEC_data_in[1]), .addr_b(VEC_address[1]), .we_b(VEC_wr_en[1]), .q_b(VEC_data_out[1])
    );
    //____________________DATA_RAM____________________

    //____________________RESULT____________________
    always @(posedge clk) begin
        m_axis_tdata_TMVP_D                     <=  m_axis_tdata_TMVP;
        m_axis_tdata_TMVP_D2                    <=  m_axis_tdata_TMVP_D;
        if (state   ==  STEP_2      ||      state   ==  STEP_3) begin
            m_axis_tvalid_TMVP_D                <=  m_axis_tvalid_TMVP;
            m_axis_tvalid_TMVP_D2               <=  m_axis_tvalid_TMVP_D;
        end
        else begin
            m_axis_tvalid_TMVP_D                <=  1'b0;
            m_axis_tvalid_TMVP_D2               <=  1'b0;
        end
    end
    always @(posedge clk) begin // Write A0*(B0+B1)
        if (state   ==  STEP_1) begin
            RES_address[0]                      <=  counter_in;
            RES_data_in[0]                      <=  m_axis_tdata_TMVP;
            RES_wr_en[0]                        <=  m_axis_tvalid_TMVP;
        end
        else begin
            RES_wr_en[0]                        <=  m_axis_tvalid_TMVP;
        end
    end
    always @(posedge clk) begin
        if (state   ==  STEP_2      ||      state   ==  STEP_3) begin
            RES_address[1]                      <=  counter_in;
            m_axis_tdata                        <=  m_axis_tdata_TMVP_D2    +   RES_data_out[1];
            m_axis_tvalid                       <=  m_axis_tvalid_TMVP_D2;
        end
        else begin
            m_axis_tvalid                       <=  1'b0;
        end
        RES_wr_en[1]                            <=  1'b0;
    end
    dual_port_ram_TMVP #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH($clog2(N))) RESULT_RAM (
        .clk(clk),
        .data_a(RES_data_in[0]), .addr_a(RES_address[0]), .we_a(RES_wr_en[0]), .q_a(RES_data_out[0]), 
        .data_b(RES_data_in[1]), .addr_b(RES_address[1]), .we_b(RES_wr_en[1]), .q_b(RES_data_out[1])
    //____________________RESULT____________________
    );
endmodule
