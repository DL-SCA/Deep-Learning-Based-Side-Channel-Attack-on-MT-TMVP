module TMVP3_main    
    #(parameter N = 240, DATA_WIDTH = 7, TILE_SIZE = 10)
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
    output  reg     [$clog2(N)+1:0]     		address_row_1, address_row_2, address_col_1, address_col_2;
    output  reg                         		address_valid;
    output  reg     [$clog2(N)+1:0]     		address_vec_1, address_vec_2;
    output  reg          						address_vec_valid;
    // maxis
    output  reg     [DATA_WIDTH-1:0]            m_axis_tdata;
    output  reg                         		m_axis_tvalid;    
    //____________________IO____________________

    //____________________Regs_and_Wires____________________
    // FSM
    localparam IDLE = 4'b0000;
    localparam LOAD_1 = 4'b0001;
    localparam MULT_1 = 4'b0011;
    localparam LOAD_2 = 4'b0010;
    localparam MULT_2 = 4'b0110;
    localparam MULT_3 = 4'b0111;
    localparam MULT_4 = 4'b0101;
    localparam LOAD_3 = 4'b0100;
    localparam MULT_5 = 4'b1100;
    localparam MULT_6 = 4'b1110;
    reg             [3:0]                       state;
    reg                                         load_done;
    reg             [$clog2(N):0]               counter_in, counter_in_D, counter_in_D2, counter_in_D3;
    reg             [$clog2(N):0]               counter_out, counter_out_D, counter_out_D2, counter_out_D3;
    // RAMs
    reg     signed  [DATA_WIDTH-1:0]            ROW_data_inside_RAM[1:0];
    reg     signed  [DATA_WIDTH-1:0]            COL_data_inside_RAM[1:0];
    wire    signed  [DATA_WIDTH-1:0]            ROW_data_out_1[1:0];
    wire    signed  [DATA_WIDTH-1:0]            COL_data_out_1[1:0];
    wire    signed  [DATA_WIDTH-1:0]            ROW_data_out_2[1:0];
    wire    signed  [DATA_WIDTH-1:0]            COL_data_out_2[1:0];
    wire    signed  [DATA_WIDTH-1:0]            VEC_data_out[1:0];
    wire    signed  [DATA_WIDTH-1:0]            RES_data_out[1:0];
    reg     signed  [DATA_WIDTH-1:0]            RES_data_out_D[1:0];
    reg     signed  [DATA_WIDTH-1:0]            ROW_data_in_1[1:0];
    reg     signed  [DATA_WIDTH-1:0]            COL_data_in_1[1:0];
    reg     signed  [DATA_WIDTH-1:0]            ROW_data_in_2[1:0];
    reg     signed  [DATA_WIDTH-1:0]            COL_data_in_2[1:0];
    reg     signed  [DATA_WIDTH-1:0]            VEC_data_in[1:0];
    reg     signed  [DATA_WIDTH-1:0]            RES_data_in[1:0];
    reg             [$clog2(3*N)-1:0]           ROW_address_1[1:0];
    reg             [$clog2(3*N)-1:0]           COL_address_1[1:0];
    reg             [$clog2(3*N)-1:0]           ROW_address_2[1:0];
    reg             [$clog2(3*N)-1:0]           COL_address_2[1:0];
    reg             [$clog2(3*N)-1:0]           VEC_address[1:0];
    reg             [$clog2(3*N)-1:0]           RES_address[1:0];
    reg                                         ROW_wr_en_1[1:0];
    reg                                         COL_wr_en_1[1:0];
    reg                                         ROW_wr_en_2[1:0];
    reg                                         COL_wr_en_2[1:0];
    reg                                         VEC_wr_en[1:0];
    reg                                         RES_wr_en[1:0];
    // TMVP common
    reg                       	 		        start_TMVP;
    wire    signed  [DATA_WIDTH-1:0]            m_axis_tdata_TMVP;
    wire                         		        m_axis_tvalid_TMVP; 
    reg     signed  [DATA_WIDTH-1:0]            m_axis_tdata_TMVP_D, m_axis_tdata_TMVP_D2, m_axis_tdata_TMVP_D3;
    reg                          		        m_axis_tvalid_TMVP_D, m_axis_tvalid_TMVP_D2, m_axis_tvalid_TMVP_D3; 
    reg     signed  [DATA_WIDTH-1:0]   	        data_vec_data_1_TMVP, data_vec_data_2_TMVP;
    reg                                   	    data_vec_valid_TMVP;  
    wire            [$clog2(N)-1:0]    	        address_vec_1_TMVP, address_vec_2_TMVP;
    wire                         				address_vec_valid_TMVP;
    reg                         				address_vec_valid_TMVP_D, address_vec_valid_TMVP_D2;
    wire                                        address_valid_TMVP;
    reg                                         address_valid_TMVP_D, address_valid_TMVP_D2;
    reg                                   	    data_valid_TMVP;  
    reg     signed	[DATA_WIDTH-1:0]   	        data_row_1_TMVP, data_row_2_TMVP;
    reg     signed  [DATA_WIDTH-1:0]            data_col_1_TMVP, data_col_2_TMVP;
    wire            [$clog2(N)-1:0]  	        address_row_1_TMVP, address_row_2_TMVP, address_col_1_TMVP, address_col_2_TMVP;
    // RESULT
    reg     signed  [DATA_WIDTH-1:0]            m_axis_tdata_TEMP;
    //____________________Regs_and_Wires____________________

    //____________________FSM____________________
    /*
    LOAD_1: 
    MAT:     1: A2, 2: A1
    VEC:     1: B1, 2: B2 
    LOAD_2: 
    MAT:     1: A3, 2: A0
    VEC:     1: B0, 2: B2
    LOAD_3:
    MAT:     1: A4, 2: -
    VEC:     1: B0, 2: B1
    MULT_1: A1*(B1 - B2)         
    MULT_2: (A0 + A1 + A2)*B2        
    MULT_3: (A1 + A2 + A3)*B1                
    MULT_4: A2*(B0 - B2)         
    MULT_5: A3*(B0 - B1)         
    MULT_6: (A2 + A3 + A4)*B0                
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
                        state               <=  LOAD_1;
                    end
                    start_TMVP              <=  1'b0;
                end 
                LOAD_1: begin
                    if (load_done) begin
                        state               <=  MULT_1;
                        start_TMVP          <=  1'b1;
                    end
                    else begin
                        start_TMVP          <=  1'b0;
                    end
                end 
                MULT_1: begin
                    if (counter_in_D == N) begin
                        state               <=  LOAD_2;
                    end
                    start_TMVP              <=  1'b0;
                end 
                LOAD_2: begin
                    if (load_done) begin
                        state               <=  MULT_2;
                        start_TMVP          <=  1'b1;
                    end
                    else begin
                        start_TMVP          <=  1'b0;
                    end
                end
                MULT_2: begin
                    if (counter_in_D3 == N) begin
                        state               <=  MULT_3;
                        start_TMVP          <=  1'b1;
                    end
                    else begin
                        start_TMVP          <=  1'b0;
                    end
                end 
                MULT_3: begin
                    if (counter_in_D3 == N) begin
                        state               <=  MULT_4;
                        start_TMVP          <=  1'b1;
                    end
                    else begin
                        start_TMVP          <=  1'b0;
                    end
                end 
                MULT_4: begin
                    if (counter_out == N) begin
                        state               <=  LOAD_3;
                    end
                    start_TMVP              <=  1'b0;
                end 
                LOAD_3: begin
                    if (load_done) begin
                        state               <=  MULT_5;
                        start_TMVP          <=  1'b1;
                    end
                    else begin
                        start_TMVP          <=  1'b0;
                    end
                end
                MULT_5: begin
                    if (counter_out == N) begin
                        state               <=  MULT_6;
                        start_TMVP          <=  1'b1;
                    end
                    else begin
                        start_TMVP          <=  1'b0;
                    end
                end 
                MULT_6: begin
                    if (counter_out_D3 == N     &&      counter_out != 0) begin
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
        if (state == LOAD_1 ||  state == LOAD_2 ||  state == LOAD_3) begin
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
        else if (state == MULT_4 ||  state == MULT_5) begin
            if (counter_out ==  N) begin
                counter_out                 <=  0;
            end
            else if (m_axis_tvalid_TMVP_D2) begin
                counter_out                 <=  counter_out + 1;
            end
        end
        else if (state == MULT_6) begin
            if (m_axis_tvalid_TMVP_D2) begin
                counter_out                 <=  counter_out + 1;
            end
        end
        else begin
            counter_out                     <=  0;
        end
        counter_out_D                       <=  counter_out; 
        counter_out_D2                      <=  counter_out_D; 
        counter_out_D3                      <=  counter_out_D2; 
    end
    always @(posedge clk) begin
        if (state == LOAD_1 ||  state == LOAD_2 ||  state == LOAD_3) begin
            if (data_valid) begin
                if (counter_in ==  N - 1) begin
                    counter_in              <=  0;
                end
                else begin
                    counter_in              <=  counter_in + 1;
                end
            end
        end
        else if (state == MULT_1 ||  state == MULT_2 || state == MULT_3 ||  state == MULT_4 ||  state == MULT_5 ||  state == MULT_6) begin
            if (counter_in == N) begin
                counter_in                  <=  0;
            end
            if (m_axis_tvalid_TMVP) begin
                counter_in                  <=  counter_in + 1;
            end
        end
        else begin
            counter_in                      <=  0;
        end
        counter_in_D                        <=  counter_in;
        counter_in_D2                       <=  counter_in_D;
        counter_in_D3                       <=  counter_in_D2;
    end
    always @(posedge clk) begin
        if (state == LOAD_1 || state == LOAD_2 || state == LOAD_3) begin
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
            LOAD_1: begin // 1:A2, 2:A1
                address_row_1                   <=  counter_out; // ROW A2, COL A1(Reverse)
                address_row_2                   <=  counter_out + N; // ROW A1
                address_col_1                   <=  N - counter_out; // COL A2(Reverse)
                address_vec_1                   <=  counter_out + N; // B1
                address_vec_2                   <=  counter_out + 2*N; // B2
                address_valid                   <=  !(counter_out == N);
                address_vec_valid               <=  !(counter_out == N);
            end
            LOAD_2: begin // row:A0, col:A3
                address_row_1                   <=  counter_out + 2*N; // ROW A0
                address_row_2                   <=  2*N - counter_out; // COL A0 
                address_col_1                   <=  N - counter_out; // ROW A3
                address_col_2                   <=  counter_out + N; // COL A3
                address_vec_1                   <=  counter_out; // B0
                address_vec_2                   <=  counter_out + 2*N; // B2
                address_valid                   <=  !(counter_out == N);
                address_vec_valid               <=  !(counter_out == N);
            end
            LOAD_3: begin // col:A4
                address_col_1                   <=  2*N - counter_out; // ROW A4
                address_col_2                   <=  counter_out + 2*N; // COL A4
                address_vec_1                   <=  counter_out; // B0
                address_vec_2                   <=  counter_out + N; // B1
                address_valid                   <=  !(counter_out == N);
                address_vec_valid               <=  !(counter_out == N);
            end
            default begin
                address_valid                   <=  1'b0;
                address_vec_valid               <=  1'b0;
            end
        endcase
    end
    //____________________LOADING____________________

    //____________________RAM_ACCESS____________________
    always @(posedge clk) begin // COL RAM ACCESS
        case (state)
            LOAD_1: begin // 1:A2, 2:A1
                if (data_valid) begin
                    if (counter_in == 0) begin
                        // A1 + A2
                        COL_address_1[0]            <=  0;
                        COL_data_in_1[0]            <=  data_row_1 + data_row_2;
                        // A1
                        COL_address_1[1]            <=  N;
                        COL_data_in_1[1]            <=  data_row_2;
                        // A2
                        COL_address_2[0]            <=  0;
                        COL_data_in_2[0]            <=  data_row_1;
                    end
                    else begin
                        // A1 + A2
                        COL_address_1[0]            <=  N - counter_in;
                        COL_data_in_1[0]            <=  data_col_1 + data_row_1;
                        // A1
                        COL_address_1[1]            <=  2*N - counter_in;
                        COL_data_in_1[1]            <=  data_row_1;
                        // A2
                        COL_address_2[0]            <=  N - counter_in;
                        COL_data_in_2[0]            <=  data_col_1;
                    end
                end
                COL_wr_en_1[0]                      <=  data_valid;
                COL_wr_en_1[1]                      <=  data_valid;
                COL_wr_en_2[0]                      <=  data_valid;
                COL_wr_en_2[1]                      <=  1'b0;
            end
            LOAD_2: begin // row:A0, col:A3
                // A1 + A2
                COL_address_1[0]                    <=  counter_out_D;
                if (data_valid) begin
                    // A3
                    COL_address_1[1]                <=  counter_in + N;
                    COL_data_in_1[1]                <=  data_col_2;
                    // A0 + A1 + A2
                    COL_address_2[0]                <=  counter_in + N;
                    COL_data_in_2[0]                <=  COL_data_inside_RAM[0] + data_row_2;
                    // A1 + A2 + A3
                    COL_address_2[1]                <=  counter_in + 2*N;
                    COL_data_in_2[1]                <=  COL_data_inside_RAM[0] + data_col_2;
                end
                COL_wr_en_1[0]                      <=  1'b0;
                COL_wr_en_1[1]                      <=  data_valid;
                COL_wr_en_2[0]                      <=  data_valid;
                COL_wr_en_2[1]                      <=  data_valid;
            end
            LOAD_3: begin // col: A4
                // A3
                COL_address_1[1]                    <=  counter_out_D + N;
                // A2
                COL_address_2[0]                    <=  counter_out_D;
                if (data_valid) begin
                    // A2 + A3 + A4
                    COL_address_1[0]                <=  counter_in;
                    COL_data_in_1[0]                <=  COL_data_inside_RAM[0] + COL_data_inside_RAM[1] + data_col_2;
                end
                COL_wr_en_1[0]                      <=  data_valid;
                COL_wr_en_1[1]                      <=  1'b0;
                COL_wr_en_2[0]                      <=  1'b0;
                COL_wr_en_2[1]                      <=  1'b0;
            end
            MULT_1, MULT_5: begin
                COL_wr_en_1[0]                      <=  1'b0;
                COL_wr_en_1[1]                      <=  1'b0;
                COL_wr_en_2[0]                      <=  1'b0;
                COL_wr_en_2[1]                      <=  1'b0;
                COL_address_1[0]                    <=  {0, address_col_1_TMVP} + N;
                COL_address_1[1]                    <=  {0, address_col_2_TMVP} + N;
            end
            MULT_2: begin
                COL_wr_en_1[0]                      <=  1'b0;
                COL_wr_en_1[1]                      <=  1'b0;
                COL_wr_en_2[0]                      <=  1'b0;
                COL_wr_en_2[1]                      <=  1'b0;
                COL_address_2[0]                    <=  {0, address_col_1_TMVP} + N;
                COL_address_2[1]                    <=  {0, address_col_2_TMVP} + N;
            end
            MULT_3: begin
                COL_wr_en_1[0]                      <=  1'b0;
                COL_wr_en_1[1]                      <=  1'b0;
                COL_wr_en_2[0]                      <=  1'b0;
                COL_wr_en_2[1]                      <=  1'b0;
                COL_address_2[0]                    <=  {0, address_col_1_TMVP} + 2*N;
                COL_address_2[1]                    <=  {0, address_col_2_TMVP} + 2*N;
            end
            MULT_4: begin
                COL_wr_en_1[0]                      <=  1'b0;
                COL_wr_en_1[1]                      <=  1'b0;
                COL_wr_en_2[0]                      <=  1'b0;
                COL_wr_en_2[1]                      <=  1'b0;
                COL_address_2[0]                    <=  {0, address_col_1_TMVP};
                COL_address_2[1]                    <=  {0, address_col_2_TMVP};
            end
            MULT_6: begin
                COL_wr_en_1[0]                      <=  1'b0;
                COL_wr_en_1[1]                      <=  1'b0;
                COL_wr_en_2[0]                      <=  1'b0;
                COL_wr_en_2[1]                      <=  1'b0;
                COL_address_1[0]                    <=  {0, address_col_1_TMVP};
                COL_address_1[1]                    <=  {0, address_col_2_TMVP};
            end
            default: begin
                COL_wr_en_1[0]                      <=  1'b0;
                COL_wr_en_1[1]                      <=  1'b0;
                COL_wr_en_2[0]                      <=  1'b0;
                COL_wr_en_2[1]                      <=  1'b0;
            end
        endcase
    end
    always @(posedge clk) begin // COL RAM ACCESS
        case (state)
            LOAD_1: begin // 1:A2, 2:A1
                if (data_valid) begin
                    // A1 + A2
                    ROW_address_1[0]                <=  counter_in;
                    ROW_data_in_1[0]                <=  data_row_1 + data_row_2;
                    // A1
                    ROW_address_1[1]                <=  counter_in + N;
                    ROW_data_in_1[1]                <=  data_row_2;
                    // A2
                    ROW_address_2[0]                <=  counter_in;
                    ROW_data_in_2[0]                <=  data_row_1;
                end
                ROW_wr_en_1[0]                      <=  data_valid;
                ROW_wr_en_1[1]                      <=  data_valid;
                ROW_wr_en_2[0]                      <=  data_valid;
                ROW_wr_en_2[1]                      <=  1'b0;
            end
            LOAD_2: begin // row:A0, col:A3
                // A1 + A2
                ROW_address_1[0]                    <=  counter_out_D;
                if (data_valid) begin
                    // A3
                    ROW_address_1[1]                <=  counter_in + N;
                    ROW_data_in_1[1]                <=  data_col_1;
                    // A0 + A1 + A2
                    ROW_address_2[0]                <=  counter_in + N;
                    ROW_data_in_2[0]                <=  ROW_data_inside_RAM[0] + data_row_1;
                    // A1 + A2 + A3
                    ROW_address_2[1]                <=  counter_in + 2*N;
                    ROW_data_in_2[1]                <=  ROW_data_inside_RAM[0] + data_col_1;
                end
                ROW_wr_en_1[0]                      <=  1'b0;
                ROW_wr_en_1[1]                      <=  data_valid;
                ROW_wr_en_2[0]                      <=  data_valid;
                ROW_wr_en_2[1]                      <=  data_valid;
            end
            LOAD_3: begin // col: A4
                // A3
                ROW_address_1[1]                    <=  counter_out_D + N;
                // A2
                ROW_address_2[0]                    <=  counter_out_D;
                if (data_valid) begin
                    // A2 + A3 + A4
                    ROW_address_1[0]                <=  counter_in;
                    ROW_data_in_1[0]                <=  ROW_data_inside_RAM[0] + ROW_data_inside_RAM[1] + data_col_1;
                end
                ROW_wr_en_1[0]                      <=  data_valid;
                ROW_wr_en_1[1]                      <=  1'b0;
                ROW_wr_en_2[0]                      <=  1'b0;
                ROW_wr_en_2[1]                      <=  1'b0;
            end
            MULT_1, MULT_5: begin
                ROW_wr_en_1[0]                      <=  1'b0;
                ROW_wr_en_1[1]                      <=  1'b0;
                ROW_wr_en_2[0]                      <=  1'b0;
                ROW_wr_en_2[1]                      <=  1'b0;
                ROW_address_1[0]                    <=  {0, address_row_1_TMVP} + N;
                ROW_address_1[1]                    <=  {0, address_row_2_TMVP} + N;
            end
            MULT_2: begin
                ROW_wr_en_1[0]                      <=  1'b0;
                ROW_wr_en_1[1]                      <=  1'b0;
                ROW_wr_en_2[0]                      <=  1'b0;
                ROW_wr_en_2[1]                      <=  1'b0;
                ROW_address_2[0]                    <=  {0, address_row_1_TMVP} + N;
                ROW_address_2[1]                    <=  {0, address_row_2_TMVP} + N;
            end
            MULT_3: begin
                ROW_wr_en_1[0]                      <=  1'b0;
                ROW_wr_en_1[1]                      <=  1'b0;
                ROW_wr_en_2[0]                      <=  1'b0;
                ROW_wr_en_2[1]                      <=  1'b0;
                ROW_address_2[0]                    <=  {0, address_row_1_TMVP} + 2*N;
                ROW_address_2[1]                    <=  {0, address_row_2_TMVP} + 2*N;
            end
            MULT_4: begin
                ROW_wr_en_1[0]                      <=  1'b0;
                ROW_wr_en_1[1]                      <=  1'b0;
                ROW_wr_en_2[0]                      <=  1'b0;
                ROW_wr_en_2[1]                      <=  1'b0;
                ROW_address_2[0]                    <=  {0, address_row_1_TMVP};
                ROW_address_2[1]                    <=  {0, address_row_2_TMVP};
            end
            MULT_6: begin
                ROW_wr_en_1[0]                      <=  1'b0;
                ROW_wr_en_1[1]                      <=  1'b0;
                ROW_wr_en_2[0]                      <=  1'b0;
                ROW_wr_en_2[1]                      <=  1'b0;
                ROW_address_1[0]                    <=  {0, address_row_1_TMVP};
                ROW_address_1[1]                    <=  {0, address_row_2_TMVP};
            end
            default: begin
                ROW_wr_en_1[0]                      <=  1'b0;
                ROW_wr_en_1[1]                      <=  1'b0;
                ROW_wr_en_2[0]                      <=  1'b0;
                ROW_wr_en_2[1]                      <=  1'b0;
            end
        endcase
    end
    always @(posedge clk) begin // VEC RAM ACCESS
        case (state)
            LOAD_1: begin // 1:B1, 2:B2
                if (data_vec_valid) begin
                    // B1 - B2
                    VEC_address[0]                  <=  counter_in;
                    VEC_data_in[0]                  <=  data_vec_data_1 - data_vec_data_2;
                    // B1
                    VEC_address[1]                  <=  N + counter_in;
                    VEC_data_in[1]                  <=  data_vec_data_1;
                end
                VEC_wr_en[0]                        <=  data_vec_valid;
                VEC_wr_en[1]                        <=  data_vec_valid;
            end
            LOAD_2: begin // 1:B0, 2:B2
                if (data_vec_valid) begin
                    // B0 - B2
                    VEC_address[0]                  <=  counter_in;
                    VEC_data_in[0]                  <=  data_vec_data_1 - data_vec_data_2;
                    // B2
                    VEC_address[1]                  <=  2*N + counter_in;
                    VEC_data_in[1]                  <=  data_vec_data_2;
                end
                VEC_wr_en[0]                        <=  data_vec_valid;
                VEC_wr_en[1]                        <=  data_vec_valid;
            end
            LOAD_3: begin // 1:B0, 2:B1
                if (data_vec_valid) begin
                    // B0 - B1
                    VEC_address[0]                  <=  counter_in;
                    VEC_data_in[0]                  <=  data_vec_data_1 - data_vec_data_2;
                    // B2
                    VEC_address[1]                  <=  N + counter_in;
                    VEC_data_in[1]                  <=  data_vec_data_1;
                end
                VEC_wr_en[0]                        <=  data_vec_valid;
                VEC_wr_en[1]                        <=  data_vec_valid;
            end
            MULT_1, MULT_4, MULT_5: begin
                VEC_wr_en[0]                                <=  1'b0;
                VEC_wr_en[1]                                <=  1'b0;
                VEC_address[0]                              <=  {0, address_vec_1_TMVP};
                VEC_address[1]                              <=  {0, address_vec_2_TMVP};
            end
            MULT_3, MULT_6: begin
                VEC_wr_en[0]                                <=  1'b0;
                VEC_wr_en[1]                                <=  1'b0;
                VEC_address[0]                              <=  {0, address_vec_1_TMVP} + N;
                VEC_address[1]                              <=  {0, address_vec_2_TMVP} + N;
            end
            MULT_2: begin
                VEC_wr_en[0]                                <=  1'b0;
                VEC_wr_en[1]                                <=  1'b0;
                VEC_address[0]                              <=  {0, address_vec_1_TMVP} + 2*N;
                VEC_address[1]                              <=  {0, address_vec_2_TMVP} + 2*N;
            end
            default: begin
                VEC_wr_en[0]                                <=  1'b0;
                VEC_wr_en[1]                                <=  1'b0;
            end
        endcase
    end
    //____________________RAM_ACCESS____________________

    //____________________WRITING_DATA____________________
    always @(posedge clk) begin
        if (state == MULT_1 ||  state == MULT_5 ||  state == MULT_6) begin
            data_row_1_TMVP                                 <=  ROW_data_out_1[0];
            data_row_2_TMVP                                 <=  ROW_data_out_1[1];
            data_col_1_TMVP                                 <=  COL_data_out_1[0];
            data_col_2_TMVP                                 <=  COL_data_out_1[1];
        end
        else if (state == MULT_2 ||  state == MULT_3 ||  state == MULT_4) begin
            data_row_1_TMVP                                 <=  ROW_data_out_2[0];
            data_row_2_TMVP                                 <=  ROW_data_out_2[1];
            data_col_1_TMVP                                 <=  COL_data_out_2[0];
            data_col_2_TMVP                                 <=  COL_data_out_2[1];
        end
        if (state == LOAD_2) begin
            COL_data_inside_RAM[0]                          <=  COL_data_out_1[0];
            ROW_data_inside_RAM[0]                          <=  ROW_data_out_1[0];
        end
        else if (state == LOAD_3) begin
            COL_data_inside_RAM[0]                          <=  COL_data_out_2[0];
            ROW_data_inside_RAM[0]                          <=  ROW_data_out_2[0];
            COL_data_inside_RAM[1]                          <=  COL_data_out_1[1];
            ROW_data_inside_RAM[1]                          <=  ROW_data_out_1[1];
        end
    end
    always @(posedge clk) begin
        address_vec_valid_TMVP_D                            <=  address_vec_valid_TMVP;
        address_vec_valid_TMVP_D2                           <=  address_vec_valid_TMVP_D;
        data_vec_valid_TMVP                                 <=  address_vec_valid_TMVP_D2;
        data_vec_data_1_TMVP                                <=  VEC_data_out[0];
        data_vec_data_2_TMVP                                <=  VEC_data_out[1];
    end
    always @(posedge clk) begin
        address_valid_TMVP_D                                <=  address_valid_TMVP;
        address_valid_TMVP_D2                               <=  address_valid_TMVP_D;
        data_valid_TMVP                                     <=  address_valid_TMVP_D2;
    end
    //____________________WRITING_DATA____________________

    //____________________NEXT_TMVP____________________
    generate
        if (N == 240 || N == 288) begin
            TMVP3_main  #(.N(N/3), .DATA_WIDTH(DATA_WIDTH), .TILE_SIZE(TILE_SIZE))  TMVP_inst
            (.clk(clk), .reset(reset), 
            .start(start_TMVP), 
            .address_row_1(address_row_1_TMVP), .address_row_2(address_row_2_TMVP), .address_col_1(address_col_1_TMVP), .address_col_2(address_col_2_TMVP), .address_valid(address_valid_TMVP),
            .data_row_1(data_row_1_TMVP), .data_row_2(data_row_2_TMVP), .data_col_1(data_col_1_TMVP), .data_col_2(data_col_2_TMVP), .data_valid(data_valid_TMVP),
            .address_vec_1(address_vec_1_TMVP), .address_vec_2(address_vec_2_TMVP), .address_vec_valid(address_vec_valid_TMVP),
            .data_vec_data_1(data_vec_data_1_TMVP), .data_vec_data_2(data_vec_data_2_TMVP), .data_vec_valid(data_vec_valid_TMVP),
            .m_axis_tdata(m_axis_tdata_TMVP), .m_axis_tvalid(m_axis_tvalid_TMVP)
            );            
        end
        else begin
            TMVP2_main  #(.N(N/2), .DATA_WIDTH(DATA_WIDTH), .TILE_SIZE(TILE_SIZE))  TMVP_inst
            (.clk(clk), .reset(reset), 
            .start(start_TMVP), 
            .address_row_1(address_row_1_TMVP), .address_row_2(address_row_2_TMVP), .address_col_1(address_col_1_TMVP), .address_col_2(address_col_2_TMVP), .address_valid(address_valid_TMVP),
            .data_row_1(data_row_1_TMVP), .data_row_2(data_row_2_TMVP), .data_col_1(data_col_1_TMVP), .data_col_2(data_col_2_TMVP), .data_valid(data_valid_TMVP),
            .address_vec_1(address_vec_1_TMVP), .address_vec_2(address_vec_2_TMVP), .address_vec_valid(address_vec_valid_TMVP),
            .data_vec_data_1(data_vec_data_1_TMVP), .data_vec_data_2(data_vec_data_2_TMVP), .data_vec_valid(data_vec_valid_TMVP),
            .m_axis_tdata(m_axis_tdata_TMVP), .m_axis_tvalid(m_axis_tvalid_TMVP)
            );            
        end
    endgenerate
    //____________________NEXT_TMVP____________________

    //____________________DATA_RAM____________________
    dual_port_ram_TMVP #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH($clog2(3*N))) ROW_RAM_1 (
        .clk(clk),
        .data_a(ROW_data_in_1[0]), .addr_a(ROW_address_1[0]), .we_a(ROW_wr_en_1[0]), .q_a(ROW_data_out_1[0]), 
        .data_b(ROW_data_in_1[1]), .addr_b(ROW_address_1[1]), .we_b(ROW_wr_en_1[1]), .q_b(ROW_data_out_1[1])
    );
    dual_port_ram_TMVP #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH($clog2(3*N))) COL_RAM_1 (
        .clk(clk),
        .data_a(COL_data_in_1[0]), .addr_a(COL_address_1[0]), .we_a(COL_wr_en_1[0]), .q_a(COL_data_out_1[0]), 
        .data_b(COL_data_in_1[1]), .addr_b(COL_address_1[1]), .we_b(COL_wr_en_1[1]), .q_b(COL_data_out_1[1])
    );
    dual_port_ram_TMVP #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH($clog2(3*N))) ROW_RAM_2 (
        .clk(clk),
        .data_a(ROW_data_in_2[0]), .addr_a(ROW_address_2[0]), .we_a(ROW_wr_en_2[0]), .q_a(ROW_data_out_2[0]), 
        .data_b(ROW_data_in_2[1]), .addr_b(ROW_address_2[1]), .we_b(ROW_wr_en_2[1]), .q_b(ROW_data_out_2[1])
    );
    dual_port_ram_TMVP #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH($clog2(3*N))) COL_RAM_2 (
        .clk(clk),
        .data_a(COL_data_in_2[0]), .addr_a(COL_address_2[0]), .we_a(COL_wr_en_2[0]), .q_a(COL_data_out_2[0]), 
        .data_b(COL_data_in_2[1]), .addr_b(COL_address_2[1]), .we_b(COL_wr_en_2[1]), .q_b(COL_data_out_2[1])
    );
    dual_port_ram_TMVP #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH($clog2(3*N))) VEC_RAM (
        .clk(clk),
        .data_a(VEC_data_in[0]), .addr_a(VEC_address[0]), .we_a(VEC_wr_en[0]), .q_a(VEC_data_out[0]), 
        .data_b(VEC_data_in[1]), .addr_b(VEC_address[1]), .we_b(VEC_wr_en[1]), .q_b(VEC_data_out[1])
    );
    //____________________DATA_RAM____________________

    //____________________RESULT____________________
    always @(posedge clk) begin
        m_axis_tdata_TMVP_D                     <=  m_axis_tdata_TMVP;
        m_axis_tdata_TMVP_D2                    <=  m_axis_tdata_TMVP_D;
        m_axis_tdata_TMVP_D3                    <=  m_axis_tdata_TMVP_D2;
        m_axis_tvalid_TMVP_D                    <=  m_axis_tvalid_TMVP;
        m_axis_tvalid_TMVP_D2                   <=  m_axis_tvalid_TMVP_D;
        m_axis_tvalid_TMVP_D3                   <=  m_axis_tvalid_TMVP_D2;
    end
    always @(posedge clk) begin 
        case (state)
            MULT_1: begin // A1*(B1 - B2)
                RES_address[0]                  <=  counter_in;
                RES_data_in[0]                  <=  m_axis_tdata_TMVP;
                RES_wr_en[0]                    <=  m_axis_tvalid_TMVP;
                RES_address[1]                  <=  counter_in + N;
                RES_data_in[1]                  <=  -m_axis_tdata_TMVP;
                RES_wr_en[1]                    <=  m_axis_tvalid_TMVP;
            end 
            MULT_2: begin // (A0 + A1 + A2)*B2
                RES_address[0]                  <=  counter_in;
                RES_wr_en[0]                    <=  1'b0;
                RES_address[1]                  <=  counter_in_D3;
                RES_data_in[1]                  <=  m_axis_tdata_TMVP_D3 + RES_data_out_D[0];
                RES_wr_en[1]                    <=  m_axis_tvalid_TMVP_D3;
            end 
            MULT_3: begin // (A1 + A2 + A3)*B1
                RES_address[0]                  <=  counter_in + N;
                RES_wr_en[0]                    <=  1'b0;
                RES_address[1]                  <=  counter_in_D3 + N;
                RES_data_in[1]                  <=  m_axis_tdata_TMVP_D3 + RES_data_out_D[0];
                RES_wr_en[1]                    <=  m_axis_tvalid_TMVP_D3;
            end 
            MULT_4: begin // A2*(B0 - B2)
                RES_address[0]                  <=  counter_in;
                RES_wr_en[0]                    <=  1'b0;
                RES_address[1]                  <=  counter_in + 2*N;
                RES_data_in[1]                  <=  -m_axis_tdata_TMVP;
                RES_wr_en[1]                    <=  m_axis_tvalid_TMVP;
            end 
            MULT_5: begin // A3*(B0 - B1)
                RES_address[0]                  <=  counter_in + N;
                RES_wr_en[0]                    <=  1'b0;
                RES_address[1]                  <=  counter_in;
                RES_data_in[1]                  <=  -m_axis_tdata_TMVP;
                RES_wr_en[1]                    <=  m_axis_tvalid_TMVP;
            end 
            MULT_6: begin // (A2 + A3 + A4)*B0
                RES_address[0]                  <=  counter_in;
                RES_wr_en[0]                    <=  1'b0;
                RES_address[1]                  <=  counter_in + 2*N;
                RES_wr_en[1]                    <=  1'b0;
            end 
            default: begin
                RES_wr_en[0]                    <=  1'b0;
                RES_wr_en[1]                    <=  1'b0;
            end
        endcase
    end
    always @(posedge clk) begin
        if (state   ==  MULT_4) begin
            m_axis_tdata                        <=  m_axis_tdata_TMVP_D3    +   RES_data_out_D[0];
            m_axis_tvalid                       <=  m_axis_tvalid_TMVP_D3;
        end
        else if(state   ==  MULT_5) begin
            m_axis_tdata                        <=  m_axis_tdata_TMVP_D3    +   RES_data_out_D[0];
            m_axis_tvalid                       <=  m_axis_tvalid_TMVP_D3;
        end
        else if (state  ==  MULT_6) begin
            m_axis_tdata_TEMP                   <=  RES_data_out[0]     +   RES_data_out[1];
            m_axis_tdata                        <=  m_axis_tdata_TEMP   +   m_axis_tdata_TMVP_D3;
            m_axis_tvalid                       <=  m_axis_tvalid_TMVP_D3;
        end
        else begin
            m_axis_tvalid                       <=  1'b0;
        end
    end
    dual_port_ram_TMVP #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH($clog2(3*N))) RESULT_RAM (
        .clk(clk),
        .data_a(RES_data_in[0]), .addr_a(RES_address[0]), .we_a(RES_wr_en[0]), .q_a(RES_data_out[0]), 
        .data_b(RES_data_in[1]), .addr_b(RES_address[1]), .we_b(RES_wr_en[1]), .q_b(RES_data_out[1])
    );
    always @(posedge clk) begin
        RES_data_out_D[0]                       <=  RES_data_out[0];  
        RES_data_out_D[1]                       <=  RES_data_out[1];  
    end
    //____________________RESULT____________________
endmodule
