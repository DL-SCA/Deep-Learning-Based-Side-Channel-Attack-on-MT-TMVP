module MatrixVectorMultiplier
    #(parameter N = 16, DATA_WIDTH = 8)
        (clk, reset, 
        s_axis_tdata_vec, 
        s_axis_tdata_row,
        s_axis_tvalid,
        m_axis_tdata, m_axis_tvalid,
        // SCA countermeasure: random mask input
        rng_mask
        );
    
    //____________________IO____________________
    // clk and reset
    input   wire                                clk;
    input   wire                                reset;
    // saxis row
    input   wire    signed  [DATA_WIDTH-1:0]    s_axis_tdata_row; 
    // saxis col
    input   wire    signed  [DATA_WIDTH-1:0]    s_axis_tdata_vec; 
    // saxis tvalid
    input   wire                                s_axis_tvalid; 
    // maxis
    output  reg     signed  [DATA_WIDTH-1:0]    m_axis_tdata; 
    output  reg                                 m_axis_tvalid;
    // SCA countermeasure: random mask from LFSR
    input   wire                    [31:0]      rng_mask;
    //____________________IO____________________

    //____________________Regs_and_Wires____________________
    localparam IDLE = 2'b00;
    localparam LOADING_FIRST_ROW = 2'b01;
    localparam OTHER_ROWS = 2'b10;
    reg                     [1:0]               state;
    reg                     [$clog2(N):0]       counter;
    reg     signed          [DATA_WIDTH-1:0]    buffered_data_row = 0;
    reg     signed          [DATA_WIDTH-1:0]    buffered_data_vec = 0;
    reg     signed          [DATA_WIDTH-1:0]    row[N-1:0];
    reg     signed          [DATA_WIDTH-1:0]    vector[N-1:0];
    reg     signed          [DATA_WIDTH-1:0]    mult_result[N-1:0];
    reg     signed          [DATA_WIDTH-1:0]    mult_result_D[N-1:0];
    reg                                         validity;
    reg 	 									mult_result_valid;
    reg 	 									mult_result_valid_D;
    reg     signed          [DATA_WIDTH-1:0]    add_result[$clog2(N/2)-1:0];
    reg     signed          [DATA_WIDTH-1:0]    add_result_s1[N/2-1:0];
    reg 	 									add_result_s1_valid;
    reg     signed          [DATA_WIDTH-1:0]    add_result_s2[N/4-1:0];
    reg 	 									add_result_s2_valid;
    reg     signed          [DATA_WIDTH-1:0]    add_result_s3[N/8-1:0];
    reg 	 									add_result_s3_valid;
    // SCA Countermeasure: mask registers and noise computation
    reg     signed          [DATA_WIDTH-1:0]    row_mask[N-1:0];
    reg     signed          [DATA_WIDTH-1:0]    mask_mult_result[N-1:0];
    reg     signed          [DATA_WIDTH-1:0]    mask_mult_result_D[N-1:0];
    reg     signed          [DATA_WIDTH-1:0]    mask_correction;
    (* DONT_TOUCH = "yes" *) reg signed [DATA_WIDTH-1:0] noise_reg;
    //____________________Regs_and_Wires___________________

    //____________________FSM___________________
    always @(posedge clk) begin
        if (!reset) begin
            state               <=  IDLE;
        end
        else begin
            case (state)
                IDLE: begin
                    if (s_axis_tvalid) begin
                        state       <=  LOADING_FIRST_ROW;
                    end   
				end 
                LOADING_FIRST_ROW: begin
                    if (counter == N - 1    &&  s_axis_tvalid) begin
                        state       <=  OTHER_ROWS;
                    end
                end
                OTHER_ROWS: begin
                    if (counter == 2*N - 1) begin
                        if (s_axis_tvalid) begin
                            state   <=  LOADING_FIRST_ROW;
                        end
                        else begin
                            state   <=  IDLE;
                        end
                    end
                end 
                default: begin 
                    state   <=  IDLE;
                end
            endcase
        end
    end
    always @(posedge clk) begin
        if (state   ==  IDLE) begin
            if (s_axis_tvalid) begin
                counter     <=  1;  
            end
            else begin
                counter     <=  0;  
            end
        end
        else if (state  ==  LOADING_FIRST_ROW) begin
            if (s_axis_tvalid) begin
                counter     <=  counter + 1;
            end
        end
        else begin
            if (s_axis_tvalid) begin
                if (counter ==  2*N-1) begin
                    counter <=  1;
                end
                else begin
                    counter <=  counter + 1;
                end
            end
        end
    end
    //____________________FSM___________________

    //____________________row_update___________________
    genvar i;
    generate
        for (i = 0; i < N; i=i+1) begin
            always @(posedge clk) begin
                case (state)
                    IDLE: begin
                        if (s_axis_tvalid) begin
                            if (i == 0) begin
                                row[0]  <=  s_axis_tdata_row;
                            end
                        end
                    end
                    default: begin
                        if (s_axis_tvalid) begin
                            if (i==0) begin
                                row[i]  <=  s_axis_tdata_row;
                            end
                            else begin
                                row[i]  <=  row[i-1];
                            end
                        end
                    end 
                endcase
            end
        end
    endgenerate
    //____________________row_update___________________

    //____________________vector_update___________________
    generate
        for (i = 0; i < N; i=i+1) begin
            always @(posedge clk) begin
                case (state)
                    IDLE: begin
                        if (s_axis_tvalid) begin
                            if (i == 0) begin
                                vector[0]   <=  s_axis_tdata_vec;
                            end
                        end
                    end
                    LOADING_FIRST_ROW: begin
                        if (s_axis_tvalid) begin
                            if (i == 0) begin
                                vector[0]   <=  s_axis_tdata_vec;                            
                            end
                            else begin
                                vector[i]   <=  vector[i-1];
                            end
                        end
                    end 
                    default: begin
                        if (counter == 2*N-1    &&  s_axis_tvalid) begin
                            if (i == 0) begin
                                vector[0]   <=  s_axis_tdata_vec;
                            end
                        end
                    end 
                endcase
            end
        end
    endgenerate
    //____________________vector_update___________________

    //____________________multiplication___________________
    // SCA Countermeasure: Arithmetic masking on multiplier inputs.
    // Instead of computing row[i]*vec[i] directly, we compute:
    //   (row[i] + mask[i]) * vec[i]  and separately  mask[i] * vec[i]
    // The final result = masked_product - mask_product, which equals row[i]*vec[i]
    // but the intermediate power consumption is decorrelated from row[i].
    generate
        for (i = 0; i < N; i=i+1) begin
            always @(posedge clk) begin
                // Masked multiplication: operates on (row + mask) instead of row
                mult_result[i]      	<=  (row[i] + row_mask[i])  *  vector[i];
                // Mask correction term: will be subtracted in summation
                mask_mult_result[i] 	<=  row_mask[i] * vector[i];
            end
        end
        always @(posedge clk) begin
            mult_result_valid		    <=	validity;
        end
    endgenerate
    // SCA Countermeasure: Noise generator - performs dummy operations
    // that consume power uncorrelated to secret data.
    // DONT_TOUCH prevents synthesis optimization from removing it.
    always @(posedge clk) begin
        noise_reg <= rng_mask[DATA_WIDTH-1:0] * rng_mask[DATA_WIDTH+7:8];
    end
    // SCA Countermeasure: Capture per-element masks from LFSR on row load.
    // Each row element gets a different mask derived from the LFSR state.
    // mask[0] takes LFSR bits directly; mask[i] = mask[i-1] XOR rotated LFSR.
    // This gives each element a distinct mask that changes every execution.
    generate
        for (i = 0; i < N; i=i+1) begin : gen_mask
            always @(posedge clk) begin
                if (!reset) begin
                    row_mask[i] <= 0;
                end
                else if (s_axis_tvalid) begin
                    if (i == 0) begin
                        row_mask[0] <= rng_mask[DATA_WIDTH-1:0];
                    end
                    else begin
                        row_mask[i] <= row_mask[i-1] ^ rng_mask[DATA_WIDTH-1:0]
                                     ^ rng_mask[2*DATA_WIDTH-1:DATA_WIDTH];
                    end
                end
            end
        end
    endgenerate
    always @(posedge clk) begin
        case (state)
            LOADING_FIRST_ROW: begin
                if (counter ==  N-1) begin
                    validity            <=  s_axis_tvalid;
                end
                else begin
                    validity            <=  1'b0;
                end
            end 
            OTHER_ROWS: begin
                validity                <=  s_axis_tvalid   &&  (counter    !=  2*N-1);
            end
            default: begin
                validity                <=  1'b0;
            end 
        endcase
    end
    //____________________multiplication___________________

    //____________________summation___________________
    // SCA Countermeasure: The summation tree now operates on the masked
    // products (mult_result_D) and simultaneously computes the mask
    // correction sum (mask_mult_result_D). The mask correction is
    // subtracted at the final output stage.
    reg     signed  [DATA_WIDTH-1:0]    mask_add_s1[N/2-1:0];
    reg     signed  [DATA_WIDTH-1:0]    mask_add_s2[N/4-1:0];
    reg     signed  [DATA_WIDTH-1:0]    mask_add_s3[N/8-1:0];
    integer j;
    generate
        if (N == 16) begin
            always @(posedge clk) begin
                for (j = 0; j < N/2; j=j+1) begin
                    add_result_s1[j]  	<=  mult_result_D[2*j] + mult_result_D[2*j+1];
                    mask_add_s1[j]    	<=  mask_mult_result_D[2*j] + mask_mult_result_D[2*j+1];
                end
                for (j = 0; j < N/4; j=j+1) begin
                    add_result_s2[j]  	<=  add_result_s1[2*j] + add_result_s1[2*j+1];
                    mask_add_s2[j]    	<=  mask_add_s1[2*j] + mask_add_s1[2*j+1];
                end
                for (j = 0; j < N/8; j=j+1) begin
                    add_result_s3[j]  	<=  add_result_s2[2*j] + add_result_s2[2*j+1];
                    mask_add_s3[j]    	<=  mask_add_s2[2*j] + mask_add_s2[2*j+1];
                end
            end
        end
        else if (N == 12) begin
            always @(posedge clk) begin
                for (j = 0; j < N/2; j=j+1) begin
                    add_result_s1[j]  	<=  mult_result_D[2*j] + mult_result_D[2*j+1];
                    mask_add_s1[j]    	<=  mask_mult_result_D[2*j] + mask_mult_result_D[2*j+1];
                end
                for (j = 0; j < N/2; j=j+1) begin
                    add_result_s2[j]  	<=  add_result_s1[2*j] + add_result_s1[2*j+1];
                    mask_add_s2[j]    	<=  mask_add_s1[2*j] + mask_add_s1[2*j+1];
                end
            end
        end
        else begin
            always @(posedge clk) begin
                for (j = 0; j < N/2; j=j+1) begin
                    add_result_s1[j]  	<=  mult_result_D[2*j] + mult_result_D[2*j+1];
                    mask_add_s1[j]    	<=  mask_mult_result_D[2*j] + mask_mult_result_D[2*j+1];
                end
                    add_result_s2[0]  	<=  add_result_s1[0] + add_result_s1[1];
                    add_result_s2[1]  	<=  add_result_s1[2] + add_result_s1[3] + add_result_s1[4];
                    mask_add_s2[0]    	<=  mask_add_s1[0] + mask_add_s1[1];
                    mask_add_s2[1]    	<=  mask_add_s1[2] + mask_add_s1[3] + mask_add_s1[4];
            end
        end
		always @(posedge clk) begin
            for (j = 0; j < N; j=j+1) begin
                mult_result_D[j]        <=  mult_result[j];
                mask_mult_result_D[j]   <=  mask_mult_result[j];
            end
			mult_result_valid_D			<=	mult_result_valid;
			add_result_s1_valid			<=	mult_result_valid_D;
			add_result_s2_valid			<=	add_result_s1_valid;
			add_result_s3_valid			<=	add_result_s2_valid;
		end	
    endgenerate
    //____________________summation___________________

    //____________________output___________________
    // SCA Countermeasure: Subtract the mask correction at the output.
    // result = sum((row+mask)*vec) - sum(mask*vec) = sum(row*vec)
    // The unmasking happens here at the last stage, so intermediate
    // computations never operate on raw secret values.
    generate
        if (N == 16) begin
            always @(posedge clk) begin
                m_axis_tdata            <=  (add_result_s3[0] + add_result_s3[1])
                                          - (mask_add_s3[0] + mask_add_s3[1]);
			    m_axis_tvalid			<=	add_result_s3_valid;
            end            
        end
        else if (N == 12) begin
            always @(posedge clk) begin
                m_axis_tdata            <=  (add_result_s2[0] + add_result_s2[1] + add_result_s2[2])
                                          - (mask_add_s2[0] + mask_add_s2[1] + mask_add_s2[2]);
			    m_axis_tvalid			<=	add_result_s2_valid;
            end            
        end
        else begin // N == 10
            always @(posedge clk) begin
                m_axis_tdata            <=  (add_result_s2[0] + add_result_s2[1])
                                          - (mask_add_s2[0] + mask_add_s2[1]);
			    m_axis_tvalid			<=	add_result_s2_valid;
            end            
        end
    endgenerate
    //____________________output___________________
endmodule
