// softmax_engine_top.sv
// Online Softmax Engine - top-level integration


module softmax_engine_top (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        valid_in,
    input  logic [15:0] score_in,   // BF16
    input  logic [15:0] value_in,   // BF16 (scalar, d=1)

    input  logic        row_start,  // pulse: reset {m,l,O} state
    input  logic        row_end,    // pulse: trigger divide and emit output

    output logic        valid_out,
    output logic [15:0] denom_out,  // l truncated to BF16 (debug)
    output logic [15:0] output_out  // O/l in BF16
);

    
// Constants

    localparam logic [15:0] BF16_NEG_INF = 16'hFF80;
    localparam logic [31:0] FP32_ZERO    = 32'h00000000;
    localparam logic [31:0] FP32_ONE     = 32'h3F800000;

// Latency parameters (measured from real Vivado synthesis)

    localparam int S0_LAT     = 1;
    localparam int S1_LAT     = 3;   
    localparam int S2_LAT     = 2;  
                                     
    localparam int S3_LAT     = 8;   
                                      
                                  
    localparam int DIV_LAT    = 29;  
    localparam int TOTAL_LAT  = S0_LAT + S1_LAT + S2_LAT + S3_LAT; 
    .
    localparam int STALL_VAL  = S3_LAT - 1;                     

 
    // State registers: m (BF16), l (FP32), O (FP32)
    // Updated only on accumulator out_valid (every 25 cycles in v1)
 
    logic [15:0] m_reg;
    logic [31:0] l_reg;
    logic [31:0] O_reg;

   typedef enum logic [2:0] {
        S_RESET_FLUSH,
        S_IDLE,
        S_ROW_ACTIVE,
        S_ROW_DRAIN,
        S_DIV_WAIT,
        S_EMIT
    } fsm_state_t;

    fsm_state_t state;
    logic [$clog2(TOTAL_LAT+1)-1:0] stall_cnt;
    logic [$clog2(DIV_LAT+1)-1:0]   div_cnt;

 
    // S0: Input register
 
    logic        s0_valid;
    logic [15:0] s0_score;
    logic [15:0] s0_value;

    always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        s0_valid <= 1'b0;
        s0_score <= 16'h0000;
        s0_value <= 16'h0000;
    end else begin
        s0_valid <= valid_in && (state == S_ROW_ACTIVE) && (stall_cnt == '0);
        
        if (valid_in && (state == S_ROW_ACTIVE) && (stall_cnt == '0)) begin
            s0_score <= score_in;
            s0_value <= value_in;
        end
        // else: s0_score / s0_value hold their previous value
    end
end

 
    // S1: Compare & Delta  (3 cycles total)
    

    // --- Combinational compare ---
    logic [15:0] s1_m_new_comb;
    logic        s1_flag_comb;

    bf16_compare u_compare (
        .a            (m_reg),
        .b            (s0_score),
        .max_out      (s1_m_new_comb),
        .new_max_flag (s1_flag_comb)
    );

       logic [15:0] s1_delta;

   
bf16_delta u_delta (
    .clk    (clk),
    .rst_n  (rst_n),
    .a      (m_reg),
    .b      (s0_score),
    .delta  (s1_delta)
);
    //S1a: companion register, cycle 1 
    logic        s1_valid;
    logic [15:0] s1_m_new;
    logic        s1_flag;
    logic [15:0] s1_value;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
            s1_m_new <= BF16_NEG_INF;
            s1_flag  <= 1'b0;
            s1_value <= 16'h0000;
        end else begin
            s1_valid <= s0_valid;
            s1_m_new <= s1_m_new_comb;
            s1_flag  <= s1_flag_comb;
            s1_value <= s0_value;
        end
    end

    // --- S1b: companion register, cycle 2 ---
    logic        s1_valid_b;
    logic [15:0] s1_m_new_b;
    logic        s1_flag_b;
    logic [15:0] s1_value_b;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid_b <= 1'b0;
            s1_m_new_b <= BF16_NEG_INF;
            s1_flag_b  <= 1'b0;
            s1_value_b <= 16'h0000;
        end else begin
            s1_valid_b <= s1_valid;
            s1_m_new_b <= s1_m_new;
            s1_flag_b  <= s1_flag;
            s1_value_b <= s1_value;
        end
    end

    // S1c: companion register, cycle 3 (aligned with bf16_delta output) 
    logic        s1_valid_d;
    logic [15:0] s1_m_new_d;
    logic        s1_flag_d;
    logic [15:0] s1_value_d;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid_d <= 1'b0;
            s1_m_new_d <= BF16_NEG_INF;
            s1_flag_d  <= 1'b0;
            s1_value_d <= 16'h0000;
        end else begin
            s1_valid_d <= s1_valid_b;
            s1_m_new_d <= s1_m_new_b;
            s1_flag_d  <= s1_flag_b;
            s1_value_d <= s1_value_b;
        end
    end

 
    // S2: Shift-LUT Exponential (1-cycle registered output, frozen module)
 
    logic        s2_valid;
    logic [11:0] s2_exp_delta;  // Q1.11

    shift_lut_exp u_exp (
        .clk        (clk),
        .rst_n      (rst_n),
        .valid_in   (s1_valid_d),   // aligned with 3-cycle bf16_delta output
        .delta      (s1_delta),     // bf16_delta registered output
        .valid_out  (s2_valid),
        .exp_result (s2_exp_delta)
    );

    // Companion registers: pipeline s1c signals to align with s2_valid
    logic        s2_flag;
    logic [15:0] s2_m_new;
    logic [15:0] s2_value;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_flag  <= 1'b0;
            s2_m_new <= BF16_NEG_INF;
            s2_value <= 16'h0000;
        end else begin
            s2_flag  <= s1_flag_d;
            s2_m_new <= s1_m_new_d;
            s2_value <= s1_value_d;
        end
    end

 
    // S2 -> S3 bridge: Q1.11->FP32 + r/n mux (combinational, same cycle as s2_valid)
    
 
    logic [3:0]  ed_lzc;
    logic [31:0] exp_fp32;

    always_comb begin
        casez (s2_exp_delta)
            12'b1???????????:  ed_lzc = 4'd0;
            12'b01??????????:  ed_lzc = 4'd1;
            12'b001?????????:  ed_lzc = 4'd2;
            12'b0001????????:  ed_lzc = 4'd3;
            12'b00001???????:  ed_lzc = 4'd4;
            12'b000001??????:  ed_lzc = 4'd5;
            12'b0000001?????:  ed_lzc = 4'd6;
            12'b00000001????:  ed_lzc = 4'd7;
            12'b000000001???:  ed_lzc = 4'd8;
            12'b0000000001??:  ed_lzc = 4'd9;
            12'b00000000001?:  ed_lzc = 4'd10;
            12'b000000000001:  ed_lzc = 4'd11;
            default:           ed_lzc = 4'd15;  // zero / underflow
        endcase
    end

    always_comb begin
        if (s2_exp_delta == 12'h000 || ed_lzc == 4'd15) begin
            exp_fp32 = FP32_ZERO;
        end else begin
          
            exp_fp32 = {1'b0,
                        8'(127 - int'(ed_lzc)),
                        {s2_exp_delta << (ed_lzc + 1), 11'h0}};
        end
    end

    // BF16 value_in -> FP32: zero-extend mantissa (sign/exp layout identical)
    logic [31:0] value_fp32;
    assign value_fp32 = {s2_value, 16'h0000};

    // r/n routing: new_max_flag=1 means s_i > m_old, new maximum
    //   r = e^(-delta) = exp_fp32,  n = 1.0
    //   otherwise: r = 1.0, n = exp_fp32
    logic [31:0] r_fp32, n_fp32;
    assign r_fp32 = s2_flag ? exp_fp32 : FP32_ONE;
    assign n_fp32 = s2_flag ? FP32_ONE : exp_fp32;




    // PIPELINE REGISTER: S2->S3 bridge -> accumulator boundary
   
    logic        s3_valid;
    logic [31:0] s3_r_fp32, s3_n_fp32, s3_value_fp32;
    logic [15:0] s3_m_new;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s3_valid      <= 1'b0;
            s3_r_fp32     <= 32'h0;
            s3_n_fp32     <= 32'h0;
            s3_value_fp32 <= 32'h0;
            s3_m_new      <= BF16_NEG_INF;
        end else begin
            s3_valid      <= s2_valid;
            s3_r_fp32     <= r_fp32;
            s3_n_fp32     <= n_fp32;
            s3_value_fp32 <= value_fp32;
            s3_m_new      <= s2_m_new;
        end
    end

 
    // S3: Accumulator update
   
    logic        accum_out_valid;
    logic [31:0] l_new;
    logic [31:0] O_new;

    accumulator_update u_accum (
        .clk           (clk),
        .rst_n         (rst_n),
        .in_valid      (s3_valid),
        .l_old         (l_reg),
        .O_old         (O_reg),
        .r_fp32        (s3_r_fp32),
        .n_fp32        (s3_n_fp32),
        .value_in_fp32 (s3_value_fp32),
        .out_valid     (accum_out_valid),
        .l_new         (l_new),
        .O_new         (O_new)
    );

    
 
 
    logic [15:0] m_new_sr       [0:S3_LAT-1];
    logic        m_new_valid_sr [0:S3_LAT-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < S3_LAT; i++) begin
                m_new_sr[i]       <= BF16_NEG_INF;
                m_new_valid_sr[i] <= 1'b0;
            end
        end else begin
            m_new_sr[0]       <= s3_m_new;
            m_new_valid_sr[0] <= s3_valid;
            for (int i = 1; i < S3_LAT; i++) begin
                m_new_sr[i]       <= m_new_sr[i-1];
                m_new_valid_sr[i] <= m_new_valid_sr[i-1];
            end
        end
    end

    logic [15:0] m_new_aligned;
    logic        m_new_aligned_valid;
    assign m_new_aligned       = m_new_sr[S3_LAT-1];
    assign m_new_aligned_valid = m_new_valid_sr[S3_LAT-1];

   
    always_ff @(posedge clk) begin
        if (rst_n && (m_new_aligned_valid !== accum_out_valid)) begin
            $error("softmax_engine_top: m_new_aligned_valid (%b) != accum_out_valid (%b) at %0t. \
m_reg and {l_reg,O_reg} update timing is misaligned.",
                m_new_aligned_valid, accum_out_valid, $time);
        end
    end
   

   
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_reg <= BF16_NEG_INF;
        end else if (row_start) begin
            m_reg <= BF16_NEG_INF;
        end else if (m_new_aligned_valid) begin
            m_reg <= m_new_aligned;
        end
    end

 

 
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            l_reg <= FP32_ZERO;
            O_reg <= FP32_ZERO;
        end else if (row_start) begin
            l_reg <= FP32_ZERO;
            O_reg <= FP32_ZERO;
        end else if (accum_out_valid) begin
            l_reg <= l_new;
            O_reg <= O_new;
        end
    end

 

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_RESET_FLUSH;
            stall_cnt <= TOTAL_LAT[($clog2(TOTAL_LAT+1)-1):0];
            div_cnt   <= '0;
        end else begin
            case (state)

                S_RESET_FLUSH: begin
                    
                    if (stall_cnt == '0)
                        state <= S_IDLE;
                    else
                        stall_cnt <= stall_cnt - 1'b1;
                end

                S_IDLE: begin
                    if (row_start) begin
                        state     <= S_ROW_ACTIVE;
                        stall_cnt <= '0;
                    end
                end

                S_ROW_ACTIVE: begin
                    if (row_end) begin
                        
                        state     <= S_ROW_DRAIN;
                        stall_cnt <= STALL_VAL[$clog2(TOTAL_LAT+1)-1:0];
                    end else if (s3_valid) begin
                       
                        stall_cnt <= STALL_VAL[$clog2(TOTAL_LAT+1)-1:0];
                    end else if (stall_cnt != '0) begin
                        stall_cnt <= stall_cnt - 1'b1;
                    end
                end

                S_ROW_DRAIN: begin
                    // Wait for any in-flight accumulator result to land
                    // before launching the divide on the final {l,O} state.
                    if (stall_cnt == '0) begin
                        state   <= S_DIV_WAIT;
                        div_cnt <= DIV_LAT[$clog2(DIV_LAT+1)-1:0];
                    end else begin
                        stall_cnt <= stall_cnt - 1'b1;
                    end
                end

                S_DIV_WAIT: begin
                    if (div_cnt == '0)
                        state <= S_EMIT;
                    else
                        div_cnt <= div_cnt - 1'b1;
                end

                S_EMIT: begin
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

 
 
    logic div_launch;
    assign div_launch = (state == S_ROW_DRAIN) && (stall_cnt == '0);

    logic [31:0] div_result;
    logic        div_result_valid;

    fp_div_synth u_div (
        .aclk                   (clk),
        .s_axis_a_tvalid        (div_launch),
        .s_axis_a_tready        (),
        .s_axis_a_tdata         (O_reg),      // numerator
        .s_axis_b_tvalid        (div_launch),
        .s_axis_b_tready        (),
        .s_axis_b_tdata         (l_reg),      // denominator
        .m_axis_result_tvalid   (div_result_valid),
        .m_axis_result_tready   (1'b1),
        .m_axis_result_tdata    (div_result)
    );

 
    // FP32 -> BF16 truncation (combinational): take upper 16 bits.
    // BF16 and FP32 share the same sign + exponent format; truncating
    // the lower 16 mantissa bits gives valid BF16.
    // Round-to-nearest: if the dropped bit 15 is 1, round up by 1 ULP.
 
    logic [15:0] result_bf16;
    always_comb begin
        if (div_result[15] && (div_result[31:16] != 16'h7FFF))
            result_bf16 = div_result[31:16] + 16'h0001;
        else
            result_bf16 = div_result[31:16];
    end

    // l_reg truncated to BF16 for denom_out (debug signal)
    logic [15:0] denom_bf16;
    assign denom_bf16 = l_reg[31:16];

 
    // Output register: latch on S_EMIT
 
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out  <= 1'b0;
            output_out <= 16'h0000;
            denom_out  <= 16'h0000;
        end else if (state == S_EMIT) begin
            valid_out  <= 1'b1;
            output_out <= result_bf16;
            denom_out  <= denom_bf16;
        end else begin
            valid_out  <= 1'b0;
        end
    end

    
   
    always_ff @(posedge clk) begin : score_launch_monitor
        if (s3_valid) begin
            $display("=== STAGE3 LAUNCH === r=%h(%e) n=%h(%e) val=%h(%e) flag=%b m_new=%h",
                s3_r_fp32, 32'h0,
                s3_n_fp32, 32'h0,
                s3_value_fp32, 32'h0,
                s2_flag, s3_m_new);
        end
    end

    always_ff @(posedge clk) begin : accum_done_monitor
        if (accum_out_valid) begin
            $display("=== ACCUM DONE === l_new=%h(%e) O_new=%h(%e)",
                l_new, 32'h0,
                O_new, 32'h0);
            $display("                  l_reg=%h(%e) O_reg=%h(%e) m_reg=%h",
                l_reg, 32'h0,
                O_reg, 32'h0,
                m_reg);
        end
    end

    always_ff @(posedge clk) begin : div_monitor
        if (div_result_valid) begin
            $display("=== DIV RESULT === div=%h(%e) l_reg=%h(%e)",
                div_result, 32'h0,
                l_reg, 32'h0);
        end
    end
   

endmodule
