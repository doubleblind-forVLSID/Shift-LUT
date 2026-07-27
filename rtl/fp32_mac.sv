// fp32_mac.sv
// FP32 multiply-accumulate: result = a * b + c
// Used in Stage 3 for:
//   l_new = l_old * r + n
//   O_new = O_old * r + n * value_in


module fp32_mac (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        valid_in,
    input  logic [31:0] a,      // multiplicand
    input  logic [31:0] b,      // multiplier
    input  logic [31:0] c,      // addend
    output logic        valid_out,
    output logic [31:0] result  // round(a*b) + c, rounded to nearest-even
);

    
    // Part 1: FP32 Multiply a * b, with round-to-nearest-even
    
    logic        ma_sign, mb_sign, mp_sign;
    logic [7:0]  ma_exp,  mb_exp;
    logic [9:0]  mp_exp_raw;
    logic [23:0] ma_mant, mb_mant;
    logic [47:0] mp_mant_full;
    logic [31:0] mul_result;

    logic a_zero, b_zero;
    assign a_zero = (a[30:0] == 31'h0);
    assign b_zero = (b[30:0] == 31'h0);

    assign ma_sign = a[31];
    assign mb_sign = b[31];
    assign ma_exp  = a[30:23];
    assign mb_exp  = b[30:23];
    assign ma_mant = {1'b1, a[22:0]};
    assign mb_mant = {1'b1, b[22:0]};

    assign mp_sign      = ma_sign ^ mb_sign;
    assign mp_mant_full = ma_mant * mb_mant;                       // exact 48-bit product
    assign mp_exp_raw   = {2'b00, ma_exp} + {2'b00, mb_exp} - 10'd127;

    logic [23:0] mp_mant24;       // 24-bit kept field pre-round
    logic        mp_guard, mp_sticky;
    logic [7:0]  mp_exp_adj;      // 0 or 1, normalization shift
    logic [7:0]  mp_exp_final;
    logic [22:0] mp_mant_rounded;

    always_comb begin
        mp_mant24  = 24'h0;
        mp_guard   = 1'b0;
        mp_sticky  = 1'b0;
        mp_exp_adj = 8'd0;

        if (mp_mant_full[47]) begin
            // Product >= 2.0: leading 1 at bit 47
            mp_mant24  = mp_mant_full[47:24];
            mp_guard   = mp_mant_full[23];
            mp_sticky  = |mp_mant_full[22:0];
            mp_exp_adj = 8'd1;
        end else begin
            // Product in [1.0, 2.0): leading 1 at bit 46
            mp_mant24  = mp_mant_full[46:23];
            mp_guard   = mp_mant_full[22];
            mp_sticky  = |mp_mant_full[21:0];
            mp_exp_adj = 8'd0;
        end
    end

    
    logic [24:0] mp_mant25_rounded;
    logic        mp_round_carry;
    always_comb begin
        mp_mant25_rounded = {1'b0, mp_mant24};
        if (mp_guard && (mp_sticky || mp_mant24[0]))
            mp_mant25_rounded = {1'b0, mp_mant24} + 25'd1;
        mp_round_carry = mp_mant25_rounded[24];
    end

    always_comb begin
        if (a_zero || b_zero) begin
            mul_result = 32'h00000000;
        end else if (mp_exp_raw[9] || mp_exp_raw == 10'h0) begin
            mul_result = {mp_sign, 31'h0};
        end else if (mp_exp_raw >= 10'd255) begin
            mul_result = {mp_sign, 8'hFF, 23'h0};
        end else begin
            mp_exp_final = mp_exp_raw[7:0] + mp_exp_adj + {7'h0, mp_round_carry};
            if (mp_round_carry)
                mp_mant_rounded = mp_mant25_rounded[23:1];
            else
                mp_mant_rounded = mp_mant25_rounded[22:0];
            mul_result = {mp_sign, mp_exp_final, mp_mant_rounded};
        end
    end
    
    logic        valid_s1;
    logic [31:0] mul_result_s1;
    logic [31:0] c_s1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s1      <= 1'b0;
            mul_result_s1 <= 32'h00000000;
            c_s1          <= 32'h00000000;
        end else begin
            valid_s1      <= valid_in;
            mul_result_s1 <= mul_result;
            c_s1          <= c;
        end
    end

    
    // Part 2: FP32 Add mul_result_s1 + c_s1, with guard/round/sticky
    
    localparam int W = 3;  // guard/round/sticky extension width

    logic        as_sign, bs_sign, rs_sign;
    logic [7:0]  as_exp,  bs_exp;
    logic [23:0] as_mant, bs_mant;
    logic [7:0]  exp_diff_add;

    logic [31:0] add_a, add_b;
    assign add_a = mul_result_s1;
    assign add_b = c_s1;

    logic a_add_zero, b_add_zero;
    assign a_add_zero = (add_a[30:0] == 31'h0);
    assign b_add_zero = (add_b[30:0] == 31'h0);

    assign as_sign = add_a[31];
    assign bs_sign = add_b[31];
    assign as_exp  = add_a[30:23];
    assign bs_exp  = add_b[30:23];
    assign as_mant = {1'b1, add_a[22:0]};
    assign bs_mant = {1'b1, add_b[22:0]};

    logic [23:0] larger_mant, smaller_mant;
    logic        larger_sign;
    logic [7:0]  result_exp_add;

    always_comb begin
        if (as_exp >= bs_exp) begin
            exp_diff_add   = as_exp - bs_exp;
            larger_mant    = as_mant;
            larger_sign    = as_sign;
            smaller_mant   = bs_mant;
            result_exp_add = as_exp;
            rs_sign        = as_sign;
        end else begin
            exp_diff_add   = bs_exp - as_exp;
            larger_mant    = bs_mant;
            larger_sign    = bs_sign;
            smaller_mant   = as_mant;
            result_exp_add = bs_exp;
            rs_sign        = bs_sign;
        end
    end

    // Extend both operands by W guard/round/sticky bits before alignment.
    logic [23+W:0] larger_ext;
    logic [23+W:0] smaller_ext_full;
    logic [23+W:0] shifted_ext;
    logic          align_sticky;

    assign larger_ext       = {larger_mant, {W{1'b0}}};
    assign smaller_ext_full = {smaller_mant, {W{1'b0}}};

    always_comb begin
        if (exp_diff_add >= (24 + W)) begin
            // entire smaller operand shifts out; folds to a single sticky bit
            shifted_ext = (smaller_mant != 24'h0) ? {{(23+W){1'b0}}, 1'b1} : '0;
        end else begin
            // sticky = OR of every bit shifted off the bottom
            align_sticky = |(smaller_ext_full & ((1 << exp_diff_add) - 1));
            shifted_ext  = (smaller_ext_full >> exp_diff_add) | {{(23+W){1'b0}}, align_sticky};
        end
    end

    logic same_sign_add;
    assign same_sign_add = (as_sign == bs_sign);

    
    // PIPELINE REGISTER: align -> add/normalize/round boundary
    
    logic        valid_s2;
    logic [23+W:0] larger_ext_s2, shifted_ext_s2;
    logic          same_sign_add_s2;
    logic          rs_sign_s2;
    logic [7:0]    result_exp_add_s2;
    logic          a_add_zero_s2, b_add_zero_s2;
    logic [31:0]   add_a_s2, add_b_s2;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s2          <= 1'b0;
            larger_ext_s2     <= '0;
            shifted_ext_s2    <= '0;
            same_sign_add_s2  <= 1'b0;
            rs_sign_s2        <= 1'b0;
            result_exp_add_s2 <= 8'h0;
            a_add_zero_s2     <= 1'b0;
            b_add_zero_s2     <= 1'b0;
            add_a_s2          <= 32'h0;
            add_b_s2          <= 32'h0;
        end else begin
            valid_s2          <= valid_s1;
            larger_ext_s2     <= larger_ext;
            shifted_ext_s2    <= shifted_ext;
            same_sign_add_s2  <= same_sign_add;
            rs_sign_s2        <= rs_sign;
            result_exp_add_s2 <= result_exp_add;
            a_add_zero_s2     <= a_add_zero;
            b_add_zero_s2     <= b_add_zero;
            add_a_s2          <= add_a;
            add_b_s2          <= add_b;
        end
    end

    
    // Stage 2b: add/subtract, normalize, guard/round/sticky rounding
    
    logic [24+W:0] add_sum_ext;   // one extra bit to catch carry-out
    always_comb begin
        if (same_sign_add_s2) begin
            add_sum_ext = {1'b0, larger_ext_s2} + {1'b0, shifted_ext_s2};
        end else begin
            if (larger_ext_s2 >= shifted_ext_s2)
                add_sum_ext = {1'b0, larger_ext_s2} - {1'b0, shifted_ext_s2};
            else
                add_sum_ext = {1'b0, shifted_ext_s2} - {1'b0, larger_ext_s2};
        end
    end

    logic add_result_sign;
    always_comb begin
        if (same_sign_add_s2)
            add_result_sign = add_a_s2[31];  // as_sign, preserved via add_a_s2
        else if (larger_ext_s2 >= shifted_ext_s2)
            add_result_sign = rs_sign_s2;
        else
            add_result_sign = ~rs_sign_s2;
    end

    // Normalize: handle carry-out (bit 24+W) or find leading one otherwise.
    logic [24+W:0] norm_ext;
    logic [7:0]    norm_exp;
    logic [4:0]    lzc_add;       // 0-23 possible, 5 bits
    logic          add_is_zero;

    assign add_is_zero = (add_sum_ext == '0);

    always_comb begin
        norm_ext = add_sum_ext;
        norm_exp = result_exp_add_s2;
        lzc_add  = 5'd0;

        if (!add_is_zero) begin
            if (add_sum_ext[24+W]) begin
                // carry out: shift right 1, exponent += 1, fold LSB into sticky
                norm_ext = {1'b0, add_sum_ext[24+W:1]};
                norm_ext[0] = norm_ext[0] | add_sum_ext[0];
                norm_exp = result_exp_add_s2 + 8'd1;
            end else begin
                // find leading one within bits [23+W:0]
                casez (add_sum_ext[23+W:0])
                    27'b1??????????????????????????: lzc_add = 5'd0;
                    27'b01?????????????????????????: lzc_add = 5'd1;
                    27'b001????????????????????????: lzc_add = 5'd2;
                    27'b0001???????????????????????: lzc_add = 5'd3;
                    27'b00001??????????????????????: lzc_add = 5'd4;
                    27'b000001?????????????????????: lzc_add = 5'd5;
                    27'b0000001????????????????????: lzc_add = 5'd6;
                    27'b00000001???????????????????: lzc_add = 5'd7;
                    27'b000000001??????????????????: lzc_add = 5'd8;
                    27'b0000000001?????????????????: lzc_add = 5'd9;
                    27'b00000000001????????????????: lzc_add = 5'd10;
                    27'b000000000001???????????????: lzc_add = 5'd11;
                    27'b0000000000001??????????????: lzc_add = 5'd12;
                    27'b00000000000001?????????????: lzc_add = 5'd13;
                    27'b000000000000001????????????: lzc_add = 5'd14;
                    27'b0000000000000001???????????: lzc_add = 5'd15;
                    27'b00000000000000001??????????: lzc_add = 5'd16;
                    27'b000000000000000001?????????: lzc_add = 5'd17;
                    27'b0000000000000000001????????: lzc_add = 5'd18;
                    27'b00000000000000000001???????: lzc_add = 5'd19;
                    27'b000000000000000000001??????: lzc_add = 5'd20;
                    27'b0000000000000000000001?????: lzc_add = 5'd21;
                    27'b00000000000000000000001????: lzc_add = 5'd22;
                    27'b000000000000000000000001???: lzc_add = 5'd23;
                    default:                         lzc_add = 5'd23; // unreachable (add_sum_ext!=0)
                endcase
                norm_ext = {1'b0, (add_sum_ext[24+W:0] << lzc_add)};
                norm_exp = result_exp_add_s2 - {3'b0, lzc_add};
            end
        end
    end

    logic [23:0] add_mant24;
    logic        add_guard, add_sticky;
    logic [23:0] add_mant24_rounded;
    logic        add_round_carry;
    logic [7:0]  result_exp_final;
    logic [22:0] result_mant_final;

    assign add_mant24 = norm_ext[W+23:W];
    assign add_guard  = norm_ext[W-1];
    assign add_sticky = |norm_ext[W-2:0];

    logic [24:0] add_mant25_rounded;
    always_comb begin
        add_mant25_rounded = {1'b0, add_mant24};
        if (add_guard && (add_sticky || add_mant24[0]))
            add_mant25_rounded = {1'b0, add_mant24} + 25'd1;
        add_round_carry = add_mant25_rounded[24];
    end

    logic [31:0] add_result_comb;

    always_comb begin
        if (a_add_zero_s2 && b_add_zero_s2) begin
            add_result_comb = 32'h00000000;
        end else if (a_add_zero_s2) begin
            add_result_comb = add_b_s2;
        end else if (b_add_zero_s2) begin
            add_result_comb = add_a_s2;
        end else if (add_is_zero) begin
            add_result_comb = 32'h00000000;
        end else if (lzc_add > norm_exp && !add_sum_ext[24+W]) begin
            // deep-cancellation underflow beyond representable exponent
            add_result_comb = 32'h00000000;
        end else begin
            result_exp_final = norm_exp + {7'h0, add_round_carry};
            if (add_round_carry)
                result_mant_final = add_mant25_rounded[23:1];
            else
                result_mant_final = add_mant25_rounded[22:0];
            add_result_comb = {add_result_sign, result_exp_final, result_mant_final};
        end
    end

    // Stage 2 output register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            result    <= 32'h00000000;
        end else begin
            valid_out <= valid_s2;
            result    <= add_result_comb;
        end
    end

endmodule
