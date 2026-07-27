// bf16_delta.sv
//
// Computes:
//     delta = |a - b|
// for BF16 values, used as delta = |m_old - s_i| in FlashAttention Stage-1.


module bf16_delta (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [15:0] a,
    input  logic [15:0] b,
    output logic [15:0] delta
);

    // STAGE A (combinational): ordering, unpack, align-shift
    
    logic        a_sign, b_sign;
    logic [14:0] a_mag, b_mag;
    logic        same_sign_comb;
    logic        a_eq_b_comb;
    logic        b_gt_a;

    logic [15:0] larger, smaller;
    logic [7:0]  larger_exp_comb, smaller_exp_comb;
    logic [7:0]  larger_mant, smaller_mant;
    logic [7:0]  exp_diff;
    logic [15:0] larger_mant_shifted_comb;
    logic [15:0] smaller_mant_shifted_comb;

    assign a_sign = a[15];
    assign b_sign = b[15];
    assign a_mag  = a[14:0];
    assign b_mag  = b[14:0];
    assign same_sign_comb = (a_sign == b_sign);
    assign a_eq_b_comb    = (a == b);

    assign b_gt_a = (b_mag > a_mag);

    always_comb begin
        if (b_gt_a) begin
            larger  = b;
            smaller = a;
        end else begin
            larger  = a;
            smaller = b;
        end
    end

    assign larger_exp_comb  = larger[14:7];
    assign smaller_exp_comb = smaller[14:7];
    assign larger_mant      = {1'b1, larger[6:0]};
    assign smaller_mant     = {1'b1, smaller[6:0]};

    always_comb begin
        exp_diff = larger_exp_comb - smaller_exp_comb;
        larger_mant_shifted_comb = {larger_mant, 8'h00};
        if (exp_diff >= 8'd16)
            smaller_mant_shifted_comb = 16'd0;
        else
            smaller_mant_shifted_comb = ({smaller_mant, 8'h00} >> exp_diff);
    end

    // --- Stage A registers ---
    logic        same_sign_r;
    logic        a_eq_b_r;
    logic [7:0]  larger_exp_r;
    logic [15:0] larger_mant_shifted_r;
    logic [15:0] smaller_mant_shifted_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            same_sign_r             <= 1'b0;
            a_eq_b_r                <= 1'b0;
            larger_exp_r            <= 8'd0;
            larger_mant_shifted_r   <= 16'd0;
            smaller_mant_shifted_r  <= 16'd0;
        end else begin
            same_sign_r             <= same_sign_comb;
            a_eq_b_r                <= a_eq_b_comb;
            larger_exp_r            <= larger_exp_comb;
            larger_mant_shifted_r   <= larger_mant_shifted_comb;
            smaller_mant_shifted_r  <= smaller_mant_shifted_comb;
        end
    end

    
    // STAGE B (combinational): SUBTRACT+LZC (same sign)
    //                       or ADD+carry-normalize (mixed sign)
    

    // --- SUBTRACT path (same sign) ---
    logic [15:0] sub_result;
    logic [15:0] sub_normalized;
    logic [7:0]  sub_exp;
    logic [6:0]  sub_mant;
    logic [3:0]  lzc;
    logic [15:0] sub_delta;

    always_comb begin
        sub_result     = larger_mant_shifted_r - smaller_mant_shifted_r;
        sub_normalized = 16'd0;
        sub_exp        = 8'd0;
        sub_mant       = 7'd0;
        lzc            = 4'd15;
        sub_delta      = 16'h0000;

        casez (sub_result)
            16'b1???????????????: lzc = 4'd0;
            16'b01??????????????: lzc = 4'd1;
            16'b001?????????????: lzc = 4'd2;
            16'b0001????????????: lzc = 4'd3;
            16'b00001???????????: lzc = 4'd4;
            16'b000001??????????: lzc = 4'd5;
            16'b0000001?????????: lzc = 4'd6;
            16'b00000001????????: lzc = 4'd7;
            16'b000000001???????: lzc = 4'd8;
            default:              lzc = 4'd15;
        endcase

        if ((sub_result == 16'd0) || (lzc == 4'd15)) begin
            sub_delta = 16'h0000;
        end else begin
            sub_normalized = sub_result << lzc;
            sub_exp        = larger_exp_r - lzc;
            sub_mant       = sub_normalized[14:8];
            sub_delta      = {1'b0, sub_exp, sub_mant};
        end
    end

    // --- ADD path (mixed sign) ---
    logic [16:0] add_result;
    logic [7:0]  add_exp;
    logic [6:0]  add_mant;
    logic [15:0] add_delta;

    always_comb begin
        add_result = {1'b0, larger_mant_shifted_r} + {1'b0, smaller_mant_shifted_r};
        add_exp    = 8'd0;
        add_mant   = 7'd0;

        if (add_result[16]) begin
            add_exp  = larger_exp_r + 8'd1;
            add_mant = add_result[15:9];
        end else begin
            add_exp  = larger_exp_r;
            add_mant = add_result[14:8];
        end

        add_delta = {1'b0, add_exp, add_mant};
    end

    // --- Final mux + output register ---
    logic [15:0] delta_comb;

    always_comb begin
        if (a_eq_b_r)
            delta_comb = 16'h0000;
        else if (same_sign_r)
            delta_comb = sub_delta;
        else
            delta_comb = add_delta;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            delta <= 16'h0000;
        else
            delta <= delta_comb;
    end

endmodule