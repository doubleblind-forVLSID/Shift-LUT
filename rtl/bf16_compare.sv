// bf16_compare.sv
// Compares two BF16 values, returns max and a flag indicating if b > a.
// BF16: [15] sign, [14:7] exponent, [6:0] mantissa
//
// Handles:
//   - Normal numbers
//   - Negative inputs (sign bit aware)
//   - Equal inputs (new_max_flag = 0, preserves m_old)
//
// Does NOT handle NaN or Inf â€” inputs assumed valid finite BF16.

module bf16_compare (
    input  logic [15:0] a,          // m_old (current running max)
    input  logic [15:0] b,          // s_i   (incoming score)
    output logic [15:0] max_out,    // max(a, b)
    output logic        new_max_flag // 1 if b > a (new maximum found)
);

    // BF16 comparison: treat as signed magnitude
    // For two positive BF16 values, direct unsigned comparison of [14:0] works
    // because the IEEE 754 bit pattern preserves order for same-sign values.
    // For mixed signs: negative < positive always.

    logic a_sign, b_sign;
    logic [14:0] a_mag, b_mag;
    logic b_gt_a;

    assign a_sign = a[15];
    assign b_sign = b[15];
    assign a_mag  = a[14:0];
    assign b_mag  = b[14:0];

    always_comb begin
        if (a_sign & ~b_sign) begin
            // a negative, b positive â†’ b > a
            b_gt_a = 1'b1;
        end else if (~a_sign & b_sign) begin
            // a positive, b negative â†’ a > b
            b_gt_a = 1'b0;
        end else if (~a_sign & ~b_sign) begin
            // both positive: larger magnitude = larger value
            b_gt_a = (b_mag > a_mag);
        end else begin
            // both negative: larger magnitude = smaller value
            b_gt_a = (b_mag < a_mag);
        end
    end

    assign new_max_flag = b_gt_a;
    assign max_out      = b_gt_a ? b : a;

endmodule
