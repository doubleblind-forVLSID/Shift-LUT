// accumulator_update.sv
//
// Implements the FlashAttention online-softmax recurrence:
//   l_new = l_old * r + n
//   O_new = O_old * r + n * value_in


module accumulator_update (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        in_valid,
    input  logic [31:0] l_old,
    input  logic [31:0] O_old,
    input  logic [31:0] r_fp32,
    input  logic [31:0] n_fp32,
    input  logic [31:0] value_in_fp32,

    output logic        out_valid,
    output logic [31:0] l_new,
    output logic [31:0] O_new
);

    
    // Stage A: two independent pipelined MACs, both launched at t=0
    
    logic        mac_l_valid_out;
    logic [31:0] mac_l_result;      // this IS l_new, once delay-aligned

    logic        mac_nval_valid_out;
    logic [31:0] mac_nval_result;   // n * value_in, Stage B's addend

    fp32_mac u_mac_l (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (in_valid),
        .a         (l_old),
        .b         (r_fp32),
        .c         (n_fp32),
        .valid_out (mac_l_valid_out),
        .result    (mac_l_result)
    );

    fp32_mac u_mac_nval (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (in_valid),
        .a         (n_fp32),
        .b         (value_in_fp32),
        .c         (32'h00000000),
        .valid_out (mac_nval_valid_out),
        .result    (mac_nval_result)
    );

    
    
    
    logic [31:0] O_old_d1, O_old_d2, O_old_d3, O_old_d4;
    logic [31:0] r_d1, r_d2, r_d3, r_d4;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            O_old_d1 <= 32'h00000000;
            O_old_d2 <= 32'h00000000;
            O_old_d3 <= 32'h00000000;
            O_old_d4 <= 32'h00000000;
            r_d1     <= 32'h00000000;
            r_d2     <= 32'h00000000;
            r_d3     <= 32'h00000000;
            r_d4     <= 32'h00000000;
        end else begin
            O_old_d1 <= O_old;
            O_old_d2 <= O_old_d1;
            O_old_d3 <= O_old_d2;
            O_old_d4 <= O_old_d3;
            r_d1     <= r_fp32;
            r_d2     <= r_d1;
            r_d3     <= r_d2;
            r_d4     <= r_d3;
        end
    end

    
    
    
    logic [31:0] l_new_d1, l_new_d2, l_new_d3, l_new_d4;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            l_new_d1 <= 32'h00000000;
            l_new_d2 <= 32'h00000000;
            l_new_d3 <= 32'h00000000;
            l_new_d4 <= 32'h00000000;
        end else begin
            l_new_d1 <= mac_l_result;
            l_new_d2 <= l_new_d1;
            l_new_d3 <= l_new_d2;
            l_new_d4 <= l_new_d3;
        end
    end

    
    // Stage B: launches when mac_nval_valid_out fires, using the
    // delay-aligned O_old/r_fp32 and the fresh nval result directly.
    
    logic        mac_o_valid_out;
    logic [31:0] mac_o_result;

    fp32_mac u_mac_o (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (mac_nval_valid_out),
        .a         (O_old_d4),
        .b         (r_d4),
        .c         (mac_nval_result),
        .valid_out (mac_o_valid_out),
        .result    (mac_o_result)
    );

        
    assign out_valid = mac_o_valid_out;
    assign O_new      = mac_o_result;
    assign l_new       = l_new_d4;

endmodule