// shift_lut_exp.sv

module shift_lut_exp (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        valid_in,
    input  logic [15:0] delta,       // BF16, ? >= 0 (guaranteed by caller)
    output logic        valid_out,
    output logic [11:0] exp_result   // Q1.11 fixed-point
);

  
    // LOG2E in BF16: 1.44269504... ? 0x3FB9
    // sign=0, exp=01111111 (127), mantissa=0111001 (0x39)
   
    localparam logic [15:0] LOG2E_BF16 = 16'h3FB9;

    // BF16 field offsets
    // [15]=sign [14:7]=exponent [6:0]=mantissa (7 bits, implicit leading 1)

  

    // Unpack delta
    logic        d_sign;
    logic [7:0]  d_exp;
    logic [7:0]  d_mant8;    // {1, delta[6:0]}

    assign d_sign  = delta[15];
    assign d_exp   = delta[14:7];
    assign d_mant8 = {1'b1, delta[6:0]};

    // LOG2E fields (constant, always positive)
    localparam logic [7:0]  C_EXP   = 8'd127;
    localparam logic [7:0]  C_MANT8 = 8'b10111001;  // {1, 0111001}

    // Mantissa multiply: 8x8 -> 16 bits 
    (* use_dsp = "yes" *)
    logic [15:0] mant_product;
    assign mant_product = d_mant8 * C_MANT8;

    // Exponent sum (9-bit to catch overflow): d_exp + C_EXP - 127
    logic [8:0] z_exp_raw;
    assign z_exp_raw = {1'b0, d_exp} + {1'b0, C_EXP} - 9'd127;

    
    logic [7:0]  z_exp_norm;
    logic [6:0]  z_mant_norm;

    always_comb begin
        if (mant_product[15]) begin
            
            z_exp_norm  = z_exp_raw[7:0] + 8'd1;
            z_mant_norm = mant_product[14:8];
        end else begin
            
            z_exp_norm  = z_exp_raw[7:0];
            z_mant_norm = mant_product[13:7];
        end
    end

    
    logic [15:0] z_bf16;
    assign z_bf16 = {1'b0, z_exp_norm, z_mant_norm};


    // step B: Saturation / underflow detection
    // Saturate if z >= 15.0
   
  
    localparam logic [14:0] BF16_15_MAGNITUDE = 15'h4170; // bits [14:0] of 15.0

    logic z_saturated;
    logic delta_zero;

    assign delta_zero  = (delta[14:0] == 15'h0);
    assign z_saturated = (~z_bf16[15]) && (z_bf16[14:0] >= BF16_15_MAGNITUDE);

  
// step C: BF16 ? Q4.5 fixed-point conversion

logic [7:0]  z_mant8;
logic [7:0]  z_exp_field;

logic signed [8:0] align;

logic [8:0]  z_fixed;        // Q4.5
logic [3:0]  shift_amt;


logic [16:0] shift_value;

assign z_exp_field = z_bf16[14:7];
assign z_mant8     = {1'b1, z_bf16[6:0]};

assign align =
    $signed({1'b0,z_exp_field})
    -
    9'sd129;

always_comb begin

    shift_amt   = 4'd0;
    shift_value = 17'd0;
    z_fixed     = 9'h000;

   
    // Delta = 0
    // e^0 = 1
   

    if (delta_zero) begin

        z_fixed = 9'h000;

    end

   
    // Saturated
    // e^-large ? 0
 

    else if (z_saturated) begin

        z_fixed = 9'h1FF;

    end

    // Zero / denormal BF16
  

    else if (z_exp_field == 8'h00) begin

        z_fixed = 9'h000;

    end

  
    // Left Shift
    

    else if (align >= 0) begin

        shift_amt = align[3:0];

        if (align >= 9'sd9) begin

            z_fixed = 9'h1FF;

        end

        else begin

            shift_value =
                ({8'd0,1'b0,z_mant8} << shift_amt);

            if (shift_value > 17'd511)

                z_fixed = 9'h1FF;

            else

                z_fixed = shift_value[8:0];

        end

    end

   
    // Right Shift
  

    else begin

        shift_amt = $unsigned(-align);

        z_fixed =
            ({1'b0,z_mant8} >> shift_amt);

    end

end


// Split Q4.5

logic [3:0] z_int;
logic [4:0] z_frac;

assign z_int  = z_fixed[8:5];
assign z_frac = z_fixed[4:0];

// Step D: ROM lookup
   
    
logic [11:0] rom_out;
always_comb begin
    case (z_frac)
        5'd0:  rom_out = 12'h800;
        5'd1:  rom_out = 12'h7D4;
        5'd2:  rom_out = 12'h7A9;
        5'd3:  rom_out = 12'h77F;
        5'd4:  rom_out = 12'h756;
        5'd5:  rom_out = 12'h72E;
        5'd6:  rom_out = 12'h706;
        5'd7:  rom_out = 12'h6E0;
        5'd8:  rom_out = 12'h6BA;
        5'd9:  rom_out = 12'h695;
        5'd10: rom_out = 12'h671;
        5'd11: rom_out = 12'h64E;
        5'd12: rom_out = 12'h62B;
        5'd13: rom_out = 12'h609;
        5'd14: rom_out = 12'h5E8;
        5'd15: rom_out = 12'h5C8;
        5'd16: rom_out = 12'h5A8;
        5'd17: rom_out = 12'h589;
        5'd18: rom_out = 12'h56B;
        5'd19: rom_out = 12'h54D;
        5'd20: rom_out = 12'h530;
        5'd21: rom_out = 12'h514;
        5'd22: rom_out = 12'h4F8;
        5'd23: rom_out = 12'h4DC;
        5'd24: rom_out = 12'h4C2;
        5'd25: rom_out = 12'h4A8;
        5'd26: rom_out = 12'h48E;
        5'd27: rom_out = 12'h475;
        5'd28: rom_out = 12'h45D;
        5'd29: rom_out = 12'h445;
        5'd30: rom_out = 12'h42D;
        5'd31: rom_out = 12'h416;
        default: rom_out = 12'h800; // unreachable: z_frac is 5 bits, all 32 values covered above
    endcase
end

  
    // Step E: Barrel right-shift by z_int
    
    logic [11:0] shifted_result;

    always_comb begin
        if (z_saturated) begin
            shifted_result = 12'h000;
        end else if (delta_zero) begin
            shifted_result = 12'h800;  // 1.0 in Q1.11
        end else begin
            shifted_result = rom_out >> z_int;
        end
    end

    
    // Step F: Register output
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out  <= 1'b0;
            exp_result <= 12'h000;
        end else begin
            valid_out  <= valid_in;
            exp_result <= shifted_result;
        end
    end

endmodule
