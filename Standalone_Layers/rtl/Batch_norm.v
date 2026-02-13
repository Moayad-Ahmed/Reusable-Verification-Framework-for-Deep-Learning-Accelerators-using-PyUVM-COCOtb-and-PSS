module batch_norm_int8 #(
    parameter DATA_WIDTH = 8
)(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   en,
    
    // Data and Parameter Inputs for the current element
    input  wire signed [DATA_WIDTH-1:0] in_data,
    input  wire signed [DATA_WIDTH-1:0] in_mean,
    input  wire signed [DATA_WIDTH-1:0] in_var,   // Treated as a pre-calc scale in some designs
    input  wire signed [DATA_WIDTH-1:0] in_gamma,
    input  wire signed [DATA_WIDTH-1:0] in_beta,
    
    output reg  signed [DATA_WIDTH-1:0] out_data,
    output reg                          valid
);

    // Internal 32-bit registers for high-precision intermediate math
    reg signed [31:0] sub_res;
    reg signed [31:0] mul_res;
    reg signed [31:0] final_res;

    // Fixed-point Epsilon approximation (for demonstration)
    localparam EPS = 1; 

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_data  <= 0;
            valid     <= 0;
            sub_res   <= 0;
            mul_res   <= 0;
            final_res <= 0;
        end else if (en) begin
            // Stage 1: Centering (x - mean)
            sub_res <= $signed(in_data) - $signed(in_mean);
            
            // Stage 2: Scaling (Simplified: multiply by gamma, ignoring complex sqrt/div for this generic logic)
            // Note: In a production DUT, 'in_var' would be a pre-calculated 1/sqrt(var+eps)
            mul_res <= (sub_res * $signed(in_gamma));
            
            // Stage 3: Shifting (+ beta)
            final_res <= mul_res + $signed(in_beta);

            // Stage 4: Saturation Logic (INT8)
            if (final_res > 127)
                out_data <= 8'd127;
            else if (final_res < -128)
                out_data <= 8'd-128;
            else
                out_data <= final_res[7:0];

            valid <= 1;
        end else begin
            valid <= 0;
        end
    end
endmodule