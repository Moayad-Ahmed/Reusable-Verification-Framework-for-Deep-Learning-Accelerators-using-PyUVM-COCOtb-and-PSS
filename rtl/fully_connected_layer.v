// verilog_lint: waive-start explicit-parameter-storage-type
// verilog_lint: waive-start unpacked-dimensions-range-ordering
// verilog_lint: waive-start generate-label

module fully_connected_layer #(
    parameter INPUT_SIZE  = 128,  // Maximum input size
    parameter OUTPUT_SIZE = 10    // Maximum output size
)(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   en,

    // Runtime actual sizes (must be <= the corresponding parameter)
    input  wire [31:0]            actual_input_size,
    input  wire [31:0]            actual_output_size,

    // Flattened input data interfaces (sized to maximum)
    input  wire [INPUT_SIZE*8-1:0]  in_vec,
    input  wire [OUTPUT_SIZE*INPUT_SIZE*8-1:0] weights,
    input  wire [OUTPUT_SIZE*8-1:0] bias,

    output reg  [OUTPUT_SIZE*8-1:0] out_vec,
    output reg                    valid
);

    integer i, j;
    reg signed [7:0] accumulator; // 8-bit wrapping accumulator

    // Internal signed representations (allocated to maximum size)
    wire signed [7:0] signed_in    [0:INPUT_SIZE-1];
    wire signed [7:0] signed_w     [0:OUTPUT_SIZE-1][0:INPUT_SIZE-1];
    wire signed [7:0] signed_bias  [0:OUTPUT_SIZE-1];

    // Unpack flattened arrays (up to maximum size)
    genvar g_i, g_j;
    generate
        for (g_i = 0; g_i < INPUT_SIZE; g_i = g_i + 1) begin
            assign signed_in[g_i] = in_vec[g_i*8 +: 8];
        end
        for (g_i = 0; g_i < OUTPUT_SIZE; g_i = g_i + 1) begin
            assign signed_bias[g_i] = bias[g_i*8 +: 8];
            for (g_j = 0; g_j < INPUT_SIZE; g_j = g_j + 1) begin
                assign signed_w[g_i][g_j] = weights[(g_i*INPUT_SIZE + g_j)*8 +: 8];
            end
        end
    endgenerate

    // Computation Logic — loops use actual sizes, not maximum parameters
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_vec <= 0;
            valid   <= 0;
        end else if (en) begin
            // Zero the entire output first
            out_vec <= 0;

            for (i = 0; i < OUTPUT_SIZE; i = i + 1) begin
                if (i < actual_output_size) begin
                    // Start with the bias
                    accumulator = $signed(signed_bias[i]);

                    // Dot product over actual input size
                    for (j = 0; j < INPUT_SIZE; j = j + 1) begin
                        if (j < actual_input_size) begin
                            accumulator = accumulator + (signed_in[j] * signed_w[i][j]);
                        end
                    end

                    // Direct 8-bit wrapping output
                    out_vec[i*8 +: 8] <= accumulator;
                end
            end
            valid <= 1;
        end else begin
            valid <= 0;
        end
    end

endmodule
