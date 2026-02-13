// ============================================================================
// Activation Functions for Deep Learning Accelerators
// Implements: ReLU, Sigmoid, Tanh, and Softmax
// Uses 8-bit signed integer arithmetic [-128, 127]
// Accepts full matrix input, processes element-wise, outputs full matrix
// ============================================================================

module activation_functions #(
    parameter DATA_WIDTH = 8,       // 8-bit signed integers
    parameter MATRIX_SIZE = 196      // Max elements in matrix (e.g., 14x14 = 196)
)(
    input wire clk,
    input wire rst_n,
    input wire [1:0] func_sel,      // 00: ReLU, 01: Sigmoid, 10: Tanh, 11: Softmax
    input wire [31:0] matrix_size,  // Actual number of elements (<= MATRIX_SIZE)
    input wire signed [DATA_WIDTH-1:0] data_in [0:MATRIX_SIZE-1],  // Full matrix input
    input wire valid_in,
    
    output reg signed [DATA_WIDTH-1:0] data_out [0:MATRIX_SIZE-1],  // Full matrix output
    output reg valid_out
);

    // Function select encoding
    localparam RELU    = 2'b00;
    localparam SIGMOID = 2'b01;
    localparam TANH    = 2'b10;
    localparam SOFTMAX = 2'b11;
    
    // 8-bit signed integer constants
    localparam signed [DATA_WIDTH-1:0] ZERO     = 8'sd0;
    localparam signed [DATA_WIDTH-1:0] MAX_VAL  = 8'sd127;
    localparam signed [DATA_WIDTH-1:0] MIN_VAL  = -8'sd128;
    
    // Intermediate signals for element-wise processing
    reg signed [DATA_WIDTH-1:0] relu_out [0:MATRIX_SIZE-1];
    reg signed [DATA_WIDTH-1:0] sigmoid_out [0:MATRIX_SIZE-1];
    reg signed [DATA_WIDTH-1:0] tanh_out [0:MATRIX_SIZE-1];
    reg signed [DATA_WIDTH-1:0] softmax_out [0:MATRIX_SIZE-1];
    
    // Loop variable
    integer i;
    
    // ========================================================================
    // ReLU Implementation: f(x) = max(0, x)
    // For signed 8-bit, clip negative values to 0
    // Process all elements in parallel
    // ========================================================================
    always @(*) begin
        for (i = 0; i < MATRIX_SIZE; i = i + 1) begin
            if (i < matrix_size) begin
                if (data_in[i] < ZERO)
                    relu_out[i] = ZERO;
                else
                    relu_out[i] = data_in[i];
            end else begin
                relu_out[i] = ZERO;
            end
        end
    end
    
    // ========================================================================
    // Sigmoid Approximation for 8-bit signed [-128, 127]
    // Maps input range [-128, 127] to output range [-128, 127]
    // Piecewise linear approximation:
    // f(x) = -96   if x < -64
    // f(x) = -32   if -64 <= x < 0
    // f(x) = 32    if 0 <= x < 64
    // f(x) = 96    if x >= 64
    // Process all elements in parallel
    // ========================================================================
    
    always @(*) begin
        for (i = 0; i < MATRIX_SIZE; i = i + 1) begin
            if (i < matrix_size) begin
                if (data_in[i] < -8'sd64) begin
                    sigmoid_out[i] = -8'sd96;
                end
                else if (data_in[i] < 8'sd0) begin
                    sigmoid_out[i] = -8'sd32;
                end
                else if (data_in[i] < 8'sd64) begin
                    sigmoid_out[i] = 8'sd32;
                end
                else begin
                    sigmoid_out[i] = 8'sd96;
                end
            end else begin
                sigmoid_out[i] = ZERO;
            end
        end
    end
    
    // ========================================================================
    // Tanh Approximation for 8-bit signed [-128, 127]
    // Maps input range [-128, 127] to output range [-128, 127]
    // Piecewise linear approximation:
    // f(x) = -112  if x < -64
    // f(x) = -48   if -64 <= x < 0
    // f(x) = 48    if 0 <= x < 64
    // f(x) = 112   if x >= 64
    // Process all elements in parallel
    // ========================================================================
    
    always @(*) begin
        for (i = 0; i < MATRIX_SIZE; i = i + 1) begin
            if (i < matrix_size) begin
                if (data_in[i] < -8'sd64) begin
                    tanh_out[i] = -8'sd112;
                end
                else if (data_in[i] < 8'sd0) begin
                    tanh_out[i] = -8'sd48;
                end
                else if (data_in[i] < 8'sd64) begin
                    tanh_out[i] = 8'sd48;
                end
                else begin
                    tanh_out[i] = 8'sd112;
                end
            end else begin
                tanh_out[i] = ZERO;
            end
        end
    end
    
    // ========================================================================
    // Softmax Approximation for 8-bit signed [-128, 127]
    // Simplified softmax: iterates over all elements and returns clamped(x - max_val)
    // This is a simplified version - real implementation would need exp() lookup
    // Process all elements: find global max, then subtract from each element
    // ========================================================================
    reg signed [DATA_WIDTH-1:0] max_val;
    reg signed [DATA_WIDTH:0] shifted_val;  // 9-bit signed for subtraction
    integer j;
    
    always @(*) begin
        // Find maximum value across active elements only
        max_val = data_in[0];
        for (j = 1; j < MATRIX_SIZE; j = j + 1) begin
            if (j < matrix_size) begin
                if (data_in[j] > max_val)
                    max_val = data_in[j];
            end
        end
        
        // Simplified softmax: subtract max from each active element and apply clamping
        for (j = 0; j < MATRIX_SIZE; j = j + 1) begin
            if (j < matrix_size) begin
                shifted_val = data_in[j] - max_val;
                
                // Clamp to signed range [-128, 127]
                if (shifted_val < MIN_VAL)
                    softmax_out[j] = MIN_VAL;
                else if (shifted_val > MAX_VAL)
                    softmax_out[j] = MAX_VAL;
                else
                    softmax_out[j] = shifted_val[DATA_WIDTH-1:0];
            end else begin
                softmax_out[j] = ZERO;
            end
        end
    end
    
    // ========================================================================
    // Register all outputs based on function select
    // ========================================================================
    integer k;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (k = 0; k < MATRIX_SIZE; k = k + 1) begin
                data_out[k] <= ZERO;
            end
            valid_out <= 1'b0;
        end
        else begin
            valid_out <= valid_in;
            
            case (func_sel)
                RELU: begin
                    for (k = 0; k < MATRIX_SIZE; k = k + 1)
                        data_out[k] <= (k < matrix_size) ? relu_out[k] : ZERO;
                end
                SIGMOID: begin
                    for (k = 0; k < MATRIX_SIZE; k = k + 1)
                        data_out[k] <= (k < matrix_size) ? sigmoid_out[k] : ZERO;
                end
                TANH: begin
                    for (k = 0; k < MATRIX_SIZE; k = k + 1)
                        data_out[k] <= (k < matrix_size) ? tanh_out[k] : ZERO;
                end
                SOFTMAX: begin
                    for (k = 0; k < MATRIX_SIZE; k = k + 1)
                        data_out[k] <= (k < matrix_size) ? softmax_out[k] : ZERO;
                end
                default: begin
                    for (k = 0; k < MATRIX_SIZE; k = k + 1)
                        data_out[k] <= ZERO;
                end
            endcase
        end
    end

endmodule


