module pooling_layer_generic #(
    parameter ELEM_WIDTH = 8,               // Width of each element
    parameter POOL_SIZE = 2,                // Pooling window size (NxN)
    parameter STRIDE_H = 2,                 // Vertical stride
    parameter STRIDE_W = 2,                 // Horizontal stride
    parameter IMG_HEIGHT = 28,               // Input image height
    parameter IMG_WIDTH = 28,                // Input image width
    
    // Derived parameters
    parameter NUM_INPUT_ELEMENTS = IMG_HEIGHT * IMG_WIDTH,
    parameter OUT_HEIGHT = (IMG_HEIGHT - POOL_SIZE) / STRIDE_H + 1,
    parameter OUT_WIDTH = (IMG_WIDTH - POOL_SIZE) / STRIDE_W + 1,
    parameter NUM_OUTPUT_ELEMENTS = OUT_HEIGHT * OUT_WIDTH,
    parameter IN_DATA_WIDTH = NUM_INPUT_ELEMENTS * ELEM_WIDTH,
    parameter OUT_DATA_WIDTH = NUM_OUTPUT_ELEMENTS * ELEM_WIDTH
) (
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [IN_DATA_WIDTH-1:0] data_in,
    input wire [1:0] pool_mode,             // 0: MAX, 1: AVG, 2: MIN

    output reg valid_out,
    output reg [OUT_DATA_WIDTH-1:0] data_out
);

    localparam NUM_POOL_ELEMENTS = POOL_SIZE * POOL_SIZE;

    // Helper function: Get element from flattened input
    function [ELEM_WIDTH-1:0] get_element;
        input [IN_DATA_WIDTH-1:0] data;
        input integer row;
        input integer col;
        integer index;
        begin
            index = row * IMG_WIDTH + col;
            get_element = data[index*ELEM_WIDTH +: ELEM_WIDTH];
        end
    endfunction

    // Helper function: max of two values
    function [ELEM_WIDTH-1:0] max2;
        input [ELEM_WIDTH-1:0] a, b;
        begin
            max2 = (a > b) ? a : b;
        end
    endfunction

    // Helper function: min of two values
    function [ELEM_WIDTH-1:0] min2;
        input [ELEM_WIDTH-1:0] a, b;
        begin
            min2 = (a < b) ? a : b;
        end
    endfunction

    // Function to compute max pooling for one output element
    function [ELEM_WIDTH-1:0] compute_max;
        input [IN_DATA_WIDTH-1:0] data;
        input integer start_row;
        input integer start_col;
        reg [ELEM_WIDTH-1:0] max_val;
        reg [ELEM_WIDTH-1:0] curr_val;
        integer m, n;
        begin
            max_val = get_element(data, start_row, start_col);
            for (m = 0; m < POOL_SIZE; m = m + 1) begin
                for (n = 0; n < POOL_SIZE; n = n + 1) begin
                    curr_val = get_element(data, start_row + m, start_col + n);
                    max_val = max2(max_val, curr_val);
                end
            end
            compute_max = max_val;
        end
    endfunction

    // Function to compute min pooling for one output element
    function [ELEM_WIDTH-1:0] compute_min;
        input [IN_DATA_WIDTH-1:0] data;
        input integer start_row;
        input integer start_col;
        reg [ELEM_WIDTH-1:0] min_val;
        reg [ELEM_WIDTH-1:0] curr_val;
        integer m, n;
        begin
            min_val = get_element(data, start_row, start_col);
            for (m = 0; m < POOL_SIZE; m = m + 1) begin
                for (n = 0; n < POOL_SIZE; n = n + 1) begin
                    curr_val = get_element(data, start_row + m, start_col + n);
                    min_val = min2(min_val, curr_val);
                end
            end
            compute_min = min_val;
        end
    endfunction

    // Function to compute avg pooling for one output element
    function [ELEM_WIDTH-1:0] compute_avg;
        input [IN_DATA_WIDTH-1:0] data;
        input integer start_row;
        input integer start_col;
        reg [ELEM_WIDTH+7:0] sum;  // Extra bits for sum
        integer m, n;
        begin
            sum = 0;
            for (m = 0; m < POOL_SIZE; m = m + 1) begin
                for (n = 0; n < POOL_SIZE; n = n + 1) begin
                    sum = sum + get_element(data, start_row + m, start_col + n);
                end
            end
            compute_avg = sum / NUM_POOL_ELEMENTS;
        end
    endfunction

    // Function to compute full max pooling output
    function [OUT_DATA_WIDTH-1:0] max_pooling;
        input [IN_DATA_WIDTH-1:0] data;
        integer out_row, out_col, out_idx;
        integer in_row, in_col;
        begin
            max_pooling = {OUT_DATA_WIDTH{1'b0}};
            for (out_row = 0; out_row < OUT_HEIGHT; out_row = out_row + 1) begin
                for (out_col = 0; out_col < OUT_WIDTH; out_col = out_col + 1) begin
                    in_row = out_row * STRIDE_H;
                    in_col = out_col * STRIDE_W;
                    out_idx = out_row * OUT_WIDTH + out_col;
                    max_pooling[out_idx*ELEM_WIDTH +: ELEM_WIDTH] = compute_max(data, in_row, in_col);
                end
            end
        end
    endfunction

    // Function to compute full min pooling output
    function [OUT_DATA_WIDTH-1:0] min_pooling;
        input [IN_DATA_WIDTH-1:0] data;
        integer out_row, out_col, out_idx;
        integer in_row, in_col;
        begin
            min_pooling = {OUT_DATA_WIDTH{1'b0}};
            for (out_row = 0; out_row < OUT_HEIGHT; out_row = out_row + 1) begin
                for (out_col = 0; out_col < OUT_WIDTH; out_col = out_col + 1) begin
                    in_row = out_row * STRIDE_H;
                    in_col = out_col * STRIDE_W;
                    out_idx = out_row * OUT_WIDTH + out_col;
                    min_pooling[out_idx*ELEM_WIDTH +: ELEM_WIDTH] = compute_min(data, in_row, in_col);
                end
            end
        end
    endfunction

    // Function to compute full avg pooling output
    function [OUT_DATA_WIDTH-1:0] avg_pooling;
        input [IN_DATA_WIDTH-1:0] data;
        integer out_row, out_col, out_idx;
        integer in_row, in_col;
        begin
            avg_pooling = {OUT_DATA_WIDTH{1'b0}};
            for (out_row = 0; out_row < OUT_HEIGHT; out_row = out_row + 1) begin
                for (out_col = 0; out_col < OUT_WIDTH; out_col = out_col + 1) begin
                    in_row = out_row * STRIDE_H;
                    in_col = out_col * STRIDE_W;
                    out_idx = out_row * OUT_WIDTH + out_col;
                    avg_pooling[out_idx*ELEM_WIDTH +: ELEM_WIDTH] = compute_avg(data, in_row, in_col);
                end
            end
        end
    endfunction

    // Main pooling logic - single clocked always block
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            data_out <= {OUT_DATA_WIDTH{1'b0}};
        end else begin
            if (valid_in) begin
                case (pool_mode)
                    2'b00: data_out <= max_pooling(data_in);
                    2'b01: data_out <= avg_pooling(data_in);
                    2'b10: data_out <= min_pooling(data_in);
                    default: data_out <= {OUT_DATA_WIDTH{1'b0}};
                endcase
                valid_out <= 1'b1;
            end else begin
                valid_out <= 1'b0;
                data_out <= {OUT_DATA_WIDTH{1'b0}};
            end
        end
    end

endmodule
