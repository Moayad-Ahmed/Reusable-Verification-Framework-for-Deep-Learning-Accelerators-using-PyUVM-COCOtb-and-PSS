// verilog_lint: waive-start explicit-parameter-storage-type
// verilog_lint: waive-start unpacked-dimensions-range-ordering
// verilog_lint: waive-start line-length

module pooling_layer_generic #(
    parameter ELEM_WIDTH = 8,               // Width of each element
    parameter MAX_IMG_HEIGHT = 32,          // Maximum input image height
    parameter MAX_IMG_WIDTH = 32,           // Maximum input image width
    parameter MAX_POOL_SIZE = 4,            // Maximum pooling window size

    // Derived parameters for maximum sizes
    parameter MAX_NUM_INPUT_ELEMENTS = MAX_IMG_HEIGHT * MAX_IMG_WIDTH,
    parameter MAX_OUT_HEIGHT = MAX_IMG_HEIGHT,
    parameter MAX_OUT_WIDTH = MAX_IMG_WIDTH,
    parameter MAX_NUM_OUTPUT_ELEMENTS = MAX_OUT_HEIGHT * MAX_OUT_WIDTH,
    parameter MAX_IN_DATA_WIDTH = MAX_NUM_INPUT_ELEMENTS * ELEM_WIDTH,
    parameter MAX_OUT_DATA_WIDTH = MAX_NUM_OUTPUT_ELEMENTS * ELEM_WIDTH
) (
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [MAX_IN_DATA_WIDTH-1:0] data_in,
    input wire [1:0] pool_mode,             // 0: MAX, 1: AVG, 2: MIN

    // Runtime configurable parameters as inputs
    input wire [7:0] pool_size,             // Pooling window size (NxN)
    input wire [7:0] stride_h,              // Vertical stride
    input wire [7:0] stride_w,              // Horizontal stride
    input wire [7:0] img_height,            // Input image height
    input wire [7:0] img_width,             // Input image width

    output reg valid_out,
    output reg [MAX_OUT_DATA_WIDTH-1:0] data_out
);

    // Calculate derived values from inputs
    wire [7:0] out_height = (img_height - pool_size) / stride_h + 1;
    wire [7:0] out_width = (img_width - pool_size) / stride_w + 1;
    wire [15:0] num_pool_elements = pool_size * pool_size;

    // Internal registers for pooling computation
    reg [ELEM_WIDTH-1:0] max_result [0:MAX_OUT_HEIGHT*MAX_OUT_WIDTH-1];
    reg [ELEM_WIDTH-1:0] min_result [0:MAX_OUT_HEIGHT*MAX_OUT_WIDTH-1];
    reg [ELEM_WIDTH-1:0] avg_result [0:MAX_OUT_HEIGHT*MAX_OUT_WIDTH-1];

    integer out_row, out_col, out_idx;
    integer in_row, in_col, idx;
    integer m, n;
    reg [ELEM_WIDTH-1:0] max_val, min_val, curr_val;
    reg [ELEM_WIDTH+15:0] sum;  // Extra bits for sum to avoid overflow

    // Main pooling logic - single clocked always block
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            data_out <= {MAX_OUT_DATA_WIDTH{1'b0}};
        end else begin
            if (valid_in) begin
                // Initialize output
                data_out <= {MAX_OUT_DATA_WIDTH{1'b0}};

                // Iterate over all output positions
                for (out_row = 0; out_row < MAX_OUT_HEIGHT; out_row = out_row + 1) begin
                    for (out_col = 0; out_col < MAX_OUT_WIDTH; out_col = out_col + 1) begin
                        if (out_row < out_height && out_col < out_width) begin
                            in_row = out_row * stride_h;
                            in_col = out_col * stride_w;
                            out_idx = out_row * out_width + out_col;

                            // Initialize for this pooling window
                            idx = in_row * img_width + in_col;
                            max_val = data_in[idx*ELEM_WIDTH +: ELEM_WIDTH];
                            min_val = data_in[idx*ELEM_WIDTH +: ELEM_WIDTH];
                            sum = 0;

                            // Iterate over pooling window
                            for (m = 0; m < MAX_POOL_SIZE; m = m + 1) begin
                                for (n = 0; n < MAX_POOL_SIZE; n = n + 1) begin
                                    if (m < pool_size && n < pool_size) begin
                                        idx = (in_row + m) * img_width + (in_col + n);
                                        curr_val = data_in[idx*ELEM_WIDTH +: ELEM_WIDTH];

                                        // Max pooling
                                        if (curr_val > max_val)
                                            max_val = curr_val;

                                        // Min pooling
                                        if (curr_val < min_val)
                                            min_val = curr_val;

                                        // Sum for average
                                        sum = sum + curr_val;
                                    end
                                end
                            end

                            // Store result based on pool_mode
                            case (pool_mode)
                                2'b00: data_out[out_idx*ELEM_WIDTH +: ELEM_WIDTH] <= max_val;
                                2'b01: data_out[out_idx*ELEM_WIDTH +: ELEM_WIDTH] <= sum / num_pool_elements;
                                2'b10: data_out[out_idx*ELEM_WIDTH +: ELEM_WIDTH] <= min_val;
                                default: data_out[out_idx*ELEM_WIDTH +: ELEM_WIDTH] <= 8'b0;
                            endcase
                        end
                    end
                end
                valid_out <= 1'b1;
            end else begin
                valid_out <= 1'b0;
                data_out <= {MAX_OUT_DATA_WIDTH{1'b0}};
            end
        end
    end

endmodule
