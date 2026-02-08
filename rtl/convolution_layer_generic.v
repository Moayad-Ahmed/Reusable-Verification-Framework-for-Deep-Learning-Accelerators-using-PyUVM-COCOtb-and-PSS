// verilog_lint: waive-start explicit-parameter-storage-type
// verilog_lint: waive-start unpacked-dimensions-range-ordering
// verilog_lint: waive-start line-length

module convolution_layer_generic #(
    parameter ELEM_WIDTH = 8,               // Width of each element
    parameter MAX_IMG_HEIGHT = 32,          // Maximum input image height
    parameter MAX_IMG_WIDTH = 32,           // Maximum input image width
    parameter MAX_IN_CHANNELS = 3,          // Maximum input channels
    parameter MAX_OUT_CHANNELS = 16,        // Maximum output channels
    parameter MAX_KERNEL_SIZE = 5,          // Maximum kernel size
    parameter MAX_WEIGHT_WIDTH = 8,         // Width of each weight element
    parameter WEIGHT_SCALE = 127,           // Quantization scale for weights

    // Derived parameters for maximum sizes
    parameter MAX_NUM_INPUT_ELEMENTS = MAX_IN_CHANNELS * MAX_IMG_HEIGHT * MAX_IMG_WIDTH,
    parameter MAX_OUT_HEIGHT = MAX_IMG_HEIGHT,
    parameter MAX_OUT_WIDTH = MAX_IMG_WIDTH,
    parameter MAX_NUM_OUTPUT_ELEMENTS = MAX_OUT_CHANNELS * MAX_OUT_HEIGHT * MAX_OUT_WIDTH,
    parameter MAX_IN_DATA_WIDTH = MAX_NUM_INPUT_ELEMENTS * ELEM_WIDTH,
    parameter MAX_OUT_DATA_WIDTH = MAX_NUM_OUTPUT_ELEMENTS * ELEM_WIDTH,
    parameter MAX_KERNEL_ELEMENTS = MAX_OUT_CHANNELS * MAX_IN_CHANNELS * MAX_KERNEL_SIZE * MAX_KERNEL_SIZE,
    parameter MAX_KERNEL_WIDTH = MAX_KERNEL_ELEMENTS * MAX_WEIGHT_WIDTH
) (
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [MAX_IN_DATA_WIDTH-1:0] data_in,
    input wire [MAX_KERNEL_WIDTH-1:0] kernel_weights,

    // Runtime configurable parameters as inputs
    input wire [7:0] kernel_size,           // Convolution kernel size (NxN)
    input wire [7:0] stride,                // Stride (same for both H and W)
    input wire [7:0] padding,               // Padding (same for all sides)
    input wire [7:0] img_height,            // Input image height
    input wire [7:0] img_width,             // Input image width
    input wire [7:0] in_channels,           // Number of input channels
    input wire [7:0] out_channels,          // Number of output channels

    output reg valid_out,
    output reg [MAX_OUT_DATA_WIDTH-1:0] data_out
);

    // Calculate derived values from inputs
    wire [7:0] out_height = (img_height + 2*padding - kernel_size) / stride + 1;
    wire [7:0] out_width = (img_width + 2*padding - kernel_size) / stride + 1;

    // Internal variables for convolution computation
    integer out_ch, out_row, out_col, out_idx;
    integer in_ch, k_row, k_col;
    integer in_row, in_col, in_idx, kernel_idx;

    // Input is UNSIGNED (0-255)
    reg [ELEM_WIDTH-1:0] input_val;

    // Weight is SIGNED (-128 to 127)
    reg signed [MAX_WEIGHT_WIDTH-1:0] weight_val;

    // Use wider accumulator to prevent overflow
    reg signed [31:0] conv_sum;      // Accumulator (before scaling)
    reg signed [31:0] product;       // Temporary for multiplication
    reg signed [31:0] scaled_sum;    // After division by WEIGHT_SCALE

    reg [ELEM_WIDTH-1:0] output_val;  // Output is unsigned

    // Main convolution logic - single clocked always block
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            data_out <= {MAX_OUT_DATA_WIDTH{1'b0}};
        end else begin
            if (valid_in) begin
                // Initialize output
                data_out <= {MAX_OUT_DATA_WIDTH{1'b0}};

                // Iterate over all output channels
                for (out_ch = 0; out_ch < MAX_OUT_CHANNELS; out_ch = out_ch + 1) begin
                    if (out_ch < out_channels) begin
                        // Iterate over output spatial dimensions
                        for (out_row = 0; out_row < MAX_OUT_HEIGHT; out_row = out_row + 1) begin
                            for (out_col = 0; out_col < MAX_OUT_WIDTH; out_col = out_col + 1) begin
                                if (out_row < out_height && out_col < out_width) begin
                                    conv_sum = 0;

                                    // Iterate over input channels
                                    for (in_ch = 0; in_ch < MAX_IN_CHANNELS; in_ch = in_ch + 1) begin
                                        if (in_ch < in_channels) begin
                                            // Iterate over kernel
                                            for (k_row = 0; k_row < MAX_KERNEL_SIZE; k_row = k_row + 1) begin
                                                for (k_col = 0; k_col < MAX_KERNEL_SIZE; k_col = k_col + 1) begin
                                                    if (k_row < kernel_size && k_col < kernel_size) begin
                                                        // Calculate input position (with padding consideration)
                                                        in_row = out_row * stride + k_row - padding;
                                                        in_col = out_col * stride + k_col - padding;

                                                        // Check if within valid input bounds (handle padding)
                                                        if (in_row >= 0 && in_row < img_height &&
                                                            in_col >= 0 && in_col < img_width) begin

                                                            // Get input value (UNSIGNED)
                                                            in_idx = in_ch * img_height * img_width +
                                                                    in_row * img_width + in_col;
                                                            input_val = data_in[in_idx*ELEM_WIDTH +: ELEM_WIDTH];

                                                            // Get kernel weight (SIGNED)
                                                            kernel_idx = out_ch * in_channels * kernel_size * kernel_size +
                                                                        in_ch * kernel_size * kernel_size +
                                                                        k_row * kernel_size + k_col;
                                                            weight_val = kernel_weights[kernel_idx*MAX_WEIGHT_WIDTH +: MAX_WEIGHT_WIDTH];

                                                            // Proper signed multiplication
                                                            // Cast unsigned input_val to signed for multiplication
                                                            product = $signed({1'b0, input_val}) * weight_val;
                                                            conv_sum = conv_sum + product;
                                                        end
                                                        // If padding area, input is implicitly 0
                                                    end
                                                end
                                            end
                                        end
                                    end

                                    // **CRITICAL FIX: Scale by dividing by WEIGHT_SCALE (127)**
                                    // This matches the Python dequantization: kernel / 127.0
                                    scaled_sum = conv_sum / WEIGHT_SCALE;

                                    // Apply clamping
                                    if (scaled_sum > 255)
                                        output_val = 8'd255;
                                    else if (scaled_sum < 0)
                                        output_val = 8'd0;
                                    else
                                        output_val = scaled_sum[ELEM_WIDTH-1:0];

                                    // Store result
                                    out_idx = out_ch * out_height * out_width +
                                             out_row * out_width + out_col;
                                    data_out[out_idx*ELEM_WIDTH +: ELEM_WIDTH] <= output_val;
                                end
                            end
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
