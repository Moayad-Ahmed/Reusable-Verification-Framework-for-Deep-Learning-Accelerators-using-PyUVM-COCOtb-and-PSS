// verilog_lint: waive-start explicit-parameter-storage-type
// verilog_lint: waive-start unpacked-dimensions-range-ordering
// verilog_lint: waive-start line-length
// verilog_lint: waive-start module-filename

module CNN_top #(
    parameter ELEM_WIDTH       = 8,
    parameter MAX_IMG_HEIGHT   = 32,
    parameter MAX_IMG_WIDTH    = 32,
    parameter MAX_POOL_SIZE    = 4,
    // Convolution-specific parameters
    parameter MAX_IN_CHANNELS  = 3,
    parameter MAX_OUT_CHANNELS = 16,
    parameter MAX_KERNEL_SIZE  = 5,
    parameter MAX_WEIGHT_WIDTH = 8
) (
    input wire clk,
    input wire rst_n,

    // ───────────────── Convolution interface ─────────────────
    input  wire        conv_valid_in,
    input  wire [MAX_IN_CHANNELS*MAX_IMG_HEIGHT*MAX_IMG_WIDTH*ELEM_WIDTH-1:0] conv_data_in,
    input  wire [MAX_OUT_CHANNELS*MAX_IN_CHANNELS*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*MAX_WEIGHT_WIDTH-1:0] conv_kernel_weights,
    input  wire [7:0]  conv_kernel_size,
    input  wire [7:0]  conv_stride,
    input  wire [7:0]  conv_padding,
    input  wire [7:0]  conv_img_height,
    input  wire [7:0]  conv_img_width,
    input  wire [7:0]  conv_in_channels,
    input  wire [7:0]  conv_out_channels,
    input  wire [1:0]  conv_activation,
    output wire        conv_valid_out,
    output wire [MAX_OUT_CHANNELS*MAX_IMG_HEIGHT*MAX_IMG_WIDTH*ELEM_WIDTH-1:0] conv_data_out,

    // ───────────────── Pooling interface ─────────────────
    input  wire        pool_valid_in,
    input  wire [MAX_IMG_HEIGHT*MAX_IMG_WIDTH*ELEM_WIDTH-1:0] pool_data_in,
    input  wire [1:0]  pool_mode,
    input  wire [7:0]  pool_size,
    input  wire [7:0]  pool_stride_h,
    input  wire [7:0]  pool_stride_w,
    input  wire [7:0]  pool_img_height,
    input  wire [7:0]  pool_img_width,
    output wire        pool_valid_out,
    output wire [MAX_IMG_HEIGHT*MAX_IMG_WIDTH*ELEM_WIDTH-1:0] pool_data_out
);

    // ───────────────── Convolution instance ─────────────────
    convolution_layer_generic #(
        .ELEM_WIDTH      (ELEM_WIDTH),
        .MAX_IMG_HEIGHT  (MAX_IMG_HEIGHT),
        .MAX_IMG_WIDTH   (MAX_IMG_WIDTH),
        .MAX_IN_CHANNELS (MAX_IN_CHANNELS),
        .MAX_OUT_CHANNELS(MAX_OUT_CHANNELS),
        .MAX_KERNEL_SIZE (MAX_KERNEL_SIZE),
        .MAX_WEIGHT_WIDTH(MAX_WEIGHT_WIDTH)
    ) conv_inst (
        .clk            (clk),
        .rst_n          (rst_n),
        .valid_in       (conv_valid_in),
        .data_in        (conv_data_in),
        .kernel_weights (conv_kernel_weights),
        .kernel_size    (conv_kernel_size),
        .stride         (conv_stride),
        .padding        (conv_padding),
        .img_height     (conv_img_height),
        .img_width      (conv_img_width),
        .in_channels    (conv_in_channels),
        .out_channels   (conv_out_channels),
        .activation     (conv_activation),
        .valid_out      (conv_valid_out),
        .data_out       (conv_data_out)
    );

    // ───────────────── Pooling instance ─────────────────
    pooling_layer_generic #(
        .ELEM_WIDTH    (ELEM_WIDTH),
        .MAX_IMG_HEIGHT(MAX_IMG_HEIGHT),
        .MAX_IMG_WIDTH (MAX_IMG_WIDTH),
        .MAX_POOL_SIZE (MAX_POOL_SIZE)
    ) pool_inst (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (pool_valid_in),
        .data_in   (pool_data_in),
        .pool_mode (pool_mode),
        .pool_size (pool_size),
        .stride_h  (pool_stride_h),
        .stride_w  (pool_stride_w),
        .img_height(pool_img_height),
        .img_width (pool_img_width),
        .valid_out (pool_valid_out),
        .data_out  (pool_data_out)
    );

endmodule
