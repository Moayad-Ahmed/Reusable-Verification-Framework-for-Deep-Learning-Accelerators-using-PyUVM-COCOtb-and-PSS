// verilog_lint: waive-start module-filename
// verilog_lint: waive-start explicit-parameter-storage-type
// verilog_lint: waive-start parameter-name-style
// verilog_lint: waive-start line-length
// verilog_lint: waive-start unpacked-dimensions-range-ordering

module dbbif_dram_model #(
    parameter ADDR_WIDTH = 32,            // 32 or 64 bits
    parameter DATA_WIDTH = 64,            // 32/64/128/256/512 bits
    parameter MEM_SIZE   = 1024           // Memory size in bytes
)(
    input  wire                      clk,
    input  wire                      rst_n,

    // AW Channel (Write Address)
    input  wire                      nvdla_core2dbb_aw_awvalid,
    output reg                       nvdla_core2dbb_aw_awready,
    input  wire [3:0]                nvdla_core2dbb_aw_awlen,
    input  wire [ADDR_WIDTH-1:0]     nvdla_core2dbb_aw_awaddr,
    input  wire [7:0]                nvdla_core2dbb_aw_awid,

    // W Channel (Write Data)
    input  wire                      nvdla_core2dbb_w_wvalid,
    output reg                       nvdla_core2dbb_w_wready,
    input  wire [DATA_WIDTH-1:0]     nvdla_core2dbb_w_wdata,
    input  wire                      nvdla_core2dbb_w_wlast,
    input  wire [DATA_WIDTH/8-1:0]   nvdla_core2dbb_w_wstrb,

    // B Channel (Write Response)
    output reg                       nvdla_core2dbb_b_bvalid,
    input  wire                      nvdla_core2dbb_b_bready,
    output reg  [7:0]                nvdla_core2dbb_b_bid,

    // AR Channel (Read Address)
    input  wire                      nvdla_core2dbb_ar_arvalid,
    output reg                       nvdla_core2dbb_ar_arready,
    input  wire [3:0]                nvdla_core2dbb_ar_arlen,
    input  wire [ADDR_WIDTH-1:0]     nvdla_core2dbb_ar_araddr,
    input  wire [7:0]                nvdla_core2dbb_ar_arid,

    // R Channel (Read Data)
    output reg                       nvdla_core2dbb_r_rvalid,
    input  wire                      nvdla_core2dbb_r_rready,
    output reg                       nvdla_core2dbb_r_rlast,
    output reg  [DATA_WIDTH-1:0]     nvdla_core2dbb_r_rdata,
    output reg  [7:0]                nvdla_core2dbb_r_rid
);

    //========================================================================
    // Memory Array - Simulated DRAM
    //========================================================================
    localparam BLOCK_SIZE = 4096;
    reg [7:0] memory [0:MEM_SIZE-1];

    // Initialize memory to zero
    integer i;
    initial begin
        for (i = 0; i < MEM_SIZE; i = i + 1) begin
            memory[i] = 8'h0;
        end
    end

    //========================================================================
    // Write Address Channel - Queue write address requests
    //========================================================================
    reg [ADDR_WIDTH-1:0] aw_addr_queue [0:15];
    reg [3:0] aw_len_queue [0:15];
    reg [7:0] aw_id_queue [0:15];
    reg [3:0] aw_wr_ptr, aw_rd_ptr;
    wire aw_queue_empty = (aw_wr_ptr == aw_rd_ptr);
    wire aw_queue_full = ((aw_wr_ptr + 1'b1) == aw_rd_ptr);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            nvdla_core2dbb_aw_awready <= 1'b1;
            aw_wr_ptr <= 4'b0;
            aw_rd_ptr <= 4'b0;
        end else begin
            if (nvdla_core2dbb_aw_awvalid && nvdla_core2dbb_aw_awready) begin
                // Push write address to queue
                aw_addr_queue[aw_wr_ptr] <= nvdla_core2dbb_aw_awaddr & ~(ADDR_WIDTH'(DATA_WIDTH/8 - 1));
                aw_len_queue[aw_wr_ptr] <= nvdla_core2dbb_aw_awlen;
                aw_id_queue[aw_wr_ptr] <= nvdla_core2dbb_aw_awid;
                aw_wr_ptr <= aw_wr_ptr + 1'b1;
                nvdla_core2dbb_aw_awready <= 1'b0;
            end else begin
                nvdla_core2dbb_aw_awready <= 1'b1;
            end
        end
    end

    //========================================================================
    // Write Data Channel - Queue write data
    //========================================================================
    reg [DATA_WIDTH-1:0] w_data_queue [0:15];
    reg [DATA_WIDTH/8-1:0] w_strb_queue [0:15];
    reg w_last_queue [0:15];
    reg [3:0] w_wr_ptr, w_rd_ptr;
    wire w_queue_empty = (w_wr_ptr == w_rd_ptr);
    wire w_queue_full = ((w_wr_ptr + 1'b1) == w_rd_ptr);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            nvdla_core2dbb_w_wready <= 1'b1;
            w_wr_ptr <= 4'b0;
            w_rd_ptr <= 4'b0;
        end else begin
            // Push write data to queue whenever wvalid
            if (nvdla_core2dbb_w_wvalid) begin
                w_data_queue[w_wr_ptr] <= nvdla_core2dbb_w_wdata;
                w_strb_queue[w_wr_ptr] <= nvdla_core2dbb_w_wstrb;
                w_last_queue[w_wr_ptr] <= nvdla_core2dbb_w_wlast;
                w_wr_ptr <= w_wr_ptr + 1'b1;
            end
        end
    end

    //========================================================================
    // Write Transaction Processing
    //========================================================================
    reg [ADDR_WIDTH-1:0] aw_current_addr;
    reg [3:0] aw_current_len;
    reg [7:0] aw_current_id;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            aw_current_addr <= {ADDR_WIDTH{1'b0}};
            aw_current_len <= 4'b0;
            aw_current_id <= 8'b0;
        end else begin
            if (!aw_queue_empty && !w_queue_empty) begin
                // Write data to memory
                for (i = 0; i < (DATA_WIDTH / 8); i = i + 1) begin
                    if (w_strb_queue[w_rd_ptr][i]) begin
                        if ((aw_addr_queue[aw_rd_ptr] + i) < MEM_SIZE) begin
                            memory[aw_addr_queue[aw_rd_ptr] + i] <= w_data_queue[w_rd_ptr][i*8 +: 8];
                        end
                    end
                end

                // Pop write data
                w_rd_ptr <= w_rd_ptr + 1'b1;

                if (w_last_queue[w_rd_ptr]) begin
                    // Last beat - pop address
                    aw_rd_ptr <= aw_rd_ptr + 1'b1;
                end else begin
                    // Continue burst - increment address
                    aw_addr_queue[aw_rd_ptr] <= aw_addr_queue[aw_rd_ptr] + (DATA_WIDTH / 8);
                    aw_len_queue[aw_rd_ptr] <= aw_len_queue[aw_rd_ptr] - 1'b1;
                end
            end
        end
    end

    //========================================================================
    // Write Response Channel - Send write ack
    //========================================================================
    reg [7:0] b_id_queue [0:15];
    reg [3:0] b_wr_ptr, b_rd_ptr;
    wire b_queue_empty = (b_wr_ptr == b_rd_ptr);
    wire b_queue_full = ((b_wr_ptr + 1'b1) == b_rd_ptr);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            nvdla_core2dbb_b_bvalid <= 1'b0;
            nvdla_core2dbb_b_bid <= 8'b0;
            b_wr_ptr <= 4'b0;
            b_rd_ptr <= 4'b0;
        end else begin
            // Push to b_fifo when write transaction completes
            if (!aw_queue_empty && !w_queue_empty && w_last_queue[w_rd_ptr] && !b_queue_full) begin
                b_id_queue[b_wr_ptr] <= aw_id_queue[aw_rd_ptr];
                b_wr_ptr <= b_wr_ptr + 1'b1;
            end

            // B channel logic: always clear first, then check if we can set
            nvdla_core2dbb_b_bvalid <= 1'b0;

            if (nvdla_core2dbb_b_bready && !b_queue_empty) begin
                nvdla_core2dbb_b_bvalid <= 1'b1;
                nvdla_core2dbb_b_bid <= b_id_queue[b_rd_ptr];
                b_rd_ptr <= b_rd_ptr + 1'b1;
            end
        end
    end

    //========================================================================
    // Read Address Channel - Queue read requests
    //========================================================================
    reg [ADDR_WIDTH-1:0] ar_addr_queue [0:15];
    reg [3:0] ar_len_queue [0:15];
    reg [7:0] ar_id_queue [0:15];
    reg [3:0] ar_wr_ptr, ar_rd_ptr;
    wire ar_queue_empty = (ar_wr_ptr == ar_rd_ptr);
    wire ar_queue_full = ((ar_wr_ptr + 1'b1) == ar_rd_ptr);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            nvdla_core2dbb_ar_arready <= 1'b1;
            ar_wr_ptr <= 4'b0;
            ar_rd_ptr <= 4'b0;
        end else begin
            if (nvdla_core2dbb_ar_arvalid && nvdla_core2dbb_ar_arready && !ar_queue_full) begin
                ar_addr_queue[ar_wr_ptr] <= nvdla_core2dbb_ar_araddr &
                                           ~(ADDR_WIDTH'(DATA_WIDTH/8 - 1));
                ar_len_queue[ar_wr_ptr] <= nvdla_core2dbb_ar_arlen;
                ar_id_queue[ar_wr_ptr] <= nvdla_core2dbb_ar_arid;
                ar_wr_ptr <= ar_wr_ptr + 1'b1;
                nvdla_core2dbb_ar_arready <= !((ar_wr_ptr + 2'b10) == ar_rd_ptr);
            end else begin
                nvdla_core2dbb_ar_arready <= !ar_queue_full;
            end
        end
    end

    //========================================================================
    // Read Data Channel - Send read data bursts
    //========================================================================
    reg [3:0] r_beat_cnt;
    reg [ADDR_WIDTH-1:0] r_addr;
    reg [3:0] r_len;
    reg [7:0] r_id;
    reg r_active;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            nvdla_core2dbb_r_rvalid <= 1'b0;
            nvdla_core2dbb_r_rlast <= 1'b0;
            nvdla_core2dbb_r_rdata <= {DATA_WIDTH{1'b0}};
            nvdla_core2dbb_r_rid <= 8'b0;
            r_beat_cnt <= 4'b0;
            r_addr <= {ADDR_WIDTH{1'b0}};
            r_len <= 4'b0;
            r_id <= 8'b0;
            r_active <= 1'b0;
        end else begin
            if (nvdla_core2dbb_r_rvalid && nvdla_core2dbb_r_rready) begin
                if (nvdla_core2dbb_r_rlast) begin
                    // Burst complete
                    nvdla_core2dbb_r_rvalid <= 1'b0;
                    r_active <= 1'b0;
                    ar_rd_ptr <= ar_rd_ptr + 1'b1;
                end else begin
                    // Continue burst
                    r_addr <= r_addr + (DATA_WIDTH / 8);
                    r_beat_cnt <= r_beat_cnt + 1'b1;

                    // Prepare next data
                    for (i = 0; i < (DATA_WIDTH / 8); i = i + 1) begin
                        if ((r_addr + (DATA_WIDTH / 8) + i) < MEM_SIZE) begin
                            nvdla_core2dbb_r_rdata[i*8 +: 8] <=
                                memory[r_addr + (DATA_WIDTH / 8) + i];
                        end else begin
                            nvdla_core2dbb_r_rdata[i*8 +: 8] <= 8'h0;
                        end
                    end

                    nvdla_core2dbb_r_rlast <= (r_beat_cnt + 1'b1 >= r_len);
                end
            end else if (!r_active && !ar_queue_empty) begin
                // Start new read burst
                r_active <= 1'b1;
                r_addr <= ar_addr_queue[ar_rd_ptr];
                r_len <= ar_len_queue[ar_rd_ptr] + 1'b1;
                r_id <= ar_id_queue[ar_rd_ptr];
                r_beat_cnt <= 4'b0;
                nvdla_core2dbb_r_rvalid <= 1'b1;
                nvdla_core2dbb_r_rid <= ar_id_queue[ar_rd_ptr];
                nvdla_core2dbb_r_rlast <= (ar_len_queue[ar_rd_ptr] == 4'b0);  // First beat is last if len=0

                // Read first data
                for (i = 0; i < (DATA_WIDTH / 8); i = i + 1) begin
                    if ((ar_addr_queue[ar_rd_ptr] + i) < MEM_SIZE) begin
                        nvdla_core2dbb_r_rdata[i*8 +: 8] <=
                            memory[ar_addr_queue[ar_rd_ptr] + i];
                    end else begin
                        nvdla_core2dbb_r_rdata[i*8 +: 8] <= 8'h0;
                    end
                end
            end
        end
    end

endmodule
