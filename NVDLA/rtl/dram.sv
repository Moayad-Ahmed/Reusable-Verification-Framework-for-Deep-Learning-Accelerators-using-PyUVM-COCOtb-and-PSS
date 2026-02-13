// verilog_lint: waive-start module-filename
// verilog_lint: waive-start explicit-parameter-storage-type
// verilog_lint: waive-start parameter-name-style
// verilog_lint: waive-start line-length
// verilog_lint: waive-start unpacked-dimensions-range-ordering

module dbbif_dual_port_dram #(
    parameter ADDR_WIDTH = 32,            // 32 or 64 bits
    parameter DATA_WIDTH = 64,            // 32/64/128/256/512 bits
    parameter MEM_SIZE   = 1024           // Memory size in bytes
)(
    input  wire                       clk,
    input  wire                       rst_n,

    //========================================================================
    // PRIMARY INTERFACE: nvdla_core2dbb
    //========================================================================
    input  wire                       nvdla_core2dbb_aw_awvalid,
    output reg                        nvdla_core2dbb_aw_awready,
    input  wire [3:0]                 nvdla_core2dbb_aw_awlen,
    input  wire [ADDR_WIDTH-1:0]      nvdla_core2dbb_aw_awaddr,
    input  wire [7:0]                 nvdla_core2dbb_aw_awid,

    input  wire                       nvdla_core2dbb_w_wvalid,
    output reg                        nvdla_core2dbb_w_wready,
    input  wire [DATA_WIDTH-1:0]      nvdla_core2dbb_w_wdata,
    input  wire                       nvdla_core2dbb_w_wlast,
    input  wire [DATA_WIDTH/8-1:0]    nvdla_core2dbb_w_wstrb,

    output reg                        nvdla_core2dbb_b_bvalid,
    input  wire                       nvdla_core2dbb_b_bready,
    output reg  [7:0]                 nvdla_core2dbb_b_bid,

    input  wire                       nvdla_core2dbb_ar_arvalid,
    output reg                        nvdla_core2dbb_ar_arready,
    input  wire [3:0]                 nvdla_core2dbb_ar_arlen,
    input  wire [ADDR_WIDTH-1:0]      nvdla_core2dbb_ar_araddr,
    input  wire [7:0]                 nvdla_core2dbb_ar_arid,

    output reg                        nvdla_core2dbb_r_rvalid,
    input  wire                       nvdla_core2dbb_r_rready,
    output reg                        nvdla_core2dbb_r_rlast,
    output reg  [DATA_WIDTH-1:0]      nvdla_core2dbb_r_rdata,
    output reg  [7:0]                 nvdla_core2dbb_r_rid,

    //========================================================================
    // SECONDARY INTERFACE: ext2dbb
    //========================================================================
    input  wire                       ext2dbb_awvalid,
    output reg                        ext2dbb_awready,
    input  wire [7:0]                 ext2dbb_awlen,
    input  wire [2:0]                 ext2dbb_awsize,
    input  wire [1:0]                 ext2dbb_awburst,
    input  wire [ADDR_WIDTH-1:0]      ext2dbb_awaddr,
    input  wire [7:0]                 ext2dbb_awid,

    input  wire                       ext2dbb_wvalid,
    output reg                        ext2dbb_wready,
    input  wire [DATA_WIDTH-1:0]      ext2dbb_wdata,
    input  wire                       ext2dbb_wlast,
    input  wire [DATA_WIDTH/8-1:0]    ext2dbb_wstrb,

    output reg                        ext2dbb_bvalid,
    input  wire                       ext2dbb_bready,
    output reg  [7:0]                 ext2dbb_bid,

    input  wire                       ext2dbb_arvalid,
    output reg                        ext2dbb_arready,
    input  wire [7:0]                 ext2dbb_arlen,
    input  wire [2:0]                 ext2dbb_arsize,
    input  wire [1:0]                 ext2dbb_arburst,
    input  wire [ADDR_WIDTH-1:0]      ext2dbb_araddr,
    input  wire [7:0]                 ext2dbb_arid,

    output reg                        ext2dbb_rvalid,
    input  wire                       ext2dbb_rready,
    output reg                        ext2dbb_rlast,
    output reg  [DATA_WIDTH-1:0]      ext2dbb_rdata,
    output reg  [7:0]                 ext2dbb_rid
);

    //========================================================================
    // Shared Memory Array
    //========================================================================
    reg [7:0] memory [0:MEM_SIZE-1];
    integer i;

    initial begin
        for (i = 0; i < MEM_SIZE; i = i + 1) begin
            memory[i] = 8'h0;
        end
    end

    //========================================================================
    // NVDLA PORT LOGIC
    //========================================================================
    reg [ADDR_WIDTH-1:0] aw_addr_q_a [0:15];
    reg [3:0] aw_len_q_a [0:15];
    reg [7:0] aw_id_q_a [0:15];
    reg [3:0] aw_wr_ptr_a, aw_rd_ptr_a;
    wire aw_q_empty_a = (aw_wr_ptr_a == aw_rd_ptr_a);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            nvdla_core2dbb_aw_awready <= 1'b1;
            aw_wr_ptr_a <= 4'b0; aw_rd_ptr_a <= 4'b0;
        end else begin
            if (nvdla_core2dbb_aw_awvalid && nvdla_core2dbb_aw_awready) begin
                aw_addr_q_a[aw_wr_ptr_a] <= nvdla_core2dbb_aw_awaddr & ~(ADDR_WIDTH'(DATA_WIDTH/8 - 1));
                aw_len_q_a[aw_wr_ptr_a]  <= nvdla_core2dbb_aw_awlen;
                aw_id_q_a[aw_wr_ptr_a]   <= nvdla_core2dbb_aw_awid;
                aw_wr_ptr_a <= aw_wr_ptr_a + 1'b1;
                nvdla_core2dbb_aw_awready <= 1'b0;
            end else nvdla_core2dbb_aw_awready <= 1'b1;
        end
    end

    reg [DATA_WIDTH-1:0] w_data_q_a [0:15];
    reg [DATA_WIDTH/8-1:0] w_strb_q_a [0:15];
    reg w_last_q_a [0:15];
    reg [3:0] w_wr_ptr_a, w_rd_ptr_a;
    wire w_q_empty_a = (w_wr_ptr_a == w_rd_ptr_a);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            nvdla_core2dbb_w_wready <= 1'b1;
            w_wr_ptr_a <= 4'b0; w_rd_ptr_a <= 4'b0;
        end else if (nvdla_core2dbb_w_wvalid) begin
            w_data_q_a[w_wr_ptr_a] <= nvdla_core2dbb_w_wdata;
            w_strb_q_a[w_wr_ptr_a] <= nvdla_core2dbb_w_wstrb;
            w_last_q_a[w_wr_ptr_a] <= nvdla_core2dbb_w_wlast;
            w_wr_ptr_a <= w_wr_ptr_a + 1'b1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (rst_n && !aw_q_empty_a && !w_q_empty_a) begin
            for (int j = 0; j < (DATA_WIDTH / 8); j = j + 1) begin
                if (w_strb_q_a[w_rd_ptr_a][j] && (aw_addr_q_a[aw_rd_ptr_a] + j) < MEM_SIZE)
                    memory[aw_addr_q_a[aw_rd_ptr_a] + j] <= w_data_q_a[w_rd_ptr_a][j*8 +: 8];
            end
            w_rd_ptr_a <= w_rd_ptr_a + 1'b1;
            if (w_last_q_a[w_rd_ptr_a]) aw_rd_ptr_a <= aw_rd_ptr_a + 1'b1;
            else aw_addr_q_a[aw_rd_ptr_a] <= aw_addr_q_a[aw_rd_ptr_a] + (DATA_WIDTH / 8);
        end
    end

    reg [7:0] b_id_q_a [0:15];
    reg [3:0] b_wr_ptr_a, b_rd_ptr_a;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin nvdla_core2dbb_b_bvalid <= 0; b_wr_ptr_a <= 0; b_rd_ptr_a <= 0; end
        else begin
            if (!aw_q_empty_a && !w_q_empty_a && w_last_q_a[w_rd_ptr_a]) begin
                b_id_q_a[b_wr_ptr_a] <= aw_id_q_a[aw_rd_ptr_a]; b_wr_ptr_a <= b_wr_ptr_a + 1'b1;
            end
            nvdla_core2dbb_b_bvalid <= 0;
            if (nvdla_core2dbb_b_bready && (b_wr_ptr_a != b_rd_ptr_a)) begin
                nvdla_core2dbb_b_bvalid <= 1; nvdla_core2dbb_b_bid <= b_id_q_a[b_rd_ptr_a]; b_rd_ptr_a <= b_rd_ptr_a + 1'b1;
            end
        end
    end

    reg [ADDR_WIDTH-1:0] ar_addr_q_a [0:15]; reg [3:0] ar_len_q_a [0:15]; reg [7:0] ar_id_q_a [0:15];
    reg [3:0] ar_wr_ptr_a, ar_rd_ptr_a;
    reg [3:0] r_beat_cnt_a; reg [ADDR_WIDTH-1:0] r_addr_a; reg [3:0] r_len_a; reg r_active_a;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin nvdla_core2dbb_ar_arready <= 1; ar_wr_ptr_a <= 0; ar_rd_ptr_a <= 0; end
        else if (nvdla_core2dbb_ar_arvalid && nvdla_core2dbb_ar_arready) begin
            ar_addr_q_a[ar_wr_ptr_a] <= nvdla_core2dbb_ar_araddr & ~(ADDR_WIDTH'(DATA_WIDTH/8 - 1));
            ar_len_q_a[ar_wr_ptr_a] <= nvdla_core2dbb_ar_arlen; ar_id_q_a[ar_wr_ptr_a] <= nvdla_core2dbb_ar_arid;
            ar_wr_ptr_a <= ar_wr_ptr_a + 1'b1; nvdla_core2dbb_ar_arready <= 0;
        end else nvdla_core2dbb_ar_arready <= 1;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin nvdla_core2dbb_r_rvalid <= 0; r_active_a <= 0; end
        else begin
            if (nvdla_core2dbb_r_rvalid && nvdla_core2dbb_r_rready) begin
                if (nvdla_core2dbb_r_rlast) begin nvdla_core2dbb_r_rvalid <= 0; r_active_a <= 0; ar_rd_ptr_a <= ar_rd_ptr_a + 1; end
                else begin
                    r_addr_a <= r_addr_a + (DATA_WIDTH/8); r_beat_cnt_a <= r_beat_cnt_a + 1;
                    for (int k=0; k<(DATA_WIDTH/8); k++) nvdla_core2dbb_r_rdata[k*8+:8] <= memory[r_addr_a+(DATA_WIDTH/8)+k];
                    nvdla_core2dbb_r_rlast <= (r_beat_cnt_a + 1 >= r_len_a);
                end
            end else if (!r_active_a && (ar_wr_ptr_a != ar_rd_ptr_a)) begin
                r_active_a <= 1; r_addr_a <= ar_addr_q_a[ar_rd_ptr_a]; r_len_a <= ar_len_q_a[ar_rd_ptr_a] + 1;
                r_beat_cnt_a <= 0; nvdla_core2dbb_r_rvalid <= 1; nvdla_core2dbb_r_rid <= ar_id_q_a[ar_rd_ptr_a];
                nvdla_core2dbb_r_rlast <= (ar_len_q_a[ar_rd_ptr_a] == 0);
                for (int m=0; m<(DATA_WIDTH/8); m++) nvdla_core2dbb_r_rdata[m*8+:8] <= memory[ar_addr_q_a[ar_rd_ptr_a]+m];
            end
        end
    end

    //========================================================================
    // EXTERNAL PORT LOGIC (Mirrored with ext2dbb names)
    //========================================================================
    reg [ADDR_WIDTH-1:0] aw_addr_q_b [0:15];
    reg [7:0] aw_len_q_b [0:15];
    reg [7:0] aw_id_q_b [0:15];
    reg [3:0] aw_wr_ptr_b, aw_rd_ptr_b;
    wire aw_q_empty_b = (aw_wr_ptr_b == aw_rd_ptr_b);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ext2dbb_awready <= 1'b1;
            aw_wr_ptr_b <= 4'b0; aw_rd_ptr_b <= 4'b0;
        end else begin
            if (ext2dbb_awvalid && ext2dbb_awready) begin
                aw_addr_q_b[aw_wr_ptr_b] <= ext2dbb_awaddr & ~(ADDR_WIDTH'(DATA_WIDTH/8 - 1));
                aw_len_q_b[aw_wr_ptr_b]  <= ext2dbb_awlen;
                aw_id_q_b[aw_wr_ptr_b]   <= ext2dbb_awid;
                aw_wr_ptr_b <= aw_wr_ptr_b + 1'b1;
                ext2dbb_awready <= 1'b0;
            end else ext2dbb_awready <= 1'b1;
        end
    end

    reg [DATA_WIDTH-1:0] w_data_q_b [0:15];
    reg [DATA_WIDTH/8-1:0] w_strb_q_b [0:15];
    reg w_last_q_b [0:15];
    reg [3:0] w_wr_ptr_b, w_rd_ptr_b;
    wire w_q_empty_b = (w_wr_ptr_b == w_rd_ptr_b);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ext2dbb_wready <= 1'b1;
            w_wr_ptr_b <= 4'b0; w_rd_ptr_b <= 4'b0;
        end else if (ext2dbb_wvalid) begin
            w_data_q_b[w_wr_ptr_b] <= ext2dbb_wdata;
            w_strb_q_b[w_wr_ptr_b] <= ext2dbb_wstrb;
            w_last_q_b[w_wr_ptr_b] <= ext2dbb_wlast;
            w_wr_ptr_b <= w_wr_ptr_b + 1'b1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (rst_n && !aw_q_empty_b && !w_q_empty_b) begin
            for (int n = 0; n < (DATA_WIDTH / 8); n = n + 1) begin
                if (w_strb_q_b[w_rd_ptr_b][n] && (aw_addr_q_b[aw_rd_ptr_b] + n) < MEM_SIZE)
                    memory[aw_addr_q_b[aw_rd_ptr_b] + n] <= w_data_q_b[w_rd_ptr_b][n*8 +: 8];
            end
            w_rd_ptr_b <= w_rd_ptr_b + 1'b1;
            if (w_last_q_b[w_rd_ptr_b]) aw_rd_ptr_b <= aw_rd_ptr_b + 1'b1;
            else aw_addr_q_b[aw_rd_ptr_b] <= aw_addr_q_b[aw_rd_ptr_b] + (DATA_WIDTH / 8);
        end
    end

    reg [7:0] b_id_q_b [0:15];
    reg [3:0] b_wr_ptr_b, b_rd_ptr_b;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin ext2dbb_bvalid <= 0; b_wr_ptr_b <= 0; b_rd_ptr_b <= 0; end
        else begin
            if (!aw_q_empty_b && !w_q_empty_b && w_last_q_b[w_rd_ptr_b]) begin
                b_id_q_b[b_wr_ptr_b] <= aw_id_q_b[aw_rd_ptr_b]; b_wr_ptr_b <= b_wr_ptr_b + 1'b1;
            end
            ext2dbb_bvalid <= 0;
            if (ext2dbb_bready && (b_wr_ptr_b != b_rd_ptr_b)) begin
                ext2dbb_bvalid <= 1; ext2dbb_bid <= b_id_q_b[b_rd_ptr_b]; b_rd_ptr_b <= b_rd_ptr_b + 1'b1;
            end
        end
    end

    reg [ADDR_WIDTH-1:0] ar_addr_q_b [0:15]; reg [7:0] ar_len_q_b [0:15]; reg [7:0] ar_id_q_b [0:15];
    reg [3:0] ar_wr_ptr_b, ar_rd_ptr_b;
    reg [7:0] r_beat_cnt_b; reg [ADDR_WIDTH-1:0] r_addr_b; reg [7:0] r_len_b; reg r_active_b;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin ext2dbb_arready <= 1; ar_wr_ptr_b <= 0; ar_rd_ptr_b <= 0; end
        else if (ext2dbb_arvalid && ext2dbb_arready) begin
            ar_addr_q_b[ar_wr_ptr_b] <= ext2dbb_araddr & ~(ADDR_WIDTH'(DATA_WIDTH/8 - 1));
            ar_len_q_b[ar_wr_ptr_b] <= ext2dbb_arlen; ar_id_q_b[ar_wr_ptr_b] <= ext2dbb_arid;
            ar_wr_ptr_b <= ar_wr_ptr_b + 1'b1; ext2dbb_arready <= 0;
        end else ext2dbb_arready <= 1;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin ext2dbb_rvalid <= 0; r_active_b <= 0; end
        else begin
            if (ext2dbb_rvalid && ext2dbb_rready) begin
                if (ext2dbb_rlast) begin ext2dbb_rvalid <= 0; r_active_b <= 0; ar_rd_ptr_b <= ar_rd_ptr_b + 1; end
                else begin
                    r_addr_b <= r_addr_b + (DATA_WIDTH/8); r_beat_cnt_b <= r_beat_cnt_b + 1;
                    for (int p=0; p<(DATA_WIDTH/8); p++) ext2dbb_rdata[p*8+:8] <= memory[r_addr_b+(DATA_WIDTH/8)+p];
                    ext2dbb_rlast <= (r_beat_cnt_b + 1 >= r_len_b);
                end
            end else if (!r_active_b && (ar_wr_ptr_b != ar_rd_ptr_b)) begin
                r_active_b <= 1; r_addr_b <= ar_addr_q_b[ar_rd_ptr_b]; r_len_b <= ar_len_q_b[ar_rd_ptr_b] + 1;
                r_beat_cnt_b <= 0; ext2dbb_rvalid <= 1; ext2dbb_rid <= ar_id_q_b[ar_rd_ptr_b];
                ext2dbb_rlast <= (ar_len_q_b[ar_rd_ptr_b] == 0);
                for (int q=0; q<(DATA_WIDTH/8); q++) ext2dbb_rdata[q*8+:8] <= memory[ar_addr_q_b[ar_rd_ptr_b]+q];
            end
        end
    end

endmodule
