// verilog_lint: waive-start explicit-parameter-storage-type

module NVDLA_top #(parameter ADDR_WIDTH = 32,
             parameter DATA_WIDTH = 64,
             parameter MEM_SIZE = 1024
) (
    input wire dla_core_clk,
    input wire dla_csb_clk,
    input wire global_clk_ovr_on,
    input wire tmc2slcg_disable_clock_gating,
    input wire dla_reset_rstn,
    input wire direct_reset_,
    input wire test_mode,
    //csb
    input wire csb2nvdla_valid,
    output wire csb2nvdla_ready,
    input wire [15:0] csb2nvdla_addr,
    input wire [31:0] csb2nvdla_wdat,
    input wire csb2nvdla_write,
    input wire csb2nvdla_nposted,
    output wire nvdla2csb_valid,
    output wire [31:0] nvdla2csb_data,
    output wire nvdla2csb_wr_complete,
    ///////////////
    output wire dla_intr,
    input wire [31:0] nvdla_pwrbus_ram_c_pd,
    input wire [31:0] nvdla_pwrbus_ram_ma_pd,
    input wire [31:0] nvdla_pwrbus_ram_mb_pd,
    input wire [31:0] nvdla_pwrbus_ram_p_pd,
    input wire [31:0] nvdla_pwrbus_ram_o_pd,
    input wire [31:0] nvdla_pwrbus_ram_a_pd
);

// internal signals
wire nvdla_core2dbb_aw_awvalid;
wire nvdla_core2dbb_aw_awready;
wire [7:0] nvdla_core2dbb_aw_awid;
wire [3:0] nvdla_core2dbb_aw_awlen;
wire [32-1:0] nvdla_core2dbb_aw_awaddr;
wire nvdla_core2dbb_w_wvalid;
wire nvdla_core2dbb_w_wready;
wire [64-1:0] nvdla_core2dbb_w_wdata;
wire [64/8-1:0] nvdla_core2dbb_w_wstrb;
wire nvdla_core2dbb_w_wlast;
wire nvdla_core2dbb_ar_arvalid;
wire nvdla_core2dbb_ar_arready;
wire [7:0] nvdla_core2dbb_ar_arid;
wire [3:0] nvdla_core2dbb_ar_arlen;
wire [32-1:0] nvdla_core2dbb_ar_araddr;
wire nvdla_core2dbb_b_bvalid;
wire nvdla_core2dbb_b_bready;
wire [7:0] nvdla_core2dbb_b_bid;
wire nvdla_core2dbb_r_rvalid;
wire nvdla_core2dbb_r_rready;
wire [7:0] nvdla_core2dbb_r_rid;
wire nvdla_core2dbb_r_rlast;
wire [64-1:0] nvdla_core2dbb_r_rdata;


NV_nvdla u_dla (
     .dla_core_clk(dla_core_clk),
     .dla_csb_clk(dla_csb_clk),
     .global_clk_ovr_on(global_clk_ovr_on),
     .tmc2slcg_disable_clock_gating(tmc2slcg_disable_clock_gating),
     .dla_reset_rstn(dla_reset_rstn),
     .direct_reset_(direct_reset_),
     .test_mode(test_mode),

     // Minimal CSB
     .csb2nvdla_valid(csb2nvdla_valid),
     .csb2nvdla_ready(csb2nvdla_ready),
     .csb2nvdla_addr(csb2nvdla_addr),
     .csb2nvdla_wdat(csb2nvdla_wdat),
     .csb2nvdla_write(csb2nvdla_write),
     .csb2nvdla_nposted(csb2nvdla_nposted),
     .nvdla2csb_valid(nvdla2csb_valid),
     .nvdla2csb_data(nvdla2csb_data),
     .nvdla2csb_wr_complete(nvdla2csb_wr_complete),

     // Interrupt
     .dla_intr(dla_intr),

     //DBBIF
     .nvdla_core2dbb_aw_awlen      (nvdla_core2dbb_aw_awlen),
     .nvdla_core2dbb_aw_awid       (nvdla_core2dbb_aw_awid),
     .nvdla_core2dbb_aw_awaddr     (nvdla_core2dbb_aw_awaddr),
     .nvdla_core2dbb_aw_awready    (nvdla_core2dbb_aw_awready),
     .nvdla_core2dbb_aw_awvalid    (nvdla_core2dbb_aw_awvalid),

     .nvdla_core2dbb_ar_araddr     (nvdla_core2dbb_ar_araddr),
     .nvdla_core2dbb_ar_arready    (nvdla_core2dbb_ar_arready),
     .nvdla_core2dbb_ar_arid       (nvdla_core2dbb_ar_arid),
     .nvdla_core2dbb_ar_arvalid    (nvdla_core2dbb_ar_arvalid),
     .nvdla_core2dbb_ar_arlen      (nvdla_core2dbb_ar_arlen),

     .nvdla_core2dbb_w_wdata       (nvdla_core2dbb_w_wdata),
     .nvdla_core2dbb_w_wlast       (nvdla_core2dbb_w_wlast),
     .nvdla_core2dbb_w_wstrb       (nvdla_core2dbb_w_wstrb),
     .nvdla_core2dbb_w_wready      (nvdla_core2dbb_w_wready),
     .nvdla_core2dbb_w_wvalid      (nvdla_core2dbb_w_wvalid),

     .nvdla_core2dbb_r_rid         (nvdla_core2dbb_r_rid),
     .nvdla_core2dbb_r_rdata       (nvdla_core2dbb_r_rdata),
     .nvdla_core2dbb_r_rlast       (nvdla_core2dbb_r_rlast),
     .nvdla_core2dbb_r_rready      (nvdla_core2dbb_r_rready),
     .nvdla_core2dbb_r_rvalid      (nvdla_core2dbb_r_rvalid),

     .nvdla_core2dbb_b_bid         (nvdla_core2dbb_b_bid),
     .nvdla_core2dbb_b_bready      (nvdla_core2dbb_b_bready),
     .nvdla_core2dbb_b_bvalid      (nvdla_core2dbb_b_bvalid),

     // Powerbus
     .nvdla_pwrbus_ram_c_pd(nvdla_pwrbus_ram_c_pd),
     .nvdla_pwrbus_ram_ma_pd(nvdla_pwrbus_ram_ma_pd),
     .nvdla_pwrbus_ram_mb_pd(nvdla_pwrbus_ram_mb_pd),
     .nvdla_pwrbus_ram_p_pd(nvdla_pwrbus_ram_p_pd),
     .nvdla_pwrbus_ram_o_pd(nvdla_pwrbus_ram_o_pd),
     .nvdla_pwrbus_ram_a_pd(nvdla_pwrbus_ram_a_pd)
  );


dbbif_dram_model #(.ADDR_WIDTH(ADDR_WIDTH),.DATA_WIDTH(DATA_WIDTH),.MEM_SIZE(MEM_SIZE)) dram_dut (
      .nvdla_core2dbb_b_bid     (nvdla_core2dbb_b_bid),
      .nvdla_core2dbb_r_rid     (nvdla_core2dbb_r_rid),
      .nvdla_core2dbb_ar_arid   (nvdla_core2dbb_ar_arid),
      .nvdla_core2dbb_aw_awid   (nvdla_core2dbb_aw_awid),
      .nvdla_core2dbb_r_rdata   (nvdla_core2dbb_r_rdata),
      .nvdla_core2dbb_r_rlast   (nvdla_core2dbb_r_rlast),
      .nvdla_core2dbb_w_wdata   (nvdla_core2dbb_w_wdata),
      .nvdla_core2dbb_w_wlast   (nvdla_core2dbb_w_wlast),
      .nvdla_core2dbb_w_wstrb   (nvdla_core2dbb_w_wstrb),
      .nvdla_core2dbb_ar_arlen  (nvdla_core2dbb_ar_arlen),
      .nvdla_core2dbb_aw_awlen  (nvdla_core2dbb_aw_awlen),
      .nvdla_core2dbb_b_bready  (nvdla_core2dbb_b_bready),
      .nvdla_core2dbb_b_bvalid  (nvdla_core2dbb_b_bvalid),
      .nvdla_core2dbb_r_rready  (nvdla_core2dbb_r_rready),
      .nvdla_core2dbb_r_rvalid  (nvdla_core2dbb_r_rvalid),
      .nvdla_core2dbb_w_wready  (nvdla_core2dbb_w_wready),
      .nvdla_core2dbb_w_wvalid  (nvdla_core2dbb_w_wvalid),
      .nvdla_core2dbb_ar_araddr (nvdla_core2dbb_ar_araddr),
      .nvdla_core2dbb_aw_awaddr (nvdla_core2dbb_aw_awaddr),
      .nvdla_core2dbb_ar_arready(nvdla_core2dbb_ar_arready),
      .nvdla_core2dbb_ar_arvalid(nvdla_core2dbb_ar_arvalid),
      .nvdla_core2dbb_aw_awready(nvdla_core2dbb_aw_awready),
      .nvdla_core2dbb_aw_awvalid(nvdla_core2dbb_aw_awvalid),

      .clk                      (dla_core_clk),
      .rst_n                    (dla_reset_rstn)
  );



endmodule
