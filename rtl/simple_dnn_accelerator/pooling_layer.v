module pooling_layer #(
    parameter IN_DATA_WIDTH = 72,           // Width of input data
    parameter OUT_DATA_WIDTH = 32,          // Width of output data
    parameter POOL_SIZE = 2,                // Pooling window size (2x2)
    parameter STRIDE_H = 1,                 // Vertical stride
    parameter STRIDE_W = 1,                 // Horizontal stride
    parameter IMG_HEIGHT = 3,               // Input image height
    parameter IMG_WIDTH = 3                 // Input image width
) (
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [IN_DATA_WIDTH-1:0] data_in,
    input wire [1:0] pool_mode,             // 0: MAX, 1: AVG, 2: MIN

    output reg valid_out,
    output reg [OUT_DATA_WIDTH-1:0] data_out

);

localparam elem_width = 8;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_out <= 0;
        data_out <= 0;
    end else begin
        if (valid_in) begin
            case (pool_mode)
                2'b00: begin
                    data_out <= max_pooling(data_in);
                end
                2'b01: begin
                    data_out <= avg_pooling(data_in);
                end
                2'b10: begin
                    data_out <= min_pooling(data_in);
                end
                default: begin
                    data_out <= 0;
                end
            endcase
            valid_out <= 1;
        end else begin
            valid_out <= 0;
            data_out <= 0;
        end
    end
end

function [OUT_DATA_WIDTH-1:0] max_pooling;

    input [IN_DATA_WIDTH-1:0] data_in_max;
    reg [elem_width-1:0] E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8, E_9;
    begin

       E_1 = data_in_max[elem_width-1            :0 ];
       E_2 = data_in_max[2*elem_width-1 : elem_width];
       E_3 = data_in_max[3*elem_width-1 : 2*elem_width];
       E_4 = data_in_max[4*elem_width-1 : 3*elem_width];
       E_5 = data_in_max[5*elem_width-1 : 4*elem_width];
       E_6 = data_in_max[6*elem_width-1 : 5*elem_width];
       E_7 = data_in_max[7*elem_width-1 : 6*elem_width];
       E_8 = data_in_max[8*elem_width-1 : 7*elem_width];
       E_9 = data_in_max[9*elem_width-1 : 8*elem_width];

        max_pooling[elem_width-1:0]              = max4(E_1, E_2, E_4, E_5);
        max_pooling[2*elem_width-1:elem_width]   = max4(E_2, E_3, E_5, E_6);
        max_pooling[3*elem_width-1:2*elem_width] = max4(E_4, E_5, E_7, E_8);
        max_pooling[4*elem_width-1:3*elem_width] = max4(E_5, E_6, E_8, E_9);
    end
endfunction

function [7:0] max4;
    input [7:0] a,b,c,d;
    reg [7:0] m1, m2;
    begin
    m1 = (a > b) ? a : b;
    m2 = (c > d) ? c : d;
    max4 = (m1 > m2) ? m1 : m2;
    end
endfunction

function [OUT_DATA_WIDTH-1:0] min_pooling;

    input [IN_DATA_WIDTH-1:0] data_in_min;
    reg [elem_width-1:0] E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8, E_9;
    begin

       E_1 = data_in_min[elem_width-1            :0 ];
       E_2 = data_in_min[2*elem_width-1 : elem_width];
       E_3 = data_in_min[3*elem_width-1 : 2*elem_width];
       E_4 = data_in_min[4*elem_width-1 : 3*elem_width];
       E_5 = data_in_min[5*elem_width-1 : 4*elem_width];
       E_6 = data_in_min[6*elem_width-1 : 5*elem_width];
       E_7 = data_in_min[7*elem_width-1 : 6*elem_width];
       E_8 = data_in_min[8*elem_width-1 : 7*elem_width];
       E_9 = data_in_min[9*elem_width-1 : 8*elem_width];

        min_pooling[elem_width-1:0]              = min4(E_1, E_2, E_4, E_5);
        min_pooling[2*elem_width-1:elem_width]   = min4(E_2, E_3, E_5, E_6);
        min_pooling[3*elem_width-1:2*elem_width] = min4(E_4, E_5, E_7, E_8);
        min_pooling[4*elem_width-1:3*elem_width] = min4(E_5, E_6, E_8, E_9);
    end

endfunction

function [7:0] min4;
    input [7:0] a,b,c,d;
    reg [7:0] m1, m2;
    begin
    m1 = (a < b) ? a : b;
    m2 = (c < d) ? c : d;
    min4 = (m1 < m2) ? m1 : m2;
    end
endfunction

function [OUT_DATA_WIDTH-1:0] avg_pooling;

    input [IN_DATA_WIDTH-1:0] data_in_avg;
    reg [elem_width-1:0] E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8, E_9;
    begin

       E_1 = data_in_avg[elem_width-1            :0 ];
       E_2 = data_in_avg[2*elem_width-1 : elem_width];
       E_3 = data_in_avg[3*elem_width-1 : 2*elem_width];
       E_4 = data_in_avg[4*elem_width-1 : 3*elem_width];
       E_5 = data_in_avg[5*elem_width-1 : 4*elem_width];
       E_6 = data_in_avg[6*elem_width-1 : 5*elem_width];
       E_7 = data_in_avg[7*elem_width-1 : 6*elem_width];
       E_8 = data_in_avg[8*elem_width-1 : 7*elem_width];
       E_9 = data_in_avg[9*elem_width-1 : 8*elem_width];

        avg_pooling[elem_width-1:0]              = avg4(E_1, E_2, E_4, E_5);
        avg_pooling[2*elem_width-1:elem_width]   = avg4(E_2, E_3, E_5, E_6);
        avg_pooling[3*elem_width-1:2*elem_width] = avg4(E_4, E_5, E_7, E_8);
        avg_pooling[4*elem_width-1:3*elem_width] = avg4(E_5, E_6, E_8, E_9);
    end

endfunction

function [7:0] avg4;
    input [7:0] a,b,c,d;
    begin
        avg4 = (a + b + c + d) / 4;
    end
endfunction

endmodule