
`timescale 1ns / 1ps

module systolic_array_8x8 #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input logic clk,
    input logic reset,
    input logic start,

    input logic signed [DATA_WIDTH-1:0] matrix_a [0:7][0:7],
    input logic [DATA_WIDTH-1:0] matrix_b [0:7][0:7],

    output logic signed [ACC_WIDTH-1:0] matrix_c [0:7][0:7],
    output logic done,
    output logic busy
);

    // FSM states
    localparam IDLE  = 2'b00;
    localparam CLEAR = 2'b01;
    localparam RUN   = 2'b10;
    localparam DONE  = 2'b11;

    logic [1:0] state;

    // 8x8 array connections
    logic signed [DATA_WIDTH-1:0] a_bus [0:7][0:8];
    logic [DATA_WIDTH-1:0] b_bus [0:8][0:7];

    logic clear_acc;

    // Counts the 22 cycles needed for an 8x8 systolic operation
    logic [4:0] cycle_count;

    integer r, c;

    // Connect matrix inputs to the systolic array.
    // Each row of A is delayed by its row index.
    // Each column of B is delayed by its column index.
    always_comb begin
        for (r = 0; r < 8; r = r + 1) begin
            a_bus[r][0] = '0;

            if ((cycle_count >= r) && ((cycle_count - r) < 8))
                a_bus[r][0] = matrix_a[r][cycle_count - r];
        end

        for (c = 0; c < 8; c = c + 1) begin
            b_bus[0][c] = '0;

            if ((cycle_count >= c) && ((cycle_count - c) < 8))
                b_bus[0][c] = matrix_b[cycle_count - c][c];
        end
    end

    // Instantiate the 64 processing elements.
    genvar row, col;
    generate
        for (row = 0; row < 8; row = row + 1) begin : ROW
            for (col = 0; col < 8; col = col + 1) begin : COL

                systolic_pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH)
                ) pe (
                    .clk(clk),
                    .reset(reset),
                    .enable(state == RUN),
                    .clear_acc(clear_acc),

                    .a_in(a_bus[row][col]),
                    .b_in(b_bus[row][col]),

                    .a_out(a_bus[row][col+1]),
                    .b_out(b_bus[row+1][col]),

                    .acc_out(matrix_c[row][col])
                );

            end
        end
    endgenerate

    // Control FSM
    always_ff @(posedge clk) begin
        if (reset) begin
            state       <= IDLE;
            cycle_count <= 0;
            done        <= 0;
            busy        <= 0;
            clear_acc   <= 0;
        end else begin
            done      <= 0;
            clear_acc <= 0;

            case (state)

                IDLE: begin
                    busy        <= 0;
                    cycle_count <= 0;

                    if (start) begin
                        state     <= CLEAR;
                        busy      <= 1;
                        clear_acc <= 1;
                    end
                end

                CLEAR: begin
                    // PEs receive clear_acc for this clock edge.
                    // Multiplication remains disabled.
                    clear_acc   <= 0;
                    cycle_count <= 0;
                    state       <= RUN;
                    busy        <= 1;
                end

                RUN: begin
                    busy <= 1;

                    if (cycle_count == 21) begin
                        state <= DONE;
                    end else begin
                        cycle_count <= cycle_count + 1;
                    end
                end

                DONE: begin
                    done  <= 1;
                    busy  <= 0;
                    state <= IDLE;
                end

                default: begin
                    state       <= IDLE;
                    cycle_count <= 0;
                    busy        <= 0;
                    done        <= 0;
                    clear_acc   <= 0;
                end

            endcase
        end
    end

endmodule
