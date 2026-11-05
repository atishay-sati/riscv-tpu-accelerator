`timescale 1ns / 1ps

module dense_tile_controller #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input logic clk,
    input logic reset,
    input logic start,

    // Eight neurons, each with eight weights.
    // Weights are signed INT8.
    input logic signed [DATA_WIDTH-1:0] weight_tile [0:7][0:7],

    // Eight input activations.
    // Activations are unsigned INT8.
    input logic [DATA_WIDTH-1:0] activation_tile [0:7],

    // Eight partial sums, one for each neuron.
    // Extra bits allow summing eight array outputs.
    output logic signed [ACC_WIDTH+2:0] partial_sums [0:7],

    output logic done,
    output logic busy
);

    // --------------------------------------------------------
    // FSM states
    // --------------------------------------------------------

    localparam IDLE        = 3'b000;
    localparam START_ARRAY = 3'b001;
    localparam WAIT_ARRAY  = 3'b010;
    localparam REDUCE      = 3'b011;
    localparam FINISHED    = 3'b100;

    logic [2:0] state;

    // --------------------------------------------------------
    // Systolic array signals
    // --------------------------------------------------------

    logic array_start;
    logic array_done;
    logic array_busy;

    logic signed [DATA_WIDTH-1:0] matrix_a [0:7][0:7];
    logic [DATA_WIDTH-1:0] matrix_b [0:7][0:7];

    logic signed [ACC_WIDTH-1:0] matrix_c [0:7][0:7];

    // Temporary storage for row reduction.
    logic signed [ACC_WIDTH+2:0] sum_temp;

    integer r;
    integer c;

    // --------------------------------------------------------
    // Construct matrices for dense-layer multiplication.
    //
    // A = weight_tile
    //
    // B = diagonal matrix containing activations.
    //
    // This makes:
    // C[r][c] = weight_tile[r][c] * activation_tile[c]
    //
    // Summing each row produces one neuron's partial sum.
    // --------------------------------------------------------

    always_comb begin
    for (int row = 0; row < 8; row++) begin
        for (int col = 0; col < 8; col++) begin
            matrix_a[row][col] = weight_tile[row][col];

            if (row == col)
                matrix_b[row][col] = activation_tile[row];
            else
                matrix_b[row][col] = '0;
            end
        end
    end

    // --------------------------------------------------------
    // Instantiate the general-purpose systolic array.
    // --------------------------------------------------------

    systolic_array_8x8 #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) array (
        .clk(clk),
        .reset(reset),
        .start(array_start),

        .matrix_a(matrix_a),
        .matrix_b(matrix_b),
        .matrix_c(matrix_c),

        .done(array_done),
        .busy(array_busy)
    );

    // --------------------------------------------------------
    // Controller FSM
    // --------------------------------------------------------

    always_ff @(posedge clk) begin

        if (reset) begin

            state       <= IDLE;
            array_start <= 0;
            done        <= 0;
            busy        <= 0;

            for (r = 0; r < 8; r = r + 1)
                partial_sums[r] <= '0;

        end else begin

            // Default values.
            array_start <= 0;
            done        <= 0;

            case (state)

                // --------------------------------------------
                // IDLE: Wait for a tile request.
                // --------------------------------------------

                IDLE: begin

                    busy <= 0;

                    if (start) begin

                        busy  <= 1;
                        state <= START_ARRAY;

                    end

                end

                // --------------------------------------------
                // START_ARRAY: Send a one-cycle start pulse.
                // --------------------------------------------

                START_ARRAY: begin

                    array_start <= 1;
                    busy        <= 1;
                    state       <= WAIT_ARRAY;

                end

                // --------------------------------------------
                // WAIT_ARRAY: Wait for matrix multiplication.
                // --------------------------------------------

                WAIT_ARRAY: begin

                    busy <= 1;

                    if (array_done) begin
                        state <= REDUCE;
                    end

                end

                // --------------------------------------------
                // REDUCE: Sum each row of the result matrix.
                //
                // partial_sums[r] =
                //   C[r][0] + C[r][1] + ... + C[r][7]
                // --------------------------------------------

                REDUCE: begin

                    for (r = 0; r < 8; r = r + 1) begin

                        sum_temp = '0;

                        for (c = 0; c < 8; c = c + 1) begin

                            sum_temp = sum_temp + matrix_c[r][c];

                        end

                        partial_sums[r] <= sum_temp;

                    end

                    state <= FINISHED;

                end

                // --------------------------------------------
                // FINISHED: Signal completion.
                // --------------------------------------------

                FINISHED: begin

                    done  <= 1;
                    busy  <= 0;
                    state <= IDLE;

                end

                default: begin

                    state       <= IDLE;
                    array_start <= 0;
                    done        <= 0;
                    busy        <= 0;

                end

            endcase

        end

    end

endmodule
