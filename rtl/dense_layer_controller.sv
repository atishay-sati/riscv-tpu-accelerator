`timescale 1ns / 1ps

module dense_layer_controller #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32,
    parameter ADDR_WIDTH = 32
)(
    input logic clk,
    input logic reset,
    input logic start,

    // Layer configuration
    input logic [ADDR_WIDTH-1:0] input_features,
    input logic [ADDR_WIDTH-1:0] output_neurons,

    // Weight memory interface
    output logic weight_read_enable,
    output logic [ADDR_WIDTH-1:0] weight_address,
    input logic signed [DATA_WIDTH-1:0] weight_data,

    // Activation memory interface
    output logic activation_read_enable,
    output logic [ADDR_WIDTH-1:0] activation_address,
    input logic [DATA_WIDTH-1:0] activation_data,

    // Bias memory interface
    output logic bias_read_enable,
    output logic [ADDR_WIDTH-1:0] bias_address,
    input logic signed [ACC_WIDTH-1:0] bias_data,

    // Per-neuron quantization parameters
    output logic multiplier_read_enable,
    output logic [ADDR_WIDTH-1:0] multiplier_address,
    input logic signed [31:0] multiplier_data,

    output logic shift_read_enable,
    output logic [ADDR_WIDTH-1:0] shift_address,
    input logic signed [7:0] shift_data,

    // Shared output quantization settings
    input logic signed [7:0] output_zero_point,
    input logic relu_enable,

    // Result handshake
    output logic result_valid,
    input logic result_ready,

    // Result output interface
    output logic result_write_enable,
    output logic [ADDR_WIDTH-1:0] result_address,
    output logic signed [7:0] result_data,

    output logic busy,
    output logic done,

    // --------------------------------------------------------
    // Shared systolic-array interface
    // --------------------------------------------------------

    // Request the shared tile controller to process a tile.
    output logic tile_start,

    // Weight and activation data for the shared tile engine.
    output logic signed [DATA_WIDTH-1:0]
        shared_weight_tile [0:7][0:7],

    output logic [DATA_WIDTH-1:0]
        shared_activation_tile [0:7],

    // Partial sums returned by the shared tile engine.
    input logic signed [ACC_WIDTH+2:0]
        shared_partial_sums [0:7],

    // Completion signal from the shared tile engine.
    input logic shared_tile_done
);

    // --------------------------------------------------------
    // FSM states
    // --------------------------------------------------------

    localparam IDLE             = 4'd0;
    localparam LOAD_ACT_ADDR    = 4'd1;
    localparam LOAD_ACT_WAIT    = 4'd2;
    localparam LOAD_ACT_SAVE    = 4'd3;
    localparam LOAD_W_ADDR      = 4'd4;
    localparam LOAD_W_WAIT      = 4'd5;
    localparam LOAD_W_SAVE      = 4'd6;
    localparam LOAD_PARAM_ADDR  = 4'd7;
    localparam LOAD_PARAM_WAIT  = 4'd8;
    localparam LOAD_PARAM_SAVE  = 4'd9;
    localparam START_TILE       = 4'd10;
    localparam WAIT_TILE        = 4'd11;
    localparam ACCUMULATE       = 4'd12;
    localparam REQUANTIZE       = 4'd13;
    localparam WRITE_RESULTS    = 4'd14;
    localparam FINISHED         = 4'd15;

    logic [3:0] state;

    // --------------------------------------------------------
    // Configuration registers
    // --------------------------------------------------------

    logic [ADDR_WIDTH-1:0] features_reg;
    logic [ADDR_WIDTH-1:0] neurons_reg;

    logic [ADDR_WIDTH-1:0] feature_base;
    logic [ADDR_WIDTH-1:0] neuron_base;

    logic [ADDR_WIDTH-1:0] feature_tiles;
    logic [ADDR_WIDTH-1:0] neuron_groups;

    logic [ADDR_WIDTH-1:0] feature_tile_count;
    logic [ADDR_WIDTH-1:0] neuron_group_count;

    // --------------------------------------------------------
    // Counters
    // --------------------------------------------------------

    logic [6:0] load_count;
    logic [3:0] output_count;
    logic [2:0] requantize_count;

    // --------------------------------------------------------
    // Local tile storage
    // --------------------------------------------------------

    logic signed [DATA_WIDTH-1:0] weight_tile [0:7][0:7];
    logic [DATA_WIDTH-1:0] activation_tile [0:7];

    logic signed [ACC_WIDTH-1:0] bias_tile [0:7];

    logic signed [31:0] multiplier_tile [0:7];
    logic signed [7:0] shift_tile [0:7];

    // Accumulated dot products before adding bias.
    logic signed [ACC_WIDTH-1:0] accumulators [0:7];

    // --------------------------------------------------------
    // Shared requantization datapath
    // --------------------------------------------------------

    logic signed [31:0] selected_biased_accumulator;
    logic signed [31:0] selected_multiplier;
    logic signed [7:0] selected_shift;
    logic signed [7:0] selected_requantized_result;

    // --------------------------------------------------------
    // Forward this layer's tile data to the shared engine.
    //
    // The top-level mux decides which layer's tile reaches
    // the single shared dense_tile_controller.
    // --------------------------------------------------------

    integer tile_row;
    integer tile_col;

    always_comb begin
        for (tile_row = 0; tile_row < 8; tile_row = tile_row + 1) begin
            shared_activation_tile[tile_row] =
                activation_tile[tile_row];

            for (tile_col = 0; tile_col < 8; tile_col = tile_col + 1) begin
                shared_weight_tile[tile_row][tile_col] =
                    weight_tile[tile_row][tile_col];
            end
        end
    end

    // --------------------------------------------------------
    // Select the current neuron's requantization parameters
    // --------------------------------------------------------

    always_comb begin
        selected_biased_accumulator =
            accumulators[requantize_count] +
            bias_tile[requantize_count];

        selected_multiplier =
            multiplier_tile[requantize_count];

        selected_shift =
            shift_tile[requantize_count];
    end

    // One requantizer shared by all eight neurons.
    requantize_int8 shared_requantizer (
        .accumulator(selected_biased_accumulator),
        .multiplier(selected_multiplier),
        .shift(selected_shift),
        .output_zero_point(output_zero_point),
        .relu(relu_enable),
        .result(selected_requantized_result)
    );

    // --------------------------------------------------------
    // Result handshake outputs
    // --------------------------------------------------------

    assign result_valid =
        (state == WRITE_RESULTS) &&
        ((neuron_base + output_count) < neurons_reg);

    assign result_write_enable = result_valid && result_ready;

    assign result_address = neuron_base + output_count;
    assign result_data = result_buffer[output_count];

    // Result storage for the eight neurons.
    logic signed [7:0] result_buffer [0:7];

    // --------------------------------------------------------
    // Main controller
    // --------------------------------------------------------

    integer i;
    integer j;

    always_ff @(posedge clk) begin

        if (reset) begin

            state <= IDLE;

            busy <= 0;
            done <= 0;

            weight_read_enable <= 0;
            activation_read_enable <= 0;
            bias_read_enable <= 0;
            multiplier_read_enable <= 0;
            shift_read_enable <= 0;

            weight_address <= 0;
            activation_address <= 0;
            bias_address <= 0;
            multiplier_address <= 0;
            shift_address <= 0;

            tile_start <= 0;

            features_reg <= 0;
            neurons_reg <= 0;

            feature_base <= 0;
            neuron_base <= 0;

            feature_tiles <= 0;
            neuron_groups <= 0;

            feature_tile_count <= 0;
            neuron_group_count <= 0;

            load_count <= 0;
            output_count <= 0;
            requantize_count <= 0;

            for (i = 0; i < 8; i = i + 1) begin

                bias_tile[i] <= 0;
                multiplier_tile[i] <= 0;
                shift_tile[i] <= 0;

                accumulators[i] <= 0;
                result_buffer[i] <= 0;

                activation_tile[i] <= 0;

                for (j = 0; j < 8; j = j + 1)
                    weight_tile[i][j] <= 0;

            end

        end else begin

            // Default one-cycle control signals.
            done <= 0;
            tile_start <= 0;

            weight_read_enable <= 0;
            activation_read_enable <= 0;
            bias_read_enable <= 0;
            multiplier_read_enable <= 0;
            shift_read_enable <= 0;

            case (state)

                IDLE: begin

                    busy <= 0;

                    if (start && input_features != 0 &&
                        output_neurons != 0) begin

                        features_reg <= input_features;
                        neurons_reg <= output_neurons;

                        feature_tiles <= (input_features + 7) / 8;
                        neuron_groups <= (output_neurons + 7) / 8;

                        feature_base <= 0;
                        neuron_base <= 0;

                        feature_tile_count <= 0;
                        neuron_group_count <= 0;

                        load_count <= 0;
                        output_count <= 0;
                        requantize_count <= 0;

                        busy <= 1;
                        state <= LOAD_PARAM_ADDR;

                    end

                end

                // ============================================
                // LOAD BIASES, MULTIPLIERS, AND SHIFTS
                // ============================================

                LOAD_PARAM_ADDR: begin

                    bias_read_enable <= 1;
                    multiplier_read_enable <= 1;
                    shift_read_enable <= 1;

                    bias_address <= neuron_base + load_count;
                    multiplier_address <= neuron_base + load_count;
                    shift_address <= neuron_base + load_count;

                    state <= LOAD_PARAM_WAIT;

                end

                LOAD_PARAM_WAIT: begin
                    state <= LOAD_PARAM_SAVE;
                end

                LOAD_PARAM_SAVE: begin

                    if ((neuron_base + load_count) < neurons_reg) begin

                        bias_tile[load_count] <= bias_data;
                        multiplier_tile[load_count] <= multiplier_data;
                        shift_tile[load_count] <= shift_data;

                    end else begin

                        bias_tile[load_count] <= 0;
                        multiplier_tile[load_count] <= 0;
                        shift_tile[load_count] <= 0;

                    end

                    if (load_count == 7) begin

                        load_count <= 0;

                        for (i = 0; i < 8; i = i + 1)
                            accumulators[i] <= 0;

                        feature_base <= 0;
                        feature_tile_count <= 0;

                        state <= LOAD_ACT_ADDR;

                    end else begin

                        load_count <= load_count + 1;
                        state <= LOAD_PARAM_ADDR;

                    end

                end

                // ============================================
                // LOAD ACTIVATIONS
                // ============================================

                LOAD_ACT_ADDR: begin

                    activation_read_enable <= 1;
                    activation_address <= feature_base + load_count;

                    state <= LOAD_ACT_WAIT;

                end

                LOAD_ACT_WAIT: begin
                    state <= LOAD_ACT_SAVE;
                end

                LOAD_ACT_SAVE: begin

                    if ((feature_base + load_count) < features_reg)
                        activation_tile[load_count] <= activation_data;
                    else
                        activation_tile[load_count] <= 0;

                    if (load_count == 7) begin

                        load_count <= 0;
                        state <= LOAD_W_ADDR;

                    end else begin

                        load_count <= load_count + 1;
                        state <= LOAD_ACT_ADDR;

                    end

                end

                // ============================================
                // LOAD WEIGHTS
                // ============================================

                LOAD_W_ADDR: begin

                    weight_read_enable <= 1;

                    weight_address <=
                        (neuron_base + (load_count / 8))
                        * features_reg
                        + feature_base
                        + (load_count % 8);

                    state <= LOAD_W_WAIT;

                end

                LOAD_W_WAIT: begin
                    state <= LOAD_W_SAVE;
                end

                LOAD_W_SAVE: begin

                    if (((neuron_base + (load_count / 8)) < neurons_reg) &&
                        ((feature_base + (load_count % 8)) < features_reg)) begin

                        weight_tile[load_count / 8][load_count % 8]
                            <= weight_data;

                    end else begin

                        weight_tile[load_count / 8][load_count % 8]
                            <= 0;

                    end

                    if (load_count == 63) begin

                        load_count <= 0;
                        state <= START_TILE;

                    end else begin

                        load_count <= load_count + 1;
                        state <= LOAD_W_ADDR;

                    end

                end

                // ============================================
                // START SHARED TILE ENGINE
                // ============================================

                START_TILE: begin

                    tile_start <= 1;
                    state <= WAIT_TILE;

                end

                // ============================================
                // WAIT FOR SHARED TILE ENGINE
                // ============================================

                WAIT_TILE: begin

                    if (shared_tile_done)
                        state <= ACCUMULATE;

                end

                // ============================================
                // ACCUMULATE PARTIAL SUMS
                // ============================================

                ACCUMULATE: begin

                    for (i = 0; i < 8; i = i + 1) begin

                        accumulators[i] <=
                            accumulators[i] + shared_partial_sums[i];

                    end

                    if (feature_tile_count + 1 < feature_tiles) begin

                        feature_tile_count <= feature_tile_count + 1;
                        feature_base <= feature_base + 8;

                        load_count <= 0;
                        state <= LOAD_ACT_ADDR;

                    end else begin

                        requantize_count <= 0;
                        state <= REQUANTIZE;

                    end

                end

                // ============================================
                // REQUANTIZE
                // ============================================

                REQUANTIZE: begin

                    result_buffer[requantize_count] <=
                        selected_requantized_result;

                    if (requantize_count == 7) begin

                        requantize_count <= 0;
                        output_count <= 0;

                        state <= WRITE_RESULTS;

                    end else begin

                        requantize_count <= requantize_count + 1;

                    end

                end

                // ============================================
                // WRITE RESULTS
                // ============================================

                WRITE_RESULTS: begin

                    if (result_valid && result_ready) begin

                        if (output_count == 7) begin

                            if (neuron_group_count + 1 < neuron_groups) begin

                                neuron_group_count <= neuron_group_count + 1;
                                neuron_base <= neuron_base + 8;

                                feature_base <= 0;
                                feature_tile_count <= 0;

                                load_count <= 0;
                                state <= LOAD_PARAM_ADDR;

                            end else begin

                                state <= FINISHED;

                            end

                        end else begin

                            output_count <= output_count + 1;

                        end

                    end else if (!result_valid) begin

                        if (neuron_group_count + 1 < neuron_groups) begin

                            neuron_group_count <= neuron_group_count + 1;
                            neuron_base <= neuron_base + 8;

                            feature_base <= 0;
                            feature_tile_count <= 0;

                            load_count <= 0;
                            state <= LOAD_PARAM_ADDR;

                        end else begin

                            state <= FINISHED;

                        end

                    end

                end

                // ============================================
                // FINISHED
                // ============================================

                FINISHED: begin

                    busy <= 0;
                    done <= 1;

                    state <= IDLE;

                end

                default: begin

                    state <= IDLE;
                    busy <= 0;

                end

            endcase

        end

    end

endmodule