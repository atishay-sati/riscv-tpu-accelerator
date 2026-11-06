
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

    input logic shared_tile_done
);

    // --------------------------------------------------------
    // FSM states
    // --------------------------------------------------------

    // Five bits are required because the new
    // ADD_BIAS_HIGH state uses state value 18.
    localparam IDLE                = 5'd0;
    localparam LOAD_ACT_ADDR       = 5'd1;
    localparam LOAD_ACT_WAIT       = 5'd2;
    localparam LOAD_ACT_SAVE       = 5'd3;
    localparam LOAD_W_ADDR         = 5'd4;
    localparam LOAD_W_WAIT         = 5'd5;
    localparam LOAD_W_SAVE         = 5'd6;
    localparam LOAD_PARAM_ADDR     = 5'd7;
    localparam LOAD_PARAM_WAIT     = 5'd8;
    localparam LOAD_PARAM_SAVE     = 5'd9;
    localparam START_TILE          = 5'd10;
    localparam WAIT_TILE           = 5'd11;
    localparam ACCUMULATE          = 5'd12;
    localparam REQUANTIZE          = 5'd13;
    localparam WRITE_RESULTS       = 5'd14;
    localparam FINISHED            = 5'd15;
    localparam WAIT_REQUANTIZE     = 5'd16;
    localparam START_REQUANTIZER   = 5'd17;
    localparam ADD_BIAS_HIGH       = 5'd18;

    logic [4:0] state;

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

    logic signed [31:0] selected_multiplier;
    logic signed [7:0] selected_shift;
    logic signed [7:0] selected_requantized_result;

    // --------------------------------------------------------
    // Split accumulator-plus-bias addition pipeline
    // --------------------------------------------------------

    // The original 32-bit addition required eight CARRY4
    // blocks in one combinational path.
    //
    // We now perform the addition in two 16-bit stages.
    // This reduces the length of the carry chain in each stage.

    // Stores the lower 16-bit addition result.
    // Bit 16 stores the carry into the upper half.
    logic [16:0] bias_low_sum_reg;

    // Stores the upper 16 bits of the accumulator and bias
    // while the lower-half addition is being completed.
    logic [15:0] accumulator_high_reg;
    logic [15:0] bias_high_reg;

    // Combinational result of the upper 16-bit addition.
    // The saved carry from the lower addition is included.
    logic [16:0] selected_bias_high_sum;

    // Final 32-bit accumulator-plus-bias result.
    // This is registered before entering the requantizer.
    logic signed [31:0] biased_accumulator_reg;

    // Registered quantization parameters for the selected neuron.
    logic signed [31:0] multiplier_reg;
    logic signed [7:0] shift_reg;

    // Starts the shared requantizer.
    // The controller pulses this for one clock cycle.
    logic requantizer_start;

    // Pulses high when the requantizer has completed.
    logic requantizer_done;

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

        selected_multiplier =
            multiplier_tile[requantize_count];

        selected_shift =
            shift_tile[requantize_count];

    end

    // --------------------------------------------------------
    // Calculate the upper half of the bias addition
    // --------------------------------------------------------

    // The lower-half addition has already been completed and
    // registered in bias_low_sum_reg.
    //
    // Add the upper accumulator and bias halves, including
    // the carry generated by the lower-half addition.
    //
    // The 17-bit result preserves the carry out. The final
    // 32-bit result uses only its lower 16 bits, matching
    // normal 32-bit two's-complement wraparound behavior.

    always_comb begin

        selected_bias_high_sum =
            {1'b0, accumulator_high_reg} +
            {1'b0, bias_high_reg} +
            bias_low_sum_reg[16];

    end

    // --------------------------------------------------------
    // Shared sequential requantizer
    // --------------------------------------------------------

    // Only one requantizer is used for all eight neurons.
    // The controller starts it, waits for done, and then
    // stores its completed INT8 result.
    //
    // Its accumulator input comes from the completed
    // two-stage accumulator-plus-bias pipeline.
    requantize_int8 shared_requantizer (

        .clk(clk),
        .reset(reset),
        .start(requantizer_start),

        .accumulator(biased_accumulator_reg),
        .multiplier(multiplier_reg),
        .shift(shift_reg),

        .output_zero_point(output_zero_point),
        .relu(relu_enable),

        .result(selected_requantized_result),
        .done(requantizer_done)

    );

    // --------------------------------------------------------
    // Result storage for the eight neurons
    // --------------------------------------------------------

    logic signed [7:0] result_buffer [0:7];

    // --------------------------------------------------------
    // Result handshake outputs
    // --------------------------------------------------------

    assign result_valid =
        (state == WRITE_RESULTS) &&
        ((neuron_base + output_count) < neurons_reg);

    assign result_write_enable =
        result_valid && result_ready;

    assign result_address =
        neuron_base + output_count;

    assign result_data =
        result_buffer[output_count];

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

            // Reset the requantizer start pulse.
            requantizer_start <= 0;

            // Reset the requantization pipeline registers.
            biased_accumulator_reg <= 0;
            multiplier_reg <= 0;
            shift_reg <= 0;

            // Reset the split bias-addition pipeline.
            bias_low_sum_reg <= 0;
            accumulator_high_reg <= 0;
            bias_high_reg <= 0;

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

                for (j = 0; j < 8; j = j + 1) begin

                    weight_tile[i][j] <= 0;

                end

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

            // Default requantizer start to zero.
            // It is asserted only in the START_REQUANTIZER state.
            requantizer_start <= 0;

            case (state)

                // ============================================
                // IDLE
                // ============================================

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

                        for (i = 0; i < 8; i = i + 1) begin

                            accumulators[i] <= 0;

                        end

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

                    if (shared_tile_done) begin

                        state <= ACCUMULATE;

                    end

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

                        // Begin requantizing neuron zero.
                        requantize_count <= 0;
                        state <= REQUANTIZE;

                    end

                end

                // ============================================
                // FIRST STAGE OF BIAS ADDITION
                // ============================================

                REQUANTIZE: begin

                    // Add the lower 16 bits of the accumulator
                    // and bias. Save the 17-bit result so that
                    // the carry into bit 16 is preserved.
                    bias_low_sum_reg <=
                        {1'b0, accumulators[requantize_count][15:0]} +
                        {1'b0, bias_tile[requantize_count][15:0]};

                    // Save the upper halves for the next stage.
                    accumulator_high_reg <=
                        accumulators[requantize_count][31:16];

                    bias_high_reg <=
                        bias_tile[requantize_count][31:16];

                    // Capture this neuron's quantization parameters.
                    multiplier_reg <= selected_multiplier;
                    shift_reg <= selected_shift;

                    // Complete the upper-half addition next.
                    state <= ADD_BIAS_HIGH;

                end

                // ============================================
                // SECOND STAGE OF BIAS ADDITION
                // ============================================

                ADD_BIAS_HIGH: begin

                    // Combine the upper-half sum with the saved
                    // lower-half result to form the complete
                    // 32-bit accumulator-plus-bias value.
                    biased_accumulator_reg <= {
                        selected_bias_high_sum[15:0],
                        bias_low_sum_reg[15:0]
                    };

                    // Start the requantizer after the final
                    // 32-bit result has been registered.
                    state <= START_REQUANTIZER;

                end

                // ============================================
                // START REQUANTIZER
                // ============================================

                START_REQUANTIZER: begin

                    // Pulse the requantizer start signal.
                    // It captures the registered parameters
                    // on the following rising clock edge.
                    requantizer_start <= 1;

                    // Wait for the requantizer to finish.
                    state <= WAIT_REQUANTIZE;

                end

                // ============================================
                // WAIT FOR REQUANTIZATION
                // ============================================

                WAIT_REQUANTIZE: begin

                    // The requantizer pulses done after its
                    // result register has been updated.
                    if (requantizer_done) begin

                        // Save the completed INT8 result for
                        // the currently selected neuron.
                        result_buffer[requantize_count] <=
                            selected_requantized_result;

                        if (requantize_count == 7) begin

                            // All eight neurons are complete.
                            requantize_count <= 0;
                            output_count <= 0;

                            state <= WRITE_RESULTS;

                        end else begin

                            // Select the next neuron.
                            requantize_count <= requantize_count + 1;

                            // Capture the next neuron's parameters.
                            state <= REQUANTIZE;

                        end

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

                // ============================================
                // DEFAULT
                // ============================================

                default: begin

                    state <= IDLE;
                    busy <= 0;

                end

            endcase

        end

    end

endmodule
