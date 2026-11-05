
`timescale 1ns / 1ps

module tpu_top #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32,
    parameter ADDR_WIDTH = 32,

    // --------------------------------------------------------
    // MEMORY FILE LOCATIONS
    // --------------------------------------------------------

    parameter HIDDEN_WEIGHTS_FILE =
        "mem_files/dense_layer_test/hidden_weights.mem",

    parameter HIDDEN_BIASES_FILE =
        "mem_files/dense_layer_test/hidden_biases.mem",

    parameter HIDDEN_MULTIPLIERS_FILE =
        "mem_files/dense_layer_test/hidden_multipliers.mem",

    parameter HIDDEN_SHIFTS_FILE =
        "mem_files/dense_layer_test/hidden_shifts.mem",

    parameter INPUT_PIXELS_FILE =
        "mem_files/dense_layer_test/input_pixels.mem",

    parameter OUTPUT_WEIGHTS_FILE =
        "mem_files/output_layer_test/output_weights.mem",

    parameter OUTPUT_BIASES_FILE =
        "mem_files/output_layer_test/output_biases.mem",

    parameter OUTPUT_MULTIPLIERS_FILE =
        "mem_files/output_layer_test/output_multipliers.mem",

    parameter OUTPUT_SHIFTS_FILE =
        "mem_files/output_layer_test/output_shifts.mem"

)(
    input  logic clk,
    input  logic reset,
    input  logic start,

    // Final predicted class: 0 through 31
    output logic [4:0] predicted_class,

    // High while the TPU is processing an image
    output logic busy,

    // High for one cycle when inference is complete
    output logic done
);

    // ========================================================
    // NETWORK CONFIGURATION
    // ========================================================

    localparam INPUT_FEATURES  = 784;
    localparam HIDDEN_NEURONS  = 128;
    localparam OUTPUT_CLASSES = 32;

    // ========================================================
    // TOP-LEVEL FSM
    // ========================================================

    localparam IDLE         = 3'd0;
    localparam START_HIDDEN = 3'd1;
    localparam WAIT_HIDDEN  = 3'd2;
    localparam START_OUTPUT = 3'd3;
    localparam WAIT_OUTPUT  = 3'd4;
    localparam FIND_MAX     = 3'd5;
    localparam FINISHED     = 3'd6;

    logic [2:0] state;

    // Start signals sent to the two dense-layer controllers.
    logic hidden_start;
    logic output_start;

    // Completion signals returned by the controllers.
    logic hidden_done;
    logic output_done;

    // ========================================================
    // SHARED SYSTOLIC ARRAY INTERFACE
    // ========================================================

    // Each layer has its own tile request signal.
    logic hidden_tile_start;
    logic output_tile_start;

    // The shared tile engine has one start/done interface.
    logic shared_tile_start;
    logic shared_tile_done;
    logic shared_tile_busy;

    // Tile data produced by the hidden layer controller.
    logic signed [DATA_WIDTH-1:0]
        hidden_weight_tile [0:7][0:7];

    logic [DATA_WIDTH-1:0]
        hidden_activation_tile [0:7];

    // Tile data produced by the output layer controller.
    logic signed [DATA_WIDTH-1:0]
        output_weight_tile [0:7][0:7];

    logic [DATA_WIDTH-1:0]
        output_activation_tile [0:7];

    // Selected tile data sent into the shared engine.
    logic signed [DATA_WIDTH-1:0]
        shared_weight_tile [0:7][0:7];

    logic [DATA_WIDTH-1:0]
        shared_activation_tile [0:7];

    // Partial sums returned by the shared engine.
    logic signed [ACC_WIDTH+2:0]
        shared_partial_sums [0:7];

    // ========================================================
    // INTERNAL MEMORY ARRAYS
    // ========================================================

    // Input image: 784 adjusted unsigned pixel values.
    logic [7:0] input_memory [0:783];

    // Hidden layer: 128 * 784 = 100,352 weights.
    logic signed [7:0] hidden_weight_memory [0:100351];

    logic signed [31:0] hidden_bias_memory [0:127];
    logic signed [31:0] hidden_multiplier_memory [0:127];
    logic signed [7:0] hidden_shift_memory [0:127];

    // Output layer: 32 * 128 = 4,096 weights.
    logic signed [7:0] output_weight_memory [0:4095];

    logic signed [31:0] output_bias_memory [0:31];
    logic signed [31:0] output_multiplier_memory [0:31];
    logic signed [7:0] output_shift_memory [0:31];

    // ========================================================
    // INTERMEDIATE ACTIVATION BUFFER
    // ========================================================

    // Stores the 128 outputs produced by the hidden layer.
    // Hidden output is signed INT8 with zero point -128.
    // The output layer expects adjusted unsigned activations:
    //
    // adjusted_activation = hidden_result + 128

    logic [7:0] hidden_activation_buffer [0:127];

    // ========================================================
    // FINAL OUTPUT SCORE MEMORY
    // ========================================================

    logic signed [7:0] output_scores [0:31];

    // ========================================================
    // HIDDEN LAYER MEMORY INTERFACE
    // ========================================================

    logic hidden_weight_read_enable;
    logic [ADDR_WIDTH-1:0] hidden_weight_address;
    logic signed [7:0] hidden_weight_data;

    logic hidden_activation_read_enable;
    logic [ADDR_WIDTH-1:0] hidden_activation_address;
    logic [7:0] hidden_activation_data;

    logic hidden_bias_read_enable;
    logic [ADDR_WIDTH-1:0] hidden_bias_address;
    logic signed [31:0] hidden_bias_data;

    logic hidden_multiplier_read_enable;
    logic [ADDR_WIDTH-1:0] hidden_multiplier_address;
    logic signed [31:0] hidden_multiplier_data;

    logic hidden_shift_read_enable;
    logic [ADDR_WIDTH-1:0] hidden_shift_address;
    logic signed [7:0] hidden_shift_data;

    logic hidden_result_valid;
    logic hidden_result_ready;

    logic hidden_result_write_enable;
    logic [ADDR_WIDTH-1:0] hidden_result_address;
    logic signed [7:0] hidden_result_data;

    logic hidden_busy;

    // ========================================================
    // OUTPUT LAYER MEMORY INTERFACE
    // ========================================================

    logic output_weight_read_enable;
    logic [ADDR_WIDTH-1:0] output_weight_address;
    logic signed [7:0] output_weight_data;

    logic output_activation_read_enable;
    logic [ADDR_WIDTH-1:0] output_activation_address;
    logic [7:0] output_activation_data;

    logic output_bias_read_enable;
    logic [ADDR_WIDTH-1:0] output_bias_address;
    logic signed [31:0] output_bias_data;

    logic output_multiplier_read_enable;
    logic [ADDR_WIDTH-1:0] output_multiplier_address;
    logic signed [31:0] output_multiplier_data;

    logic output_shift_read_enable;
    logic [ADDR_WIDTH-1:0] output_shift_address;
    logic signed [7:0] output_shift_data;

    logic output_result_valid;
    logic output_result_ready;

    logic output_result_write_enable;
    logic [ADDR_WIDTH-1:0] output_result_address;
    logic signed [7:0] output_result_data;

    logic output_busy;

    // ========================================================
    // ARGMAX SIGNALS
    // ========================================================

    integer argmax_index;
    integer argmax_score;
    integer argmax_loop;

    // ========================================================
    // MEMORY INITIALIZATION
    // ========================================================

    initial begin

        $display("========================================");
        $display("LOADING TPU MODEL MEMORIES");
        $display("========================================");

        $readmemh(HIDDEN_WEIGHTS_FILE, hidden_weight_memory);
        $readmemh(HIDDEN_BIASES_FILE, hidden_bias_memory);
        $readmemh(HIDDEN_MULTIPLIERS_FILE, hidden_multiplier_memory);
        $readmemh(HIDDEN_SHIFTS_FILE, hidden_shift_memory);

        $readmemh(INPUT_PIXELS_FILE, input_memory);

        $readmemh(OUTPUT_WEIGHTS_FILE, output_weight_memory);
        $readmemh(OUTPUT_BIASES_FILE, output_bias_memory);
        $readmemh(OUTPUT_MULTIPLIERS_FILE, output_multiplier_memory);
        $readmemh(OUTPUT_SHIFTS_FILE, output_shift_memory);

        $display("All TPU model memories loaded.");
        $display("");

    end

    // ========================================================
    // CONTROLLER START SIGNALS
    // ========================================================

    assign hidden_start = (state == START_HIDDEN);
    assign output_start = (state == START_OUTPUT);

    assign hidden_result_ready = 1'b1;
    assign output_result_ready = 1'b1;

    // ========================================================
    // SHARED TILE ENGINE CONTROL
    // ========================================================

    // Only one dense layer is active at a time.
    // Therefore, their tile start requests can be combined.
    assign shared_tile_start =
        hidden_tile_start | output_tile_start;

    // Select the active layer's tile data.
    //
    // The top-level FSM stays in WAIT_HIDDEN while the hidden
    // layer uses the engine, and WAIT_OUTPUT while the output
    // layer uses it. The selected inputs therefore remain
    // stable throughout each tile operation.

    integer mux_row;
    integer mux_col;

    always_comb begin

        // Default to hidden-layer tile data.
        shared_weight_tile = hidden_weight_tile;
        shared_activation_tile = hidden_activation_tile;

        // Select output-layer tile data during output inference.
        if (state == START_OUTPUT || state == WAIT_OUTPUT) begin

            shared_weight_tile = output_weight_tile;
            shared_activation_tile = output_activation_tile;

        end

    end

    // ========================================================
    // ONE SHARED DENSE TILE CONTROLLER
    // ========================================================

    // Both dense layers use this same tile controller and
    // therefore the same physical 8x8 systolic array.

    dense_tile_controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) shared_tile_controller (

        .clk(clk),
        .reset(reset),
        .start(shared_tile_start),

        .weight_tile(shared_weight_tile),
        .activation_tile(shared_activation_tile),

        .partial_sums(shared_partial_sums),

        .done(shared_tile_done),
        .busy(shared_tile_busy)

    );

    // ========================================================
    // HIDDEN LAYER CONTROLLER
    // ========================================================

    dense_layer_controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) hidden_layer (

        .clk(clk),
        .reset(reset),
        .start(hidden_start),

        .input_features(32'd784),
        .output_neurons(32'd128),

        .weight_read_enable(hidden_weight_read_enable),
        .weight_address(hidden_weight_address),
        .weight_data(hidden_weight_data),

        .activation_read_enable(hidden_activation_read_enable),
        .activation_address(hidden_activation_address),
        .activation_data(hidden_activation_data),

        .bias_read_enable(hidden_bias_read_enable),
        .bias_address(hidden_bias_address),
        .bias_data(hidden_bias_data),

        .multiplier_read_enable(hidden_multiplier_read_enable),
        .multiplier_address(hidden_multiplier_address),
        .multiplier_data(hidden_multiplier_data),

        .shift_read_enable(hidden_shift_read_enable),
        .shift_address(hidden_shift_address),
        .shift_data(hidden_shift_data),

        .output_zero_point(-8'sd128),
        .relu_enable(1'b1),

        .result_valid(hidden_result_valid),
        .result_ready(hidden_result_ready),

        .result_write_enable(hidden_result_write_enable),
        .result_address(hidden_result_address),
        .result_data(hidden_result_data),

        .busy(hidden_busy),
        .done(hidden_done),

        // Shared tile engine interface.
        .tile_start(hidden_tile_start),
        .shared_weight_tile(hidden_weight_tile),
        .shared_activation_tile(hidden_activation_tile),
        .shared_partial_sums(shared_partial_sums),
        .shared_tile_done(shared_tile_done)

    );

    // ========================================================
    // OUTPUT LAYER CONTROLLER
    // ========================================================

    dense_layer_controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) output_layer (

        .clk(clk),
        .reset(reset),
        .start(output_start),

        .input_features(32'd128),
        .output_neurons(32'd32),

        .weight_read_enable(output_weight_read_enable),
        .weight_address(output_weight_address),
        .weight_data(output_weight_data),

        .activation_read_enable(output_activation_read_enable),
        .activation_address(output_activation_address),
        .activation_data(output_activation_data),

        .bias_read_enable(output_bias_read_enable),
        .bias_address(output_bias_address),
        .bias_data(output_bias_data),

        .multiplier_read_enable(output_multiplier_read_enable),
        .multiplier_address(output_multiplier_address),
        .multiplier_data(output_multiplier_data),

        .shift_read_enable(output_shift_read_enable),
        .shift_address(output_shift_address),
        .shift_data(output_shift_data),

        .output_zero_point(8'sd39),
        .relu_enable(1'b0),

        .result_valid(output_result_valid),
        .result_ready(output_result_ready),

        .result_write_enable(output_result_write_enable),
        .result_address(output_result_address),
        .result_data(output_result_data),

        .busy(output_busy),
        .done(output_done),

        // Shared tile engine interface.
        .tile_start(output_tile_start),
        .shared_weight_tile(output_weight_tile),
        .shared_activation_tile(output_activation_tile),
        .shared_partial_sums(shared_partial_sums),
        .shared_tile_done(shared_tile_done)

    );

    // ========================================================
    // SYNCHRONOUS MEMORY READS AND RESULT WRITES
    // ========================================================

    always_ff @(posedge clk) begin

        // ----------------------------------------------------
        // HIDDEN LAYER MEMORY READS
        // ----------------------------------------------------

        if (hidden_weight_read_enable)
            hidden_weight_data <=
                hidden_weight_memory[hidden_weight_address];

        if (hidden_activation_read_enable)
            hidden_activation_data <=
                input_memory[hidden_activation_address];

        if (hidden_bias_read_enable)
            hidden_bias_data <=
                hidden_bias_memory[hidden_bias_address];

        if (hidden_multiplier_read_enable)
            hidden_multiplier_data <=
                hidden_multiplier_memory[hidden_multiplier_address];

        if (hidden_shift_read_enable)
            hidden_shift_data <=
                hidden_shift_memory[hidden_shift_address];

        // ----------------------------------------------------
        // HIDDEN LAYER RESULT WRITES
        // ----------------------------------------------------

        if (hidden_result_write_enable &&
            hidden_result_address < HIDDEN_NEURONS) begin

            hidden_activation_buffer[hidden_result_address] <=
                $unsigned(hidden_result_data) + 8'd128;

        end

        // ----------------------------------------------------
        // OUTPUT LAYER MEMORY READS
        // ----------------------------------------------------

        if (output_weight_read_enable)
            output_weight_data <=
                output_weight_memory[output_weight_address];

        if (output_activation_read_enable)
            output_activation_data <=
                hidden_activation_buffer[output_activation_address];

        if (output_bias_read_enable)
            output_bias_data <=
                output_bias_memory[output_bias_address];

        if (output_multiplier_read_enable)
            output_multiplier_data <=
                output_multiplier_memory[output_multiplier_address];

        if (output_shift_read_enable)
            output_shift_data <=
                output_shift_memory[output_shift_address];

        // ----------------------------------------------------
        // FINAL OUTPUT WRITES
        // ----------------------------------------------------

        if (output_result_write_enable &&
            output_result_address < OUTPUT_CLASSES) begin

            output_scores[output_result_address] <=
                output_result_data;

        end

    end

    // ========================================================
    // ARGMAX CALCULATION
    // ========================================================

    always_comb begin

        argmax_index = 0;
        argmax_score = -129;

        for (argmax_loop = 0;
             argmax_loop < OUTPUT_CLASSES;
             argmax_loop = argmax_loop + 1) begin

            if ($signed(output_scores[argmax_loop]) > argmax_score) begin

                argmax_score = $signed(output_scores[argmax_loop]);
                argmax_index = argmax_loop;

            end

        end

    end

    // ========================================================
    // TOP-LEVEL CONTROL FSM
    // ========================================================

    always_ff @(posedge clk) begin

        if (reset) begin

            state <= IDLE;
            predicted_class <= 5'd0;

        end

        else begin

            case (state)

                IDLE: begin

                    if (start)
                        state <= START_HIDDEN;

                end

                START_HIDDEN: begin

                    state <= WAIT_HIDDEN;

                end

                WAIT_HIDDEN: begin

                    if (hidden_done)
                        state <= START_OUTPUT;

                end

                START_OUTPUT: begin

                    state <= WAIT_OUTPUT;

                end

                WAIT_OUTPUT: begin

                    if (output_done)
                        state <= FIND_MAX;

                end

                FIND_MAX: begin

                    predicted_class <= argmax_index[4:0];
                    state <= FINISHED;

                end

                FINISHED: begin

                    state <= IDLE;

                end

                default: begin

                    state <= IDLE;

                end

            endcase

        end

    end

    // ========================================================
    // TOP-LEVEL STATUS SIGNALS
    // ========================================================

    assign done = (state == FINISHED);

    assign busy = (state != IDLE) && (state != FINISHED);

endmodule

