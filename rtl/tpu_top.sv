
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

    // Set to 1 for the original file-based simulation.
    // Set to 0 when an external interface (such as UART)
    // supplies the 784 image pixels at runtime.
    parameter bit LOAD_INPUT_FROM_FILE = 1'b1,

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

    // --------------------------------------------------------
    // EXTERNAL INPUT IMAGE WRITE INTERFACE
    // --------------------------------------------------------

    // Assert for one clock cycle to write one pixel into
    // input_memory. The UART controller will drive these.
    input  logic        input_write_enable,

    // Address of the pixel being written: 0 through 783.
    input  logic [9:0]  input_write_address,

    // Unsigned 8-bit pixel value being written.
    input  logic [7:0]  input_write_data,

    // Final predicted class: 0 through 31.
    output logic [4:0] predicted_class,

    // High while the TPU is processing an image.
    output logic busy,

    // High for one cycle when inference is complete.
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

    logic hidden_tile_start;
    logic output_tile_start;

    logic shared_tile_start;
    logic shared_tile_done;
    logic shared_tile_busy;

    logic signed [DATA_WIDTH-1:0]
        hidden_weight_tile [0:7][0:7];

    logic [DATA_WIDTH-1:0]
        hidden_activation_tile [0:7];

    logic signed [DATA_WIDTH-1:0]
        output_weight_tile [0:7][0:7];

    logic [DATA_WIDTH-1:0]
        output_activation_tile [0:7];

    logic signed [DATA_WIDTH-1:0]
        shared_weight_tile [0:7][0:7];

    logic [DATA_WIDTH-1:0]
        shared_activation_tile [0:7];

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

    // The output layer writes 32 signed INT8 class scores here.
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
    // SEQUENTIAL ARGMAX SIGNALS
    // ========================================================

    // Selects which output score is currently being examined.
    // Five bits address classes 0 through 31.
    logic [4:0] argmax_counter;

    // Highest score found so far.
    // Nine bits allow initialization to -129, below the
    // minimum signed INT8 score of -128.
    logic signed [8:0] argmax_best_score;

    // Class index corresponding to the best score.
    logic [4:0] argmax_best_index;

    // Current output score, sign-extended from 8 to 9 bits.
    logic signed [8:0] argmax_current_score;

    // Select one score using the current counter value.
    assign argmax_current_score = {
        output_scores[argmax_counter][7],
        output_scores[argmax_counter]
    };

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

        // Preserve the original input-file workflow for the
        // existing TPU verification testbench. For UART-based
        // deployment, set LOAD_INPUT_FROM_FILE to 0 instead.
        if (LOAD_INPUT_FROM_FILE) begin
            $readmemh(INPUT_PIXELS_FILE, input_memory);
            $display("Input image loaded from: %s", INPUT_PIXELS_FILE);
        end
        else begin
            $display("Input image will be supplied externally.");
        end

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
    // The top-level FSM stays in WAIT_HIDDEN while the hidden
    // layer uses the engine, and WAIT_OUTPUT while the output
    // layer uses it. The selected inputs remain stable during
    // each tile operation.

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
        // EXTERNAL INPUT IMAGE WRITES
        // ----------------------------------------------------

        // The UART controller writes one pixel per received
        // byte. Only accept writes while the TPU is idle, so
        // the input image cannot change during inference.
        if (input_write_enable &&
            (input_write_address < INPUT_FEATURES) &&
            (state == IDLE)) begin


             input_memory[input_write_address] <= input_write_data;

        end

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
    // TOP-LEVEL CONTROL FSM WITH SEQUENTIAL ARGMAX
    // ========================================================

    always_ff @(posedge clk) begin

        if (reset) begin

            // Return the TPU to its idle state.
            state <= IDLE;

            // Clear the previous prediction.
            predicted_class <= 5'd0;

            // Reset argmax tracking registers.
            argmax_counter <= 5'd0;
            argmax_best_score <= -9'sd129;
            argmax_best_index <= 5'd0;

        end

        else begin

            case (state)

                // ------------------------------------------------
                // IDLE
                // ------------------------------------------------

                IDLE: begin

                    if (start)
                        state <= START_HIDDEN;

                end

                // ------------------------------------------------
                // HIDDEN LAYER
                // ------------------------------------------------

                START_HIDDEN: begin

                    state <= WAIT_HIDDEN;

                end

                WAIT_HIDDEN: begin

                    if (hidden_done)
                        state <= START_OUTPUT;

                end

                // ------------------------------------------------
                // OUTPUT LAYER
                // ------------------------------------------------

                START_OUTPUT: begin

                    state <= WAIT_OUTPUT;

                end

                WAIT_OUTPUT: begin

                    if (output_done) begin

                        // Prepare to scan all 32 output scores.
                        argmax_counter <= 5'd0;

                        // Initialize below the minimum possible
                        // signed INT8 output score.
                        argmax_best_score <= -9'sd129;

                        argmax_best_index <= 5'd0;

                        state <= FIND_MAX;

                    end

                end

                // ------------------------------------------------
                // SEQUENTIAL ARGMAX
                // ------------------------------------------------

                FIND_MAX: begin

                    // Compare the current score against the
                    // highest score found so far.
                    //
                    // Strict greater-than preserves the original
                    // behavior: ties select the first class.

                    if (argmax_current_score > argmax_best_score) begin

                        argmax_best_score <= argmax_current_score;
                        argmax_best_index <= argmax_counter;

                    end

                    // Check whether class 31 is the final score.
                    if (argmax_counter == 5'd31) begin

                        // Handle class 31 winning in this same cycle.
                        // Nonblocking assignments mean best_index
                        // would otherwise still contain its old value.
                        if (argmax_current_score > argmax_best_score)
                            predicted_class <= argmax_counter;

                        else
                            predicted_class <= argmax_best_index;

                        state <= FINISHED;

                    end

                    else begin

                        // Advance to the next class.
                        argmax_counter <= argmax_counter + 1'b1;

                    end

                end

                // ------------------------------------------------
                // FINISHED
                // ------------------------------------------------

                FINISHED: begin

                    state <= IDLE;

                end

                // ------------------------------------------------
                // DEFAULT
                // ------------------------------------------------

                default: begin

                    state <= IDLE;

                end

            endcase

        end

    end

    // ========================================================
    // TOP-LEVEL STATUS SIGNALS
    // ========================================================

    // done is asserted for one clock cycle while in FINISHED.
    assign done = (state == FINISHED);

    // busy is high during inference and argmax processing.
    assign busy = (state != IDLE) && (state != FINISHED);

endmodule
