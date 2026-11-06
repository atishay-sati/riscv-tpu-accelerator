`timescale 1ns / 1ps

module tb_dense_layer_real_model;

    // ========================================================
    // Configuration
    // ========================================================

    localparam DATA_WIDTH = 8;
    localparam ACC_WIDTH  = 32;
    localparam ADDR_WIDTH = 32;

    localparam INPUT_FEATURES = 784;
    localparam OUTPUT_NEURONS = 128;

    // ========================================================
    // Clock and reset
    // ========================================================

    logic clk;
    logic reset;
    logic start;

    initial clk = 0;
    always #5 clk = ~clk;

    // ========================================================
    // Controller configuration
    // ========================================================

    logic [ADDR_WIDTH-1:0] input_features;
    logic [ADDR_WIDTH-1:0] output_neurons;

    logic signed [7:0] output_zero_point;
    logic relu_enable;

    // ========================================================
    // Weight memory interface
    // ========================================================

    logic weight_read_enable;
    logic [ADDR_WIDTH-1:0] weight_address;
    logic signed [7:0] weight_data;

    // ========================================================
    // Activation memory interface
    // ========================================================

    logic activation_read_enable;
    logic [ADDR_WIDTH-1:0] activation_address;
    logic [7:0] activation_data;

    // ========================================================
    // Bias memory interface
    // ========================================================

    logic bias_read_enable;
    logic [ADDR_WIDTH-1:0] bias_address;
    logic signed [31:0] bias_data;

    // ========================================================
    // Quantization parameter memory interfaces
    // ========================================================

    logic multiplier_read_enable;
    logic [ADDR_WIDTH-1:0] multiplier_address;
    logic signed [31:0] multiplier_data;

    logic shift_read_enable;
    logic [ADDR_WIDTH-1:0] shift_address;
    logic signed [7:0] shift_data;

    // ========================================================
    // Result interface
    // ========================================================

    logic result_valid;
    logic result_ready;

    logic result_write_enable;
    logic [ADDR_WIDTH-1:0] result_address;
    logic signed [7:0] result_data;

    logic busy;
    logic done;

    // ========================================================
    // Memory arrays
    // ========================================================

    logic signed [7:0] weight_mem
        [0:OUTPUT_NEURONS*INPUT_FEATURES-1];

    logic [7:0] activation_mem
        [0:INPUT_FEATURES-1];

    logic signed [31:0] bias_mem
        [0:OUTPUT_NEURONS-1];

    logic signed [31:0] multiplier_mem
        [0:OUTPUT_NEURONS-1];

    logic signed [7:0] shift_mem
        [0:OUTPUT_NEURONS-1];

    logic signed [7:0] expected_activations
        [0:OUTPUT_NEURONS-1];

    // ========================================================
    // Instantiate dense layer controller
    // ========================================================

    dense_layer_controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (
        .clk(clk),
        .reset(reset),
        .start(start),

        .input_features(input_features),
        .output_neurons(output_neurons),

        .weight_read_enable(weight_read_enable),
        .weight_address(weight_address),
        .weight_data(weight_data),

        .activation_read_enable(activation_read_enable),
        .activation_address(activation_address),
        .activation_data(activation_data),

        .bias_read_enable(bias_read_enable),
        .bias_address(bias_address),
        .bias_data(bias_data),

        .multiplier_read_enable(multiplier_read_enable),
        .multiplier_address(multiplier_address),
        .multiplier_data(multiplier_data),

        .shift_read_enable(shift_read_enable),
        .shift_address(shift_address),
        .shift_data(shift_data),

        .output_zero_point(output_zero_point),
        .relu_enable(relu_enable),

        .result_valid(result_valid),
        .result_ready(result_ready),

        .result_write_enable(result_write_enable),
        .result_address(result_address),
        .result_data(result_data),

        .busy(busy),
        .done(done)
    );

    // ========================================================
    // Load real model data
    // ========================================================

    initial begin

        $display("------------------------------------------");
        $display("Loading real QuickDraw model parameters");
        $display("------------------------------------------");

        $readmemh(
            "C:/Users/Atish/QuickDrawModel/fpga_params/dense_layer_test/hidden_weights.mem",
            weight_mem
        );

        $readmemh(
            "C:/Users/Atish/QuickDrawModel/fpga_params/dense_layer_test/hidden_biases.mem",
            bias_mem
        );

        $readmemh(
            "C:/Users/Atish/QuickDrawModel/fpga_params/dense_layer_test/hidden_multipliers.mem",
            multiplier_mem
        );

        $readmemh(
            "C:/Users/Atish/QuickDrawModel/fpga_params/dense_layer_test/hidden_shifts.mem",
            shift_mem
        );

        $readmemh(
            "C:/Users/Atish/QuickDrawModel/fpga_params/dense_layer_test/input_pixels.mem",
            activation_mem
        );

        $readmemh(
            "C:/Users/Atish/QuickDrawModel/fpga_params/dense_layer_test/expected_activations.mem",
            expected_activations
        );

        $display("All model files loaded.");

    end

    // ========================================================
    // Synchronous memory models
    //
    // These memories return data one clock cycle after
    // read_enable is asserted.
    // ========================================================

    always_ff @(posedge clk) begin

        if (weight_read_enable)
            weight_data <= weight_mem[weight_address];

        if (activation_read_enable)
            activation_data <= activation_mem[activation_address];

        if (bias_read_enable)
            bias_data <= bias_mem[bias_address];

        if (multiplier_read_enable)
            multiplier_data <= multiplier_mem[multiplier_address];

        if (shift_read_enable)
            shift_data <= shift_mem[shift_address];

    end

    // ========================================================
    // Scoreboard
    // ========================================================

    integer passed;
    integer failed;

    integer neuron_index;

    logic signed [7:0] actual_value;
    logic signed [7:0] expected_value;

    always @(posedge clk) begin

        if (!reset && result_write_enable) begin

            neuron_index = result_address;

            actual_value = result_data;
            expected_value = expected_activations[neuron_index];

            if (actual_value === expected_value) begin

                passed = passed + 1;

                $display(
                    "PASS neuron %0d: actual=%0d expected=%0d",
                    neuron_index,
                    actual_value,
                    expected_value
                );

            end else begin

                failed = failed + 1;

                $display(
                    "FAIL neuron %0d: actual=%0d expected=%0d",
                    neuron_index,
                    actual_value,
                    expected_value
                );

                $display(
                    "      actual hex=%h expected hex=%h",
                    actual_value,
                    expected_value
                );

            end

        end

    end

    // ========================================================
    // Main test
    // ========================================================

    initial begin

        // Initialize signals.
        reset = 1;
        start = 0;

        input_features = INPUT_FEATURES;
        output_neurons = OUTPUT_NEURONS;

        // Input quantization has zero point -128.
        // The memory file contains adjusted pixels 0..255.
        output_zero_point = -8'sd128;

        // Hidden layer uses ReLU.
        relu_enable = 1;

        // Accept every output immediately.
        result_ready = 1;

        passed = 0;
        failed = 0;

        neuron_index = 0;

        actual_value = 0;
        expected_value = 0;

        // Allow reset to take effect.
        repeat (5) @(negedge clk);

        reset = 0;

        repeat (3) @(negedge clk);

        $display("");
        $display("------------------------------------------");
        $display("STARTING REAL MODEL HIDDEN LAYER TEST");
        $display("------------------------------------------");

        $display("Input features: %0d", INPUT_FEATURES);
        $display("Output neurons: %0d", OUTPUT_NEURONS);
        $display("Output zero point: %0d", output_zero_point);
        $display("ReLU enabled: %0d", relu_enable);

        $display("Feature tiles per group: 98");
        $display("Neuron groups: 16");
        $display("Total systolic tile operations: 1568");

        // Start the controller.
        start = 1;

        @(negedge clk);
        start = 0;

        // Wait for the entire hidden layer to finish.
        wait(done == 1'b1);

        // Allow the scoreboard to finish processing the
        // final clock edge.
        @(negedge clk);
        #1;

        $display("");
        $display("------------------------------------------");
        $display("REAL MODEL TEST RESULTS");
        $display("------------------------------------------");

        $display("Expected outputs: %0d", OUTPUT_NEURONS);
        $display("Passed: %0d", passed);
        $display("Failed: %0d", failed);

        if ((passed == OUTPUT_NEURONS) && (failed == 0)) begin

            $display("");
            $display("PASS: ALL 128 HIDDEN NEURONS MATCH!");
            $display("The FPGA hidden layer matches TensorFlow Lite.");

        end else begin

            $display("");
            $display("FAIL: HIDDEN LAYER MISMATCHES DETECTED.");

        end

        $display("------------------------------------------");

        $finish;

    end

    // ========================================================
    // Simulation timeout
    // ========================================================

    initial begin

        // 100 ms simulation-time limit.
        #100_000_000;

        $display("");
        $display("ERROR: Simulation timeout.");
        $display("Passed so far: %0d", passed);
        $display("Failed so far: %0d", failed);

        $fatal;

    end

endmodule
