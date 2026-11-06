`timescale 1ns / 1ps

module tb_output_layer_real_model;

    // ============================================================
    // CONFIGURATION
    // ============================================================

    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter ADDR_WIDTH = 32;

    // Output layer: 128 hidden activations -> 32 classes
    parameter INPUT_FEATURES = 128;
    parameter OUTPUT_NEURONS = 32;

    // ============================================================
    // CLOCK AND RESET
    // ============================================================

    logic clk;
    logic reset;
    logic start;

    // 10 ns clock period = 100 MHz
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // ============================================================
    // CONTROLLER CONFIGURATION
    // ============================================================

    logic [ADDR_WIDTH-1:0] input_features;
    logic [ADDR_WIDTH-1:0] output_neurons;

    logic signed [7:0] output_zero_point;
    logic relu_enable;

    // ============================================================
    // WEIGHT MEMORY
    // ============================================================

    logic weight_read_enable;
    logic [ADDR_WIDTH-1:0] weight_address;
    logic signed [DATA_WIDTH-1:0] weight_data;

    // 32 output classes * 128 input activations
    logic signed [7:0] weight_mem [0:4095];

    // ============================================================
    // ACTIVATION MEMORY
    // ============================================================

    logic activation_read_enable;
    logic [ADDR_WIDTH-1:0] activation_address;
    logic [DATA_WIDTH-1:0] activation_data;

    // 128 hidden-layer activations
    logic [7:0] activation_mem [0:127];

    // ============================================================
    // BIAS MEMORY
    // ============================================================

    logic bias_read_enable;
    logic [ADDR_WIDTH-1:0] bias_address;
    logic signed [ACC_WIDTH-1:0] bias_data;

    logic signed [31:0] bias_mem [0:31];

    // ============================================================
    // MULTIPLIER MEMORY
    // ============================================================

    logic multiplier_read_enable;
    logic [ADDR_WIDTH-1:0] multiplier_address;
    logic signed [31:0] multiplier_data;

    logic signed [31:0] multiplier_mem [0:31];

    // ============================================================
    // SHIFT MEMORY
    // ============================================================

    logic shift_read_enable;
    logic [ADDR_WIDTH-1:0] shift_address;
    logic signed [7:0] shift_data;

    logic signed [7:0] shift_mem [0:31];

    // ============================================================
    // RESULT INTERFACE
    // ============================================================

    logic result_valid;
    logic result_ready;

    logic result_write_enable;
    logic [ADDR_WIDTH-1:0] result_address;
    logic signed [7:0] result_data;

    logic busy;
    logic done;

    // ============================================================
    // EXPECTED OUTPUTS FROM TENSORFLOW LITE
    // ============================================================

    logic signed [7:0] expected_outputs [0:31];

    // Store the FPGA's output values for predicted-class comparison.
    integer actual_outputs [0:31];

    // Track which output classes have been checked.
    logic seen [0:31];

    integer passed;
    integer failed;
    integer i;

    integer expected_argmax;
    integer actual_argmax;

    integer expected_max;
    integer actual_max;

    // ============================================================
    // INSTANTIATE DENSE LAYER CONTROLLER
    // ============================================================

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

        // Weight memory
        .weight_read_enable(weight_read_enable),
        .weight_address(weight_address),
        .weight_data(weight_data),

        // Activation memory
        .activation_read_enable(activation_read_enable),
        .activation_address(activation_address),
        .activation_data(activation_data),

        // Bias memory
        .bias_read_enable(bias_read_enable),
        .bias_address(bias_address),
        .bias_data(bias_data),

        // Multiplier memory
        .multiplier_read_enable(multiplier_read_enable),
        .multiplier_address(multiplier_address),
        .multiplier_data(multiplier_data),

        // Shift memory
        .shift_read_enable(shift_read_enable),
        .shift_address(shift_address),
        .shift_data(shift_data),

        // Quantization configuration
        .output_zero_point(output_zero_point),
        .relu_enable(relu_enable),

        // Result interface
        .result_valid(result_valid),
        .result_ready(result_ready),

        .result_write_enable(result_write_enable),
        .result_address(result_address),
        .result_data(result_data),

        .busy(busy),
        .done(done)
    );

    // ============================================================
    // LOAD MEMORY FILES
    // ============================================================

    initial begin

        $display("========================================");
        $display("LOADING OUTPUT LAYER MEMORY FILES");
        $display("========================================");

        $readmemh(
            "C:/Users/Atish/QuickDrawModel/fpga_params/output_layer_test/output_weights.mem",
            weight_mem
        );

        $readmemh(
            "C:/Users/Atish/QuickDrawModel/fpga_params/output_layer_test/input_activations.mem",
            activation_mem
        );

        $readmemh(
            "C:/Users/Atish/QuickDrawModel/fpga_params/output_layer_test/output_biases.mem",
            bias_mem
        );

        $readmemh(
            "C:/Users/Atish/QuickDrawModel/fpga_params/output_layer_test/output_multipliers.mem",
            multiplier_mem
        );

        $readmemh(
            "C:/Users/Atish/QuickDrawModel/fpga_params/output_layer_test/output_shifts.mem",
            shift_mem
        );

        $readmemh(
            "C:/Users/Atish/QuickDrawModel/fpga_params/output_layer_test/expected_outputs.mem",
            expected_outputs
        );

        $display("All memory files loaded.");
        $display("");

    end

    // ============================================================
    // SYNCHRONOUS MEMORY MODEL
    // ============================================================

    // Mimics FPGA block RAM behavior.
    // When a read enable is asserted, the corresponding memory
    // value becomes available after the rising clock edge.

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

    // ============================================================
    // OUTPUT SCOREBOARD
    // ============================================================

    // Every time the controller writes an output, compare it
    // against the corresponding TensorFlow Lite result.

    always @(posedge clk) begin

        if (result_write_enable) begin

            if (result_address < OUTPUT_NEURONS) begin

                seen[result_address] = 1'b1;

                // Save the FPGA output for argmax comparison.
                actual_outputs[result_address] = $signed(result_data);

                if (result_data === expected_outputs[result_address]) begin

                    passed = passed + 1;

                    $display(
                        "PASS class %0d: actual=%0d expected=%0d",
                        result_address,
                        $signed(result_data),
                        $signed(expected_outputs[result_address])
                    );

                end

                else begin

                    failed = failed + 1;

                    $display(
                        "FAIL class %0d: actual=%0d expected=%0d",
                        result_address,
                        $signed(result_data),
                        $signed(expected_outputs[result_address])
                    );

                end

            end

            else begin

                failed = failed + 1;

                $display(
                    "ERROR: Invalid output address %0d",
                    result_address
                );

            end

        end

    end

    // ============================================================
    // MAIN TEST
    // ============================================================

    initial begin

        // Initialize controller inputs.
        reset = 1'b1;
        start = 1'b0;

        input_features = INPUT_FEATURES;
        output_neurons = OUTPUT_NEURONS;

        // Output layer quantization settings.
        output_zero_point = 8'sd39;

        // The output layer does NOT use ReLU.
        relu_enable = 1'b0;

        // The testbench always accepts output results.
        result_ready = 1'b1;

        passed = 0;
        failed = 0;

        // Initialize scoreboard tracking and stored outputs.
        for (i = 0; i < OUTPUT_NEURONS; i = i + 1) begin
            seen[i] = 1'b0;
            actual_outputs[i] = 0;
        end

        // Wait for memory initialization.
        repeat (5) @(negedge clk);

        // Release reset.
        reset = 1'b0;

        repeat (3) @(negedge clk);

        $display("========================================");
        $display("STARTING OUTPUT LAYER TEST");
        $display("========================================");

        $display("Input features: %0d", INPUT_FEATURES);
        $display("Output classes: %0d", OUTPUT_NEURONS);
        $display("Output zero point: %0d", output_zero_point);
        $display("ReLU enabled: %0d", relu_enable);

        $display("Feature tiles per group: %0d",
                 INPUT_FEATURES / 8);

        $display("Neuron groups: %0d",
                 OUTPUT_NEURONS / 8);

        $display("Total systolic tile operations: %0d",
                 (INPUT_FEATURES / 8) * (OUTPUT_NEURONS / 8));

        $display("");

        // Start the dense layer controller.
        start = 1'b1;

        @(negedge clk);

        start = 1'b0;

        // Wait until the controller finishes.
        wait (done == 1'b1);

        @(negedge clk);

        // ========================================================
        // CHECK FOR MISSING OUTPUTS
        // ========================================================

        for (i = 0; i < OUTPUT_NEURONS; i = i + 1) begin

            if (!seen[i]) begin

                failed = failed + 1;

                $display(
                    "MISSING class %0d: controller never produced output",
                    i
                );

            end

        end

        // ========================================================
        // CALCULATE EXPECTED AND ACTUAL PREDICTIONS
        // ========================================================

        expected_argmax = 0;
        actual_argmax = 0;

        expected_max = -1000000;
        actual_max = -1000000;

        for (i = 0; i < OUTPUT_NEURONS; i = i + 1) begin

            if ($signed(expected_outputs[i]) > expected_max) begin
                expected_max = $signed(expected_outputs[i]);
                expected_argmax = i;
            end

            if (actual_outputs[i] > actual_max) begin
                actual_max = actual_outputs[i];
                actual_argmax = i;
            end

        end

        // ========================================================
        // FINAL RESULTS
        // ========================================================

        $display("");
        $display("========================================");
        $display("OUTPUT LAYER TEST RESULTS");
        $display("========================================");

        $display("Expected outputs: %0d", OUTPUT_NEURONS);
        $display("Passed comparisons: %0d", passed);
        $display("Failed comparisons: %0d", failed);

        $display("TensorFlow Lite predicted class: %0d",
                 expected_argmax);

        $display("FPGA predicted class: %0d",
                 actual_argmax);

        if (expected_argmax == actual_argmax)
            $display("PASS: PREDICTED CLASSES MATCH!");
        else
            $display("FAIL: PREDICTED CLASSES DO NOT MATCH!");

        $display("");

        if (passed == OUTPUT_NEURONS && failed == 0) begin

            $display("PASS: ALL 32 OUTPUT CLASSES MATCH!");
            $display("The FPGA output layer matches TensorFlow Lite.");

        end

        else begin

            $display("FAIL: OUTPUT LAYER DOES NOT FULLY MATCH.");
            $display("Check the failed classes and requantization logic.");

        end

        $display("========================================");

        $finish;

    end

    // ============================================================
    // SIMULATION TIMEOUT
    // ============================================================

    initial begin

        #10_000_000;

        $display("");
        $display("ERROR: SIMULATION TIMEOUT");
        $display("The output layer did not finish in time.");

        $finish;

    end

endmodule