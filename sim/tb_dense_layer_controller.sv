`timescale 1ns / 1ps

module tb_dense_layer_controller;

    // --------------------------------------------------------
    // Parameters
    // --------------------------------------------------------

    localparam DATA_WIDTH = 8;
    localparam ACC_WIDTH  = 32;
    localparam ADDR_WIDTH = 32;

    localparam MAX_FEATURES = 16;
    localparam MAX_NEURONS  = 10;

    // --------------------------------------------------------
    // Clock and reset
    // --------------------------------------------------------

    logic clk;
    logic reset;
    logic start;

    initial clk = 0;
    always #5 clk = ~clk;

    // --------------------------------------------------------
    // Layer configuration
    // --------------------------------------------------------

    logic [ADDR_WIDTH-1:0] input_features;
    logic [ADDR_WIDTH-1:0] output_neurons;

    // --------------------------------------------------------
    // Memory interfaces
    // --------------------------------------------------------

    logic weight_read_enable;
    logic [ADDR_WIDTH-1:0] weight_address;
    logic signed [DATA_WIDTH-1:0] weight_data;

    logic activation_read_enable;
    logic [ADDR_WIDTH-1:0] activation_address;
    logic [DATA_WIDTH-1:0] activation_data;

    logic bias_read_enable;
    logic [ADDR_WIDTH-1:0] bias_address;
    logic signed [ACC_WIDTH-1:0] bias_data;

    logic multiplier_read_enable;
    logic [ADDR_WIDTH-1:0] multiplier_address;
    logic signed [31:0] multiplier_data;

    logic shift_read_enable;
    logic [ADDR_WIDTH-1:0] shift_address;
    logic signed [7:0] shift_data;

    // --------------------------------------------------------
    // Quantization settings
    // --------------------------------------------------------

    logic signed [7:0] output_zero_point;
    logic relu_enable;

    // --------------------------------------------------------
    // Result interface
    // --------------------------------------------------------

    logic result_valid;
    logic result_ready;

    logic result_write_enable;
    logic [ADDR_WIDTH-1:0] result_address;
    logic signed [7:0] result_data;

    logic busy;
    logic done;

    // --------------------------------------------------------
    // Simulated memories
    // --------------------------------------------------------

    logic signed [7:0] weight_mem [0:MAX_FEATURES*MAX_NEURONS-1];
    logic [7:0] activation_mem [0:MAX_FEATURES-1];

    logic signed [31:0] bias_mem [0:MAX_NEURONS-1];
    logic signed [31:0] multiplier_mem [0:MAX_NEURONS-1];
    logic signed [7:0] shift_mem [0:MAX_NEURONS-1];

    // --------------------------------------------------------
    // Expected results and scoreboard
    // --------------------------------------------------------

    integer expected_results [0:MAX_NEURONS-1];

    integer received_results;
    integer expected_count;
    integer errors;
    integer test_errors;

    string current_test;

    // --------------------------------------------------------
    // Instantiate dense layer controller
    // --------------------------------------------------------

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

    // --------------------------------------------------------
    // Synchronous memory models
    //
    // One clock cycle of read latency.
    // --------------------------------------------------------

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

    // --------------------------------------------------------
    // Scoreboard
    //
    // Check every accepted result.
    // --------------------------------------------------------

    always @(posedge clk) begin

        if (result_write_enable) begin

            $display(
                "[%s] Neuron %0d: RTL = %0d, Expected = %0d",
                current_test,
                result_address,
                result_data,
                expected_results[result_address]
            );

            if (result_address >= expected_count) begin

                $display("ERROR: Unexpected output address.");

                errors = errors + 1;
                test_errors = test_errors + 1;

            end
            else if (result_data !== expected_results[result_address]) begin

                $display(
                    "ERROR: Incorrect output for neuron %0d",
                    result_address
                );

                errors = errors + 1;
                test_errors = test_errors + 1;

            end

            received_results = received_results + 1;

        end

    end

    // --------------------------------------------------------
    // Testbench variables
    // --------------------------------------------------------

    integer i;
    integer j;

    logic signed [7:0] held_result;
    logic [ADDR_WIDTH-1:0] held_address;

    // --------------------------------------------------------
    // Reset DUT between tests
    // --------------------------------------------------------

    task automatic reset_dut;

        begin

            @(negedge clk);
            reset = 1;
            start = 0;
            result_ready = 0;

            repeat (4) @(posedge clk);

            @(negedge clk);
            reset = 0;

            repeat (2) @(posedge clk);
            @(negedge clk);

        end

    endtask

    // --------------------------------------------------------
    // Clear simulated memories
    // --------------------------------------------------------

    task automatic clear_memories;

        integer k;

        begin

            for (k = 0; k < MAX_FEATURES*MAX_NEURONS; k = k + 1)
                weight_mem[k] = 0;

            for (k = 0; k < MAX_FEATURES; k = k + 1)
                activation_mem[k] = 0;

            for (k = 0; k < MAX_NEURONS; k = k + 1) begin

                bias_mem[k] = 0;

                // Effective multiplier = 0.5
                multiplier_mem[k] = 32'sd1073741824;

                shift_mem[k] = 0;

                expected_results[k] = 0;

            end

        end

    endtask

    // --------------------------------------------------------
    // Run one test
    //
    // stall_first_result = 1:
    //     Test the valid/ready handshake.
    //
    // stall_first_result = 0:
    //     Accept results immediately.
    // --------------------------------------------------------

    task automatic run_test(input logic stall_first_result);

        begin

            received_results = 0;
            test_errors = 0;

            $display("");
            $display("----------------------------------------");
            $display("RUNNING: %s", current_test);
            $display("----------------------------------------");

            if (stall_first_result)
                result_ready = 0;
            else
                result_ready = 1;

            // Start the layer.
            start = 1;

            @(negedge clk);
            start = 0;

            // Optional handshake stall test.
            if (stall_first_result) begin

                wait (result_valid == 1);

                $display("First result is valid.");

                held_result = result_data;
                held_address = result_address;

                repeat (3) begin

                    @(posedge clk);

                    if (!result_valid ||
                        result_data !== held_result ||
                        result_address !== held_address) begin

                        $display(
                            "ERROR: Result changed during handshake stall."
                        );

                        errors = errors + 1;
                        test_errors = test_errors + 1;

                    end

                end

                $display("Handshake stall test completed.");

                @(negedge clk);
                result_ready = 1;

            end

            // Wait for the controller to finish.
            wait (done == 1);

            @(negedge clk);
            result_ready = 0;

            // Give the scoreboard time to finish.
            repeat (2) @(posedge clk);

            // Check result count.
            if (received_results != expected_count) begin

                $display(
                    "ERROR: Expected %0d results, received %0d.",
                    expected_count,
                    received_results
                );

                errors = errors + 1;
                test_errors = test_errors + 1;

            end

            if (test_errors == 0) begin

                $display("PASS: %s", current_test);

            end else begin

                $display("FAIL: %s", current_test);

            end

            $display("Results received: %0d", received_results);
            $display("Test errors: %0d", test_errors);

        end

    endtask

    // ========================================================
    // MAIN TEST SEQUENCE
    // ========================================================

    initial begin

        $display("");
        $display("========================================");
        $display("DENSE LAYER CONTROLLER TEST SUITE");
        $display("========================================");

        // Initial values.
        reset = 1;
        start = 0;
        result_ready = 0;

        input_features = 0;
        output_neurons = 0;

        output_zero_point = 0;
        relu_enable = 0;

        weight_data = 0;
        activation_data = 0;
        bias_data = 0;
        multiplier_data = 0;
        shift_data = 0;

        received_results = 0;
        expected_count = 0;
        errors = 0;
        test_errors = 0;

        current_test = "INITIALIZATION";

        clear_memories();

        repeat (5) @(posedge clk);
        @(negedge clk);
        reset = 0;

        // ====================================================
        // TEST 1: BASIC 8 INPUTS, 8 NEURONS
        // ====================================================

        reset_dut();
        clear_memories();

        current_test = "TEST 1: BASIC 8x8";

        input_features = 8;
        output_neurons = 8;

        expected_count = 8;

        // All activations = 2.
        for (i = 0; i < 8; i = i + 1)
            activation_mem[i] = 8'd2;

        // Neuron n has weights equal to n+1.
        for (i = 0; i < 8; i = i + 1) begin

            for (j = 0; j < 8; j = j + 1)
                weight_mem[i*8+j] = i + 1;

            expected_results[i] = (i + 1) * 8;

        end

        run_test(1'b1);

        // ====================================================
        // TEST 2: PADDING
        //
        // 10 inputs, 10 neurons.
        //
        // Tests:
        // - Partial second feature tile.
        // - Partial second neuron group.
        // ====================================================

        reset_dut();
        clear_memories();

        current_test = "TEST 2: PADDING 10x10";

        input_features = 10;
        output_neurons = 10;

        expected_count = 10;

        // All ten activations = 2.
        for (i = 0; i < 10; i = i + 1)
            activation_mem[i] = 8'd2;

        // Row-major weights.
        for (i = 0; i < 10; i = i + 1) begin

            for (j = 0; j < 10; j = j + 1)
                weight_mem[i*10+j] = i + 1;

            // Dot product = 10 * 2 * (i+1).
            // Requantization multiplier = 0.5.
            expected_results[i] = (i + 1) * 10;

        end

        run_test(1'b0);

        // ====================================================
        // TEST 3: MULTIPLE FEATURE TILES
        //
        // 16 inputs, 8 neurons.
        //
        // Tests accumulation across two 8-feature tiles.
        // ====================================================

        reset_dut();
        clear_memories();

        current_test = "TEST 3: MULTIPLE TILES 16x8";

        input_features = 16;
        output_neurons = 8;

        expected_count = 8;

        // All sixteen activations = 2.
        for (i = 0; i < 16; i = i + 1)
            activation_mem[i] = 8'd2;

        // Each neuron has sixteen identical weights.
        for (i = 0; i < 8; i = i + 1) begin

            for (j = 0; j < 16; j = j + 1)
                weight_mem[i*16+j] = i + 1;

            // Dot product = 16 * 2 * (i+1).
            // Requantization multiplier = 0.5.
            expected_results[i] = (i + 1) * 16;

            // INT8 saturation: maximum signed value is 127.
            if (expected_results[i] > 127)
                expected_results[i] = 127;

        end

        run_test(1'b0);

        // ====================================================
        // TEST 4: RELU
        //
        // 8 inputs, 8 neurons.
        //
        // Even neurons have negative weights.
        // Odd neurons have positive weights.
        // ====================================================

        reset_dut();
        clear_memories();

        current_test = "TEST 4: RELU";

        input_features = 8;
        output_neurons = 8;

        expected_count = 8;

        output_zero_point = 0;
        relu_enable = 1;

        // All activations = 2.
        for (i = 0; i < 8; i = i + 1)
            activation_mem[i] = 8'd2;

        for (i = 0; i < 8; i = i + 1) begin

            for (j = 0; j < 8; j = j + 1) begin

                if ((i % 2) == 0)
                    weight_mem[i*8+j] = -(i + 1);
                else
                    weight_mem[i*8+j] = i + 1;

            end

            if ((i % 2) == 0)
                expected_results[i] = 0;
            else
                expected_results[i] = (i + 1) * 8;

        end

        run_test(1'b0);

        // ====================================================
        // FINAL SUMMARY
        // ====================================================

        $display("");
        $display("========================================");
        $display("FINAL TEST SUITE SUMMARY");
        $display("========================================");

        $display("Total errors: %0d", errors);

        if (errors == 0) begin

            $display("PASS: ALL DENSE LAYER TESTS PASSED!");

        end else begin

            $display("FAIL: SOME DENSE LAYER TESTS FAILED.");

        end

        $display("========================================");

        $finish;

    end

    // --------------------------------------------------------
    // Timeout protection
    // --------------------------------------------------------

    initial begin

        #1000000;

        $display("ERROR: Simulation timed out.");
        $finish;

    end

endmodule
