
`timescale 1ns / 1ps

module tb_requantize_int8_all;

    // Number of hidden neurons.
    localparam NUM_NEURONS = 128;

    // Memory arrays loaded from Python.
    reg [31:0] accumulators [0:NUM_NEURONS-1];
    reg [31:0] multipliers [0:NUM_NEURONS-1];
    reg [7:0] shifts [0:NUM_NEURONS-1];
    reg [7:0] expected_activations [0:NUM_NEURONS-1];

    // Inputs to the requantization module.
    reg signed [31:0] accumulator;
    reg signed [31:0] multiplier;
    reg signed [7:0] shift;
    reg signed [7:0] output_zero_point;
    reg relu;

    // Output from the requantization module.
    wire signed [7:0] result;

    // Test counters.
    integer i;
    integer passed;
    integer failed;

    // Instantiate the requantization module.
    requantize_int8 uut (
        .accumulator(accumulator),
        .multiplier(multiplier),
        .shift(shift),
        .output_zero_point(output_zero_point),
        .relu(relu),
        .result(result)
    );

    initial begin

        // ----------------------------------------
        // 1. Initialize inputs.
        // ----------------------------------------

        accumulator = 32'sd0;
        multiplier = 32'sd0;
        shift = 8'sd0;

        // Hidden layer output zero point.
        output_zero_point = -8'sd128;

        // Hidden layer uses ReLU.
        relu = 1'b1;

        passed = 0;
        failed = 0;

        // ----------------------------------------
        // 2. Load Python-generated memory files.
        // ----------------------------------------

        $readmemh(
            "C:/Users/Atish/QuickDrawModel/fpga_params/all_neurons_test/accumulators.mem",
            accumulators
        );

        $readmemh(
            "C:/Users/Atish/QuickDrawModel/fpga_params/all_neurons_test/multipliers.mem",
            multipliers
        );

        $readmemh(
            "C:/Users/Atish/QuickDrawModel/fpga_params/all_neurons_test/shifts.mem",
            shifts
        );

        $readmemh(
            "C:/Users/Atish/QuickDrawModel/fpga_params/all_neurons_test/expected_activations.mem",
            expected_activations
        );

        // ----------------------------------------
        // 3. Test all 128 neurons.
        // ----------------------------------------

        $display("----------------------------------------");
        $display("ALL HIDDEN NEURON REQUANTIZATION TEST");
        $display("----------------------------------------");

        for (i = 0; i < NUM_NEURONS; i = i + 1) begin

            // Load this neuron's parameters.
            accumulator = accumulators[i];
            multiplier = multipliers[i];
            shift = shifts[i];

            // Allow combinational logic to settle.
            #10;

            // Compare against the Python/TFLite result.
            if (result === expected_activations[i]) begin
                passed = passed + 1;
            end
            else begin
                failed = failed + 1;

                $display("FAIL: Neuron %0d", i);
                $display("  Accumulator: %0d", accumulator);
                $display("  Multiplier: %0d", multiplier);
                $display("  Shift: %0d", shift);
                $display("  RTL result: %0d", result);
                $display(
                    "  Expected: %0d",
                    $signed(expected_activations[i])
                );
            end

        end

        // ----------------------------------------
        // 4. Display final results.
        // ----------------------------------------

        $display("----------------------------------------");
        $display("REQUANTIZATION TEST RESULTS");
        $display("----------------------------------------");

        $display("Total neurons: %0d", NUM_NEURONS);
        $display("Passed: %0d", passed);
        $display("Failed: %0d", failed);

        if (failed == 0)
            $display("PASS: ALL 128 NEURONS MATCH!");
        else
            $display("FAIL: SOME NEURONS DO NOT MATCH.");

        $display("----------------------------------------");

        #10;
        $finish;

    end

endmodule
