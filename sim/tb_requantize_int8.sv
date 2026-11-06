`timescale 1ns / 1ps

module tb_requantize_int8;

    // Inputs to the requantization module
    reg signed [31:0] accumulator;
    reg signed [31:0] multiplier;
    reg signed [7:0] shift;
    reg signed [7:0] output_zero_point;
    reg relu;

    // Output from the requantization module
    wire signed [7:0] result;

    // Instantiate the module being tested
    requantize_int8 uut (
        .accumulator(accumulator),
        .multiplier(multiplier),
        .shift(shift),
        .output_zero_point(output_zero_point),
        .relu(relu),
        .result(result)
    );

    initial begin

        // Test 1: Neuron 0 from the Python reference
        accumulator = 32'sd389270;
        multiplier = 32'sd1608709942;
        shift = -8'sd13;
        output_zero_point = -8'sd128;
        relu = 1'b1;

        #10;

        $display("--------------------------------");
        $display("REQUANTIZATION TEST");
        $display("--------------------------------");
        $display("Accumulator: %0d", accumulator);
        $display("Multiplier: %0d", multiplier);
        $display("Shift: %0d", shift);
        $display("Output zero point: %0d", output_zero_point);
        $display("ReLU enabled: %b", relu);
        $display("RTL output: %0d", result);
        $display("Expected output: -92");

        if (result == -8'sd92)
            $display("PASS: Output matches Python!");
        else
            $display("FAIL: Output does not match Python!");

        $display("--------------------------------");

        #10;
        $finish;

    end

endmodule