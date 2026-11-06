
`timescale 1ns / 1ps

// ============================================================
// Testbench: processing_element_tb
// Purpose:
//   Verify the processing_element module in simulation.
//
// A testbench is simulation code. It generates a clock, drives
// inputs, and checks outputs. It is not synthesized as hardware.
// ============================================================

module processing_element_tb;

    // -------------------- Testbench signals -----------------

    logic clk = 0;
    // Simulated clock. It toggles every 5 ns, giving a 10 ns
    // period, equivalent to a 100 MHz simulated clock.

    logic reset = 0;
    // Drives the PE's synchronous reset input.

    logic enable = 0;
    // Controls whether the PE accumulates the current product.

    logic signed [7:0] a = 0;
    // Test value for the PE's first signed operand.

    logic signed [7:0] b = 0;
    // Test value for the PE's second signed operand.

    logic signed [31:0] acc_out;
    // Receives the PE's accumulated result.

    // Instantiate the device under test (DUT).
    processing_element dut (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .a(a),
        .b(b),
        .acc_out(acc_out)
    );

    // Generate a clock that toggles every 5 ns.
    always #5 clk = ~clk;

    // Run the test cases in sequence.
    initial begin

        // TEST 1: Reset the accumulator.
        reset = 1;
        @(negedge clk);
        reset = 0;

        // TEST 2: Multiply 3*4 and accumulate 12.
        a = 8'sd3;
        b = 8'sd4;
        enable = 1;

        @(negedge clk);
        if (acc_out !== 32'sd12)
            $fatal(1, "Test 1 failed: got %0d", acc_out);

        // TEST 3: Add another product, 2*5 = 10.
        // The accumulator should become 12+10 = 22.
        a = 8'sd2;
        b = 8'sd5;

        @(negedge clk);
        if (acc_out !== 32'sd22)
            $fatal(1, "Test 2 failed: got %0d", acc_out);

        // TEST 4: Add (-2)*3 = -6.
        // The accumulator should become 22-6 = 16.
        a = -8'sd2;
        b = 8'sd3;

        @(negedge clk);
        if (acc_out !== 32'sd16)
            $fatal(1, "Test 3 failed: got %0d", acc_out);

        // TEST 5: Disable accumulation.
        // Even though 10*10 = 100, the accumulator must stay 16.
        enable = 0;
        a = 8'sd10;
        b = 8'sd10;

        @(negedge clk);
        if (acc_out !== 32'sd16)
            $fatal(1, "Test 4 failed: got %0d", acc_out);

        $display("All PE tests passed!");
        $finish;
    end

endmodule
