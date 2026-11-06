`timescale 1ns / 1ps

// ============================================================
// Testbench: systolic_pe_tb
//
// Tests:
//   1. Reset clears the accumulator and forwarded outputs.
//   2. Signed multiplication and accumulation work correctly.
//   3. Inputs are forwarded one clock cycle later.
//   4. enable = 0 causes the PE to hold its values.
//   5. Negative numbers are handled correctly.
// ============================================================

module systolic_pe_tb;

    // -------------------- Testbench signals -----------------

    logic clk = 0;
    // Simulated clock, with a 10 ns period.

    logic reset = 0;
    // Clears the PE when sampled high on a rising clock edge.

    logic enable = 0;
    // Controls accumulation and forwarding.

    logic signed [7:0] a_in;
    // Test value entering the PE horizontally.

    logic signed [7:0] b_in;
    // Test value entering the PE vertically.

    logic signed [7:0] a_out;
    // Forwarded A value from the PE.

    logic signed [7:0] b_out;
    // Forwarded B value from the PE.

    logic signed [31:0] acc_out;
    // Accumulator output from the PE.

    // -------------------- Device under test -----------------

    systolic_pe dut (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .a_in(a_in),
        .b_in(b_in),
        .a_out(a_out),
        .b_out(b_out),
        .acc_out(acc_out)
    );

    // Toggle the clock every 5 ns.
    always #5 clk = ~clk;

    // -------------------- Test procedure --------------------

    initial begin

        // Initialize all input signals.
        reset = 0;
        enable = 0;
        a_in = 8'sd0;
        b_in = 8'sd0;

        // ====================================================
        // TEST 1: RESET
        // ====================================================

        // Assert reset before a rising clock edge.
        reset = 1;

        // Wait for the rising edge where reset is sampled.
        @(posedge clk);

        // Wait a little so nonblocking assignments update.
        #1;

        // Verify that the accumulator and forwarded outputs
        // have all been cleared.
        if (acc_out !== 32'sd0 ||
            a_out !== 8'sd0 ||
            b_out !== 8'sd0) begin

            $fatal(1, "FAIL: Reset did not clear PE outputs!");

        end

        $display("PASS: Reset");

        // Release reset away from the rising edge.
        @(negedge clk);
        reset = 0;

        // ====================================================
        // TEST 2: MULTIPLICATION AND ACCUMULATION
        // ====================================================

        // First product: 3 * 4 = 12.
        a_in = 8'sd3;
        b_in = 8'sd4;
        enable = 1;

        // Wait for the PE to capture the inputs and accumulate.
        @(posedge clk);
        #1;

        // Verify the accumulator and forwarded values.
        if (acc_out !== 32'sd12 ||
            a_out !== 8'sd3 ||
            b_out !== 8'sd4) begin

            $fatal(1, "FAIL: First multiplication or forwarding failed!");

        end

        $display("PASS: 3 * 4 = 12");

        // Second product: 2 * 5 = 10.
        // The accumulator should become 12 + 10 = 22.
        @(negedge clk);
        a_in = 8'sd2;
        b_in = 8'sd5;

        @(posedge clk);
        #1;

        if (acc_out !== 32'sd22 ||
            a_out !== 8'sd2 ||
            b_out !== 8'sd5) begin

            $fatal(1, "FAIL: Accumulation or forwarding failed!");

        end

        $display("PASS: Accumulation gives 22");

        // ====================================================
        // TEST 3: SIGNED MULTIPLICATION
        // ====================================================

        // Third product: -2 * 3 = -6.
        // The accumulator should become 22 - 6 = 16.
        @(negedge clk);
        a_in = -8'sd2;
        b_in = 8'sd3;

        @(posedge clk);
        #1;

        if (acc_out !== 32'sd16 ||
            a_out !== -8'sd2 ||
            b_out !== 8'sd3) begin

            $fatal(1, "FAIL: Signed multiplication failed!");

        end

        $display("PASS: Signed multiplication gives 16");

        // ====================================================
        // TEST 4: ENABLE LOW
        // ====================================================

        // Disable the PE and change its inputs.
        // The accumulator and forwarded outputs should hold
        // their previous values even though the inputs change.
        @(negedge clk);
        enable = 0;
        a_in = 8'sd7;
        b_in = 8'sd8;

        @(posedge clk);
        #1;

        if (acc_out !== 32'sd16 ||
            a_out !== -8'sd2 ||
            b_out !== 8'sd3) begin

            $fatal(1, "FAIL: PE did not hold values when disabled!");

        end

        $display("PASS: Enable low holds values");

        // ====================================================
        // TEST 5: RESET AFTER ACCUMULATION
        // ====================================================

        // Assert reset again to confirm it clears a nonzero sum.
        @(negedge clk);
        reset = 1;

        @(posedge clk);
        #1;

        if (acc_out !== 32'sd0 ||
            a_out !== 8'sd0 ||
            b_out !== 8'sd0) begin

            $fatal(1, "FAIL: Second reset failed!");

        end

        $display("PASS: Reset clears accumulated data");

        // ====================================================
        // ALL TESTS PASSED
        // ====================================================

        $display("--------------------------------------");
        $display("ALL SYSTOLIC PE TESTS PASSED!");
        $display("--------------------------------------");

        $finish;

    end

endmodule