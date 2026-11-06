`timescale 1ns / 1ps

// ============================================================
// Testbench: matrix_engine_tb
//
// Purpose:
//   Verify the matrix_engine by calculating C = A x B.
//
// Test matrix A:
//       | 1  2 |
//       | 3  4 |
//
// Test matrix B:
//       | 5  6 |
//       | 7  8 |
//
// Expected matrix C:
//       | 19  22 |
//       | 43  50 |
//
// A testbench is used only for simulation, not synthesized
// into the FPGA hardware.
// ============================================================

module matrix_engine_tb;

    // -------------------- Testbench signals -----------------

    logic clk = 0;
    // Simulated clock.
    // Toggles every 5 ns, creating a 10 ns clock period.

    logic reset = 0;
    // Drives the matrix engine's reset input.

    logic start = 0;
    // Requests a new matrix multiplication.

    // -------------------- Matrix A --------------------------

    logic signed [7:0] a00, a01, a10, a11;
    // Four signed 8-bit elements of matrix A.

    // -------------------- Matrix B --------------------------

    logic signed [7:0] b00, b01, b10, b11;
    // Four signed 8-bit elements of matrix B.

    // -------------------- Matrix C outputs ------------------

    logic signed [31:0] c00, c01, c10, c11;
    // Four 32-bit matrix multiplication results.

    logic done;
    // Indicates that the matrix engine has completed its work.

    // -------------------- Device under test -----------------

    // Connect the testbench signals to the matrix_engine module.
    matrix_engine dut (
        .clk(clk),
        .reset(reset),
        .start(start),

        .a00(a00),
        .a01(a01),
        .a10(a10),
        .a11(a11),

        .b00(b00),
        .b01(b01),
        .b10(b10),
        .b11(b11),

        .c00(c00),
        .c01(c01),
        .c10(c10),
        .c11(c11),

        .done(done)
    );

    // -------------------- Clock generation ------------------

    // Invert clk every 5 ns.
    // Full clock period = 10 ns, or 100 MHz in simulation.
    always #5 clk = ~clk;

    // -------------------- Test procedure --------------------

    initial begin

        // Load matrix A:
        //
        // A = | 1  2 |
        //     | 3  4 |
        a00 = 8'sd1;
        a01 = 8'sd2;
        a10 = 8'sd3;
        a11 = 8'sd4;

        // Load matrix B:
        //
        // B = | 5  6 |
        //     | 7  8 |
        b00 = 8'sd5;
        b01 = 8'sd6;
        b10 = 8'sd7;
        b11 = 8'sd8;

        // -------------------- Reset -------------------------

        // Assert reset before beginning the calculation.
        reset = 1;

        // Wait until a falling edge so reset is held across
        // at least one rising edge of the clock.
        @(negedge clk);

        // Release reset.
        reset = 0;

        // -------------------- Start operation ---------------

        // Request the matrix multiplication.
        start = 1;

        // Keep start high until the next falling edge.
        @(negedge clk);

        // Deassert start after the request is accepted.
        start = 0;

        // -------------------- Wait for completion ------------

        // Wait until the controller asserts done.
        wait(done == 1);

        // Allow the output signals to settle before checking.
        #1;

        // -------------------- Display results ----------------

        $display("Matrix A:");
        $display("%0d %0d", a00, a01);
        $display("%0d %0d", a10, a11);

        $display("Matrix B:");
        $display("%0d %0d", b00, b01);
        $display("%0d %0d", b10, b11);

        $display("Result C = A x B:");
        $display("%0d %0d", c00, c01);
        $display("%0d %0d", c10, c11);

        // -------------------- Verify expected values ----------

        // Expected:
        // c00 = 1*5 + 2*7 = 19
        // c01 = 1*6 + 2*8 = 22
        // c10 = 3*5 + 4*7 = 43
        // c11 = 3*6 + 4*8 = 50

        if (c00 !== 32'sd19 ||
            c01 !== 32'sd22 ||
            c10 !== 32'sd43 ||
            c11 !== 32'sd50) begin

            // Stop simulation if any output is incorrect.
            $fatal(1, "FAIL: Matrix multiplication results are incorrect!");

        end

        // Reaching this line means every output matched.
        $display("PASS: Matrix multiplication results are correct!");

        // End the simulation.
        $finish;

    end

endmodule