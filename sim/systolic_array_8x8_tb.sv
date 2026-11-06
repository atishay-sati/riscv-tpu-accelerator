
`timescale 1ns / 1ps

// ============================================================
// Testbench: systolic_array_8x8_tb
//
// Tests:
//   1. Reset the array.
//   2. Multiply two 8x8 matrices with positive values.
//   3. Check all 64 outputs.
//   4. Perform a second multiplication with signed A values
//      and unsigned B values.
//   5. Verify that previous accumulated values are cleared
//      automatically when a new operation starts.
//
// IMPORTANT:
//   Matrix A = signed INT8
//   Matrix B = unsigned INT8
//
//   This tests separate matrix multiplications.
//   It does NOT yet test neural-network multi-tile accumulation.
// ============================================================

module systolic_array_8x8_tb;

    // --------------------------------------------------------
    // Clock and control signals
    // --------------------------------------------------------

    logic clk;
    logic reset;
    logic start;

    logic done;
    logic busy;

    // --------------------------------------------------------
    // Matrix inputs and outputs
    // --------------------------------------------------------

    logic signed [7:0] matrix_a [0:7][0:7];

    // B is now unsigned because it represents activations.
    logic [7:0] matrix_b [0:7][0:7];

    wire signed [31:0] matrix_c [0:7][0:7];

    // Expected results calculated by the testbench.
    integer expected [0:7][0:7];

    integer r;
    integer c;

    // --------------------------------------------------------
    // Instantiate the design under test (DUT)
    // --------------------------------------------------------

    systolic_array_8x8 dut (
        .clk(clk),
        .reset(reset),
        .start(start),
        .matrix_a(matrix_a),
        .matrix_b(matrix_b),
        .matrix_c(matrix_c),
        .done(done),
        .busy(busy)
    );

    // --------------------------------------------------------
    // Generate a 100 MHz clock.
    // Period = 10 ns.
    // --------------------------------------------------------

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    // --------------------------------------------------------
    // Task: Load matrices and calculate expected results.
    //
    // Test 1:
    //   A[r][c] = r + c + 1
    //   B[r][c] = r + 2*c + 1
    //
    // Test 2:
    //   A[r][c] = r - c - 2
    //   B[r][c] = r + c + 1
    //
    // Matrix multiplication:
    //   C[r][c] = sum(A[r][kk] * B[kk][c])
    // --------------------------------------------------------

    task automatic load_and_calculate(input integer test_number);

        integer kk;

        begin

            // Load both input matrices.
            for (r = 0; r < 8; r = r + 1) begin
                for (c = 0; c < 8; c = c + 1) begin

                    if (test_number == 1) begin
                        matrix_a[r][c] = r + c + 1;
                        matrix_b[r][c] = r + 2*c + 1;
                    end
                    else begin
                        // Negative and positive signed weights.
                        matrix_a[r][c] = r - c - 2;

                        // Unsigned activations from 0 to 255.
                        matrix_b[r][c] = r + c + 1;
                    end

                end
            end

            // Calculate the expected matrix multiplication.
            for (r = 0; r < 8; r = r + 1) begin
                for (c = 0; c < 8; c = c + 1) begin

                    expected[r][c] = 0;

                    for (kk = 0; kk < 8; kk = kk + 1) begin

                        expected[r][c] = expected[r][c]
                            + (
                                int'($signed(matrix_a[r][kk]))
                                *
                                int'($unsigned(matrix_b[kk][c]))
                              );

                    end

                end
            end

        end
    endtask

    // --------------------------------------------------------
    // Task: Start multiplication and check all 64 outputs.
    // --------------------------------------------------------

    task automatic run_and_check(input integer test_number);

        begin

            // Start the operation for one clock cycle.
            @(negedge clk);
            start = 1'b1;

            @(negedge clk);
            start = 1'b0;

            // Wait for the array to finish.
            wait (done == 1'b1);

            // Sample after completion.
            @(negedge clk);

            // Check all 64 output values.
            for (r = 0; r < 8; r = r + 1) begin
                for (c = 0; c < 8; c = c + 1) begin

                    if (matrix_c[r][c] !== expected[r][c]) begin

                        $error(
                            "Test %0d FAILED: C[%0d][%0d] = %0d, expected %0d",
                            test_number,
                            r,
                            c,
                            matrix_c[r][c],
                            expected[r][c]
                        );

                        $fatal(1, "Matrix multiplication mismatch.");

                    end

                end
            end

            $display(
                "Test %0d PASSED: all 64 matrix outputs match.",
                test_number
            );

        end
    endtask

    // --------------------------------------------------------
    // Main test sequence
    // --------------------------------------------------------

    initial begin

        // Initialize control signals.
        reset = 1'b1;
        start = 1'b0;

        // Initialize input matrices.
        for (r = 0; r < 8; r = r + 1) begin
            for (c = 0; c < 8; c = c + 1) begin
                matrix_a[r][c] = 8'sd0;
                matrix_b[r][c] = 8'd0;
            end
        end

        // Hold reset across two falling clock edges.
        repeat (2) @(negedge clk);
        reset = 1'b0;

        // ====================================================
        // TEST 1: Positive matrix multiplication
        // ====================================================

        load_and_calculate(1);

        $display("Test 1 expected C[0][0] = %0d",
                 expected[0][0]);

        run_and_check(1);

        // ====================================================
        // TEST 2: Signed weights and unsigned activations
        // ====================================================

        load_and_calculate(2);

        $display("Test 2 expected C[0][0] = %0d",
                 expected[0][0]);

        run_and_check(2);

        // ====================================================
        // TEST 3: Confirm the array is idle after completion
        // ====================================================

        if (busy !== 1'b0) begin
            $fatal(1, "BUSY signal should be low after completion.");
        end

        $display("PASS: Array returned to idle.");

        // ====================================================
        // All tests completed
        // ====================================================

        $display("--------------------------------------");
        $display("ALL SYSTOLIC ARRAY TESTS PASSED!");
        $display("--------------------------------------");

        $finish;

    end

endmodule
