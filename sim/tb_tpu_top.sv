`timescale 1ns / 1ps

module tb_tpu_top;

    // ============================================================
    // CONFIGURATION
    // ============================================================

    parameter OUTPUT_CLASSES = 32;

    // ============================================================
    // CLOCK AND RESET
    // ============================================================

    logic clk;
    logic reset;
    logic start;

    // 10 ns clock period = 100 MHz
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    // ============================================================
    // TPU OUTPUTS
    // ============================================================

    logic [4:0] predicted_class;
    logic busy;
    logic done;

    // ============================================================
    // EXPECTED OUTPUTS FROM TENSORFLOW LITE
    // ============================================================

    logic signed [7:0] expected_outputs [0:31];

    // ============================================================
    // SCOREBOARD VARIABLES
    // ============================================================

    integer passed;
    integer failed;

    integer i;

    integer expected_argmax;
    integer actual_argmax;

    integer expected_max;
    integer actual_max;

    // ============================================================
    // INSTANTIATE THE COMPLETE TPU
    // ============================================================

    tpu_top dut (

        .clk(clk),
        .reset(reset),
        .start(start),

        .predicted_class(predicted_class),
        .busy(busy),
        .done(done)

    );

    // ============================================================
    // LOAD EXPECTED OUTPUTS
    // ============================================================

    initial begin

        $display("========================================");
        $display("LOADING EXPECTED TPU OUTPUTS");
        $display("========================================");

        $readmemh(
            "mem_files/output_layer_test/expected_outputs.mem",
            expected_outputs
        );

        $display("Loaded all 32 expected output values.");
        $display("");

    end

    // ============================================================
    // MAIN TEST
    // ============================================================

    initial begin

        // --------------------------------------------------------
        // INITIALIZE INPUTS
        // --------------------------------------------------------

        reset = 1'b1;
        start = 1'b0;

        passed = 0;
        failed = 0;

        expected_argmax = 0;
        actual_argmax = 0;

        expected_max = -1000000;
        actual_max = -1000000;

        // --------------------------------------------------------
        // WAIT FOR MEMORY INITIALIZATION
        // --------------------------------------------------------

        repeat (5) @(negedge clk);

        // Release reset.
        reset = 1'b0;

        repeat (3) @(negedge clk);

        // --------------------------------------------------------
        // START TPU INFERENCE
        // --------------------------------------------------------

        $display("========================================");
        $display("STARTING COMPLETE TPU INFERENCE");
        $display("========================================");

        $display("Input features: 784");
        $display("Hidden neurons: 128");
        $display("Output classes: 32");
        $display("");

        // Assert start for exactly one clock cycle.
        start = 1'b1;

        @(negedge clk);

        start = 1'b0;

        $display("TPU started.");
        $display("Waiting for inference to finish...");
        $display("");

        // --------------------------------------------------------
        // WAIT FOR COMPLETE INFERENCE
        // --------------------------------------------------------

        wait (done == 1'b1);

        // Allow nonblocking assignments to settle.
        #1;

        $display("========================================");
        $display("TPU INFERENCE COMPLETED");
        $display("========================================");

        $display("Predicted class: %0d", predicted_class);
        $display("");

        // --------------------------------------------------------
        // COMPARE ALL 32 OUTPUT SCORES
        // --------------------------------------------------------

        $display("========================================");
        $display("CHECKING ALL 32 OUTPUT SCORES");
        $display("========================================");

        for (i = 0; i < OUTPUT_CLASSES; i = i + 1) begin

            // Read the output score directly from the top-level
            // module's internal output_scores memory.

            if (dut.output_scores[i] === expected_outputs[i]) begin

                passed = passed + 1;

                $display(
                    "PASS class %0d: actual=%0d expected=%0d",
                    i,
                    $signed(dut.output_scores[i]),
                    $signed(expected_outputs[i])
                );

            end

            else begin

                failed = failed + 1;

                $display(
                    "FAIL class %0d: actual=%0d expected=%0d",
                    i,
                    $signed(dut.output_scores[i]),
                    $signed(expected_outputs[i])
                );

            end

        end

        // --------------------------------------------------------
        // CALCULATE EXPECTED PREDICTED CLASS
        // --------------------------------------------------------

        expected_max = -1000000;
        expected_argmax = 0;

        actual_max = -1000000;
        actual_argmax = 0;

        for (i = 0; i < OUTPUT_CLASSES; i = i + 1) begin

            if ($signed(expected_outputs[i]) > expected_max) begin

                expected_max = $signed(expected_outputs[i]);
                expected_argmax = i;

            end

            if ($signed(dut.output_scores[i]) > actual_max) begin

                actual_max = $signed(dut.output_scores[i]);
                actual_argmax = i;

            end

        end

        // --------------------------------------------------------
        // CHECK PREDICTED CLASS
        // --------------------------------------------------------

        $display("");
        $display("========================================");
        $display("PREDICTION COMPARISON");
        $display("========================================");

        $display(
            "TensorFlow Lite predicted class: %0d",
            expected_argmax
        );

        $display(
            "TPU predicted class: %0d",
            predicted_class
        );

        if (predicted_class == expected_argmax &&
            actual_argmax == expected_argmax) begin

            $display("PASS: PREDICTED CLASSES MATCH!");

        end

        else begin

            failed = failed + 1;

            $display("FAIL: PREDICTED CLASSES DO NOT MATCH!");

        end

        // --------------------------------------------------------
        // FINAL RESULTS
        // --------------------------------------------------------

        $display("");
        $display("========================================");
        $display("COMPLETE TPU TEST RESULTS");
        $display("========================================");

        $display("Expected output scores: %0d", OUTPUT_CLASSES);
        $display("Passed comparisons: %0d", passed);
        $display("Failed comparisons: %0d", failed);

        $display("");

        if (passed == OUTPUT_CLASSES && failed == 0) begin

            $display("========================================");
            $display("SUCCESS: COMPLETE TPU VERIFIED!");
            $display("All 32 output scores match TensorFlow Lite.");
            $display("========================================");

        end

        else begin

            $display("========================================");
            $display("FAIL: COMPLETE TPU TEST FAILED.");
            $display("Investigate the mismatched output scores.");
            $display("========================================");

        end

        $finish;

    end

    // ============================================================
    // SIMULATION TIMEOUT
    // ============================================================

    initial begin

        // 10 ms simulated-time timeout.
        #10_000_000;

        $display("");
        $display("ERROR: TPU SIMULATION TIMEOUT");
        $display("The complete inference did not finish.");

        $finish;

    end

endmodule
