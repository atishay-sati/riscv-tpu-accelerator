`timescale 1ns / 1ps

module tb_tpu_top;

    // ============================================================
    // CONFIGURATION
    // ============================================================

    // Number of pixels in one 28x28 input image.
    parameter INPUT_FEATURES = 784;

    // Number of output classes produced by the TPU.
    parameter OUTPUT_CLASSES = 32;

    // ============================================================
    // CLOCK AND RESET
    // ============================================================

    logic clk;
    logic reset;
    logic start;

    // 100 MHz clock:
    // 10 ns period, with a 5 ns half-period.
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    // ============================================================
    // EXTERNAL INPUT MEMORY WRITE INTERFACE
    // ============================================================

    // These signals simulate an external device sending image
    // pixels to the TPU.
    //
    // Later, the UART receiver will drive these same signals.

    logic        input_write_enable;
    logic [9:0]  input_write_address;
    logic [7:0]  input_write_data;

    // ============================================================
    // TPU OUTPUTS
    // ============================================================

    logic [4:0] predicted_class;
    logic busy;
    logic done;

    // ============================================================
    // TEST IMAGE MEMORY
    // ============================================================

    // The testbench loads the reference image from the .mem file.
    //
    // IMPORTANT:
    // This array belongs to the testbench, NOT the TPU.
    //
    // The testbench will send these 784 pixels individually
    // through the TPU's external write interface.

    logic [7:0] input_pixels [0:INPUT_FEATURES-1];

    // ============================================================
    // EXPECTED OUTPUTS FROM TENSORFLOW LITE
    // ============================================================

    // Expected signed INT8 output score for each of the 32 classes.

    logic signed [7:0] expected_outputs [0:OUTPUT_CLASSES-1];

    // Flags indicate whether the two reference files loaded
    // successfully and contain known values.

    logic input_file_valid;
    logic expected_file_valid;

    // ============================================================
    // SCOREBOARD VARIABLES
    // ============================================================

    integer passed;
    integer failed;

    integer i;
    integer load_index;

    // Variables used to calculate the expected and actual
    // predicted classes using argmax.

    integer expected_argmax;
    integer actual_argmax;

    // Largest output score found during argmax calculation.

    integer expected_max;
    integer actual_max;

    // Tracks whether every TPU output score is a known value.
    logic all_scores_known;

    // ============================================================
    // INSTANTIATE THE COMPLETE TPU
    // ============================================================

    // LOAD_INPUT_FROM_FILE = 0:
    //
    // The TPU must NOT load its input image from input_pixels.mem.
    //
    // Instead, the testbench will write all 784 pixels through
    // input_write_enable, input_write_address, and input_write_data.
    //
    // The TPU's weights, biases, multipliers, and shifts still
    // load from their existing memory files.

    tpu_top #(
        .LOAD_INPUT_FROM_FILE(1'b0)
    ) dut (

        // Clock and reset.
        .clk(clk),
        .reset(reset),

        // Starts TPU inference.
        .start(start),

        // External input memory write interface.
        .input_write_enable(input_write_enable),
        .input_write_address(input_write_address),
        .input_write_data(input_write_data),

        // TPU outputs.
        .predicted_class(predicted_class),
        .busy(busy),
        .done(done)

    );

    // ============================================================
    // LOAD REFERENCE FILES
    // ============================================================

    initial begin

        input_file_valid = 1'b0;
        expected_file_valid = 1'b0;

        $display("========================================");
        $display("LOADING REFERENCE FILES");
        $display("========================================");

        // --------------------------------------------------------
        // LOAD THE 784 INPUT PIXELS
        // --------------------------------------------------------

        $display("Loading reference input image...");

        $readmemh(
            "mem_files/dense_layer_test/input_pixels.mem",
            input_pixels
        );

        // Check that every pixel contains a known value.
        //
        // A missing or incomplete file leaves X values in the
        // array, so the test will fail rather than continue.

        input_file_valid = 1'b1;

        for (load_index = 0;
             load_index < INPUT_FEATURES;
             load_index = load_index + 1) begin

            if (^input_pixels[load_index] === 1'bx) begin

                input_file_valid = 1'b0;

                $display(
                    "ERROR: Input pixel %0d is unknown.",
                    load_index
                );

            end

        end

        // --------------------------------------------------------
        // LOAD THE 32 EXPECTED OUTPUT SCORES
        // --------------------------------------------------------

        $display("Loading TensorFlow Lite reference outputs...");

        $readmemh(
            "mem_files/output_layer_test/expected_outputs.mem",
            expected_outputs
        );

        // Check that every expected output contains a known value.

        expected_file_valid = 1'b1;

        for (load_index = 0;
             load_index < OUTPUT_CLASSES;
             load_index = load_index + 1) begin

            if (^expected_outputs[load_index] === 1'bx) begin

                expected_file_valid = 1'b0;

                $display(
                    "ERROR: Expected output %0d is unknown.",
                    load_index
                );

            end

        end

        // --------------------------------------------------------
        // REPORT FILE STATUS
        // --------------------------------------------------------

        if (input_file_valid && expected_file_valid) begin

            $display("Input image: 784 valid pixels");
            $display("Expected outputs: 32 valid scores");
            $display("Reference files loaded successfully.");

        end

        else begin

            $display("ERROR: One or more reference files failed.");
            $display("The testbench will not run inference.");

        end

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

        // Disable external writes initially.
        input_write_enable = 1'b0;
        input_write_address = 10'd0;
        input_write_data = 8'd0;

        // Initialize scoreboard counters.
        passed = 0;
        failed = 0;

        expected_argmax = 0;
        actual_argmax = 0;

        expected_max = -1000000;
        actual_max = -1000000;

        all_scores_known = 1'b1;

        // --------------------------------------------------------
        // WAIT FOR MEMORY INITIALIZATION
        // --------------------------------------------------------

        // Give the TPU and reference-file loader time to initialize.

        repeat (5) @(negedge clk);

        // Do not continue if either reference file failed to load.

        if (!input_file_valid || !expected_file_valid) begin

            $display("========================================");
            $display("FAIL: REFERENCE FILE LOADING FAILED");
            $display("========================================");

            $fatal(1, "Cannot run TPU test with invalid reference files.");

        end

        // Release reset.
        reset = 1'b0;

        repeat (3) @(negedge clk);

        // ========================================================
        // STEP 1: WRITE ALL 784 INPUT PIXELS INTO THE TPU
        // ========================================================

        $display("========================================");
        $display("WRITING INPUT IMAGE TO TPU");
        $display("========================================");

        $display("Input features: %0d", INPUT_FEATURES);
        $display("Input source: external write interface");
        $display("");

        // Enable external writes.
        //
        // The TPU accepts writes only while its state is IDLE.
        //
        // We update the address and data on each falling clock edge.
        // The TPU captures each pixel on the following rising edge.

        input_write_enable = 1'b1;

        for (i = 0; i < INPUT_FEATURES; i = i + 1) begin

            // Wait for the falling edge before changing the signals.
            @(negedge clk);

            // Write one pixel into the TPU's input memory.
            //
            // Address 0   = first pixel
            // Address 1   = second pixel
            // ...
            // Address 783 = final pixel

            input_write_address = i;
            input_write_data = input_pixels[i];

        end

        // Wait for the final write to be captured on a rising edge.
        @(negedge clk);

        // Disable further writes.
        input_write_enable = 1'b0;

        input_write_address = 10'd0;
        input_write_data = 8'd0;

        $display("Successfully issued all 784 pixel writes.");
        $display("");

        // ========================================================
        // STEP 2: START TPU INFERENCE
        // ========================================================

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

        // ========================================================
        // STEP 3: WAIT FOR COMPLETE INFERENCE
        // ========================================================

        // Wait until the TPU asserts its done signal.

        wait (done == 1'b1);

        // Allow nonblocking assignments to settle.
        #1;

        $display("========================================");
        $display("TPU INFERENCE COMPLETED");
        $display("========================================");

        $display("Predicted class: %0d", predicted_class);
        $display("");

        // ========================================================
        // STEP 4: COMPARE ALL 32 OUTPUT SCORES
        // ========================================================

        $display("========================================");
        $display("CHECKING ALL 32 OUTPUT SCORES");
        $display("========================================");

        for (i = 0; i < OUTPUT_CLASSES; i = i + 1) begin

            // First check whether the TPU output contains X or Z.
            //
            // Reduction XOR returns X if any bit is unknown.
            // This prevents X === X from incorrectly passing.

            if ((^dut.output_scores[i] === 1'bx) ||
                (^expected_outputs[i] === 1'bx)) begin

                failed = failed + 1;
                all_scores_known = 1'b0;

                $display(
                    "FAIL class %0d: unknown output value detected.",
                    i
                );

            end

            // Compare the actual TPU score against TensorFlow Lite.
            else if (dut.output_scores[i] === expected_outputs[i]) begin

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

        // ========================================================
        // STEP 5: CALCULATE EXPECTED AND ACTUAL PREDICTED CLASSES
        // ========================================================

        expected_max = -1000000;
        expected_argmax = 0;

        actual_max = -1000000;
        actual_argmax = 0;

        for (i = 0; i < OUTPUT_CLASSES; i = i + 1) begin

            // Find the class with the largest TensorFlow Lite score.

            if ($signed(expected_outputs[i]) > expected_max) begin

                expected_max = $signed(expected_outputs[i]);
                expected_argmax = i;

            end

            // Independently find the class with the largest
            // output score produced by the TPU.

            if (all_scores_known &&
                ($signed(dut.output_scores[i]) > actual_max)) begin

                actual_max = $signed(dut.output_scores[i]);
                actual_argmax = i;

            end

        end

        // ========================================================
        // STEP 6: CHECK PREDICTED CLASS
        // ========================================================

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

        // The top-level TPU prediction and independently calculated
        // argmax must both match TensorFlow Lite.
        //
        // Also make sure predicted_class contains no unknown bits.

        if (all_scores_known &&
            (^predicted_class !== 1'bx) &&
            (predicted_class == expected_argmax) &&
            (actual_argmax == expected_argmax)) begin

            $display("PASS: PREDICTED CLASSES MATCH!");

        end

        else begin

            failed = failed + 1;

            $display("FAIL: PREDICTED CLASSES DO NOT MATCH!");

        end

        // ========================================================
        // STEP 7: FINAL RESULTS
        // ========================================================

        $display("");
        $display("========================================");
        $display("COMPLETE TPU TEST RESULTS");
        $display("========================================");

        $display("Input pixels written: %0d", INPUT_FEATURES);
        $display("Expected output scores: %0d", OUTPUT_CLASSES);
        $display("Passed comparisons: %0d", passed);
        $display("Failed comparisons: %0d", failed);

        $display("");

        if (passed == OUTPUT_CLASSES &&
            failed == 0 &&
            all_scores_known) begin

            $display("========================================");
            $display("SUCCESS: EXTERNAL INPUT TEST PASSED!");
            $display("All 784 pixels were written externally.");
            $display("All 32 output scores match TensorFlow Lite.");
            $display("========================================");

        end

        else begin

            $display("========================================");
            $display("FAIL: EXTERNAL INPUT TEST FAILED.");
            $display("Investigate the input writes and output scores.");
            $display("========================================");

        end

        $finish;

    end

    // ============================================================
    // SIMULATION TIMEOUT
    // ============================================================

    initial begin

        // Stop the simulation if inference takes longer than
        // 10 ms of simulated time.

        #10_000_000;

        $display("");
        $display("ERROR: TPU SIMULATION TIMEOUT");
        $display("The complete inference did not finish.");

        $finish;

    end

endmodule