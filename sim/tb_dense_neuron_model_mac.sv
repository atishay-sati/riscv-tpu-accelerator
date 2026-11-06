
`timescale 1ns / 1ps

module tb_dense_neuron_model_mac;

    reg clk;
    reg reset;
    reg start;
    reg valid;

    reg [7:0] pixel;
    reg signed [7:0] current_weight;
    reg signed [31:0] bias;
    reg signed [31:0] expected_accumulator;

    // One-element memory arrays for loading the files.
    reg [31:0] bias_mem [0:0];
    reg [31:0] expected_accumulator_mem [0:0];

    // Outputs from the MAC module.
    wire signed [31:0] result;
    wire done;
    wire busy;

    // Memory arrays for all 784 input pixels and weights.
    reg [7:0] pixels [0:783];
    reg signed [7:0] weights [0:783];

    integer i;

    // Instantiate the model MAC module.
    dense_neuron_model_mac uut (
        .clk(clk),
        .reset(reset),
        .start(start),
        .valid(valid),
        .pixel(pixel),
        .weight(current_weight),
        .bias(bias),
        .result(result),
        .done(done),
        .busy(busy)
    );

    // 10 ns clock period.
    always #5 clk = ~clk;

    initial begin

        // Load the exported test data.
        $readmemh(
            "C:/Users/Atish/VerliogProjects/TPU_Integration/mem_files/pixels.mem",
            pixels
        );

        $readmemh(
            "C:/Users/Atish/VerliogProjects/TPU_Integration/mem_files/weights.mem",
            weights
        );

        $readmemh(
            "C:/Users/Atish/VerliogProjects/TPU_Integration/mem_files/bias.mem",
            bias_mem
        );

        $readmemh(
            "C:/Users/Atish/VerliogProjects/TPU_Integration/mem_files/expected_accumulator.mem",
            expected_accumulator_mem
        );

        // Copy loaded values into signed registers.
        bias = $signed(bias_mem[0]);
        expected_accumulator = $signed(expected_accumulator_mem[0]);

        // Initialize signals.
        clk = 0;
        reset = 1;
        start = 0;
        valid = 0;
        pixel = 0;
        current_weight = 0;

        // Release reset.
        repeat (2) @(posedge clk);
        @(negedge clk);
        reset = 0;

        // Start the MAC operation.
        start = 1;
        @(negedge clk);
        start = 0;

        // Send all 784 pixel-weight pairs.
        for (i = 0; i < 784; i = i + 1) begin
            pixel = pixels[i];
            current_weight = weights[i];
            valid = 1;

            @(negedge clk);
        end

        // Stop sending data.
        valid = 0;

        // Wait for the MAC to finish.
        wait (done == 1);
        #1;

        // Display and compare results.
        $display("--------------------------------");
        $display("MAC TEST RESULTS");
        $display("--------------------------------");
        $display("RTL result:      %0d", result);
        $display("Expected result: %0d", expected_accumulator);

        if (result === expected_accumulator) begin
            $display("PASS: RTL matches Python!");
        end
        else begin
            $display("FAIL: RTL does not match Python.");
        end

        $display("--------------------------------");

        $finish;
    end

endmodule
