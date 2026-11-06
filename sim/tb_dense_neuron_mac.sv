
`timescale 1ns / 1ps

module tb_dense_neuron_mac;

    reg clk;
    reg reset;
    reg start;
    reg valid;
    reg signed [7:0] pixel;
    reg signed [7:0] weight;

    wire signed [31:0] sum;
    wire done;
    wire busy;

    // One neuron's 784 weights and the 784 test inputs.
    reg [7:0] weights_mem [0:100351];
    reg [7:0] pixels_mem [0:783];

    integer i;

    dense_neuron_mac dut (
        .clk(clk),
        .reset(reset),
        .start(start),
        .valid(valid),
        .pixel(pixel),
        .weight(weight),
        .sum(sum),
        .done(done),
        .busy(busy)
    );

    // 100 MHz clock: 10 ns period.
    always #5 clk = ~clk;

    initial begin
        clk = 0;
        reset = 1;
        start = 0;
        valid = 0;
        pixel = 0;
        weight = 0;

        // Paths are relative to Vivado's simulation working directory.
        $readmemh("hidden_weights.mem", weights_mem);
        $readmemh("test_pixels.mem", pixels_mem);

        // Hold reset for two clock edges.
        repeat (2) @(negedge clk);
        reset = 0;

        // Start the MAC operation.
        @(negedge clk);
        start = 1;

        @(negedge clk);
        start = 0;

        // Feed neuron 0's 784 weights and the matching inputs.
        for (i = 0; i < 784; i = i + 1) begin
            pixel = $signed(pixels_mem[i]);
            weight = $signed(weights_mem[i]);
            valid = 1;

            @(negedge clk);
        end

        valid = 0;

        // Wait for the MAC to signal completion.
        wait (done == 1);

        $display("MAC result: %0d", $signed(sum));
        $display("Simulation finished.");

        #10;
        $finish;
    end

endmodule
