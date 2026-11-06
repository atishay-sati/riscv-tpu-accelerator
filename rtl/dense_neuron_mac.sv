`timescale 1ns / 1ps

module dense_neuron_mac (
    input wire clk,
    input wire reset,
    input wire start,

    // One input pixel and one weight are supplied per valid cycle.
    input wire valid,
    input wire signed [7:0] pixel,
    input wire signed [7:0] weight,

    // Result after all 784 multiply-accumulate operations.
    output reg signed [31:0] sum,
    output reg done,
    output reg busy
);

    // 784 inputs require a 10-bit counter.
    reg [9:0] count;

    // Accumulator stores the running total.
    reg signed [31:0] accumulator;

    // INT8 * INT8 produces a signed 16-bit product.
    wire signed [15:0] product;
    assign product = pixel * weight;

    // Sign-extend the product to 32 bits before adding.
    wire signed [31:0] product_extended;
    assign product_extended = {{16{product[15]}}, product};

    always @(posedge clk) begin
        if (reset) begin
            count       <= 10'd0;
            accumulator <= 32'sd0;
            sum         <= 32'sd0;
            done        <= 1'b0;
            busy        <= 1'b0;
        end
        else begin
            // done is a one-cycle pulse.
            done <= 1'b0;

            if (start && !busy) begin
                count       <= 10'd0;
                accumulator <= 32'sd0;
                sum         <= 32'sd0;
                busy        <= 1'b1;
            end
            else if (busy && valid) begin
                if (count == 10'd783) begin
                    // Include the final product in the result.
                    sum <= accumulator + product_extended;
                    busy <= 1'b0;
                    done <= 1'b1;
                    count <= 10'd0;
                end
                else begin
                    accumulator <= accumulator + product_extended;
                    count <= count + 10'd1;
                end
            end
        end
    end

endmodule
