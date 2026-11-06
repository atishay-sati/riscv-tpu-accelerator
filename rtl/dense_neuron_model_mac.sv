`timescale 1ns / 1ps

module dense_neuron_model_mac (
    input wire clk,
    input wire reset,
    input wire start,

    // One adjusted pixel and one weight per valid cycle.
    input wire valid,
    input wire [7:0] pixel,
    input wire signed [7:0] weight,

    // Bias for this particular neuron.
    input wire signed [31:0] bias,

    // Final accumulator, including bias.
    output reg signed [31:0] result,
    output reg done,
    output reg busy
);

    // 784 input pixels require a 10-bit counter.
    reg [9:0] count;

    // Stores the running sum of multiplication results.
    reg signed [31:0] accumulator;

    // Convert unsigned pixel into a positive signed value.
    wire signed [8:0] pixel_signed;
    assign pixel_signed = {1'b0, pixel};

    // The product fits in a signed 16-bit value:
    // 255 * -128 = -32640
    // 255 *  127 =  32385
    wire signed [15:0] product;
    assign product = pixel_signed * weight;

    // Extend the product to 32 bits before accumulation.
    wire signed [31:0] product_extended;
    assign product_extended = {
        {16{product[15]}}, product
    };

    always @(posedge clk) begin
        if (reset) begin
            count       <= 10'd0;
            accumulator <= 32'sd0;
            result      <= 32'sd0;
            done        <= 1'b0;
            busy        <= 1'b0;
        end
        else begin
            // Done is a one-clock-cycle pulse.
            done <= 1'b0;

            if (start && !busy) begin
                count       <= 10'd0;
                accumulator <= 32'sd0;
                result      <= 32'sd0;
                busy        <= 1'b1;
            end

            else if (busy && valid) begin

                // Process the final pixel.
                if (count == 10'd783) begin

                    // Include the final product and bias.
                    result <= accumulator
                            + product_extended
                            + bias;

                    busy  <= 1'b0;
                    done  <= 1'b1;
                    count <= 10'd0;
                end

                else begin
                    accumulator <= accumulator
                                 + product_extended;

                    count <= count + 10'd1;
                end
            end
        end
    end

endmodule
