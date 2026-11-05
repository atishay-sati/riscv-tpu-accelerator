`timescale 1ns / 1ps

module requantize_int8 (
    input wire signed [31:0] accumulator,

    // Fixed-point quantization parameters.
    input wire signed [31:0] multiplier,
    input wire signed [7:0] shift,
    input wire signed [7:0] output_zero_point,

    // Enable ReLU for layers that use it.
    input wire relu,

    // Final quantized INT8 activation.
    output reg signed [7:0] result
);

    // Intermediate values for fixed-point multiplication.
    reg signed [63:0] product;
    reg signed [63:0] nudge;
    reg signed [63:0] high_result;
    reg signed [63:0] shifted_result;

    reg signed [63:0] mask;
    reg signed [63:0] remainder;
    reg signed [63:0] threshold;

    reg signed [63:0] final_value;

    integer right_shift;

    always @(*) begin

        // ----------------------------------------
        // 1. Saturating Rounding Doubling High Mul
        // ----------------------------------------

        product = accumulator * multiplier;

        if (product >= 0)
            nudge = 64'sd1073741824;
        else
            nudge = 64'sd1 - 64'sd1073741824;

        high_result = (product + nudge) >>> 31;

        // ----------------------------------------
        // 2. Apply the right shift
        // ----------------------------------------

        right_shift = -shift;

        if (right_shift > 0) begin

            mask = (64'sd1 <<< right_shift) - 64'sd1;
            remainder = high_result & mask;

            threshold = (mask >>> 1);

            if (high_result < 0)
                threshold = threshold + 64'sd1;

            shifted_result = high_result >>> right_shift;

            if (remainder > threshold)
                shifted_result = shifted_result + 64'sd1;

        end
        else begin
            shifted_result = high_result;
        end

        // ----------------------------------------
        // 3. Add output zero point
        // ----------------------------------------

        final_value = shifted_result + output_zero_point;

        // ----------------------------------------
        // 4. Apply quantized ReLU
        // ----------------------------------------

        // ReLU's quantized minimum is the output
        // zero point, not necessarily integer zero.
        if (relu && final_value < output_zero_point)
            final_value = output_zero_point;

        // ----------------------------------------
        // 5. Clamp to INT8 range
        // ----------------------------------------

        if (final_value > 127)
            result = 8'sd127;
        else if (final_value < -128)
            result = -8'sd128;
        else
            result = final_value[7:0];

    end

endmodule