
`timescale 1ns / 1ps

module requantize_int8 (
    input logic clk,
    input logic reset,
    input logic start,

    input logic signed [31:0] accumulator,
    input logic signed [31:0] multiplier,
    input logic signed [7:0] shift,
    input logic signed [7:0] output_zero_point,
    input logic relu,

    output logic signed [7:0] result,
    output logic done
);

    // --------------------------------------------------------
    // FSM states
    // --------------------------------------------------------

    localparam IDLE          = 3'd0;
    localparam ROUND         = 3'd1;
    localparam SHIFT_PREP    = 3'd2;
    localparam SHIFT_CORRECT = 3'd3;
    localparam ZERO_POINT    = 3'd4;
    localparam APPLY_RELU    = 3'd5;
    localparam CLAMP         = 3'd6;

    logic [2:0] state;

    // --------------------------------------------------------
    // Captured input parameters
    // --------------------------------------------------------

    logic signed [7:0] shift_reg;
    logic signed [7:0] zero_point_reg;
    logic relu_reg;

    // Number of bits to shift right.
    // This preserves the original behavior:
    // positive shift values produce right_shift <= 0
    // and therefore do not shift the high result.
    integer right_shift;

    // --------------------------------------------------------
    // Pipeline registers
    // --------------------------------------------------------

    // Stage 1: 32-bit accumulator multiplied by 32-bit multiplier.
    logic signed [63:0] product_reg;

    // Stage 2: rounded high multiplication result.
    logic signed [63:0] high_result_reg;

    // Stage 3: shift and rounding intermediates.
    logic signed [63:0] mask_reg;
    logic signed [63:0] remainder_reg;
    logic signed [63:0] threshold_reg;
    logic signed [63:0] shifted_result_reg;

    // Stage 4 onward: final quantized value.
    logic signed [63:0] final_value_reg;

    // --------------------------------------------------------
    // Sequential requantization pipeline
    // --------------------------------------------------------

    always_ff @(posedge clk) begin

        if (reset) begin

            state <= IDLE;

            product_reg <= 0;
            high_result_reg <= 0;

            mask_reg <= 0;
            remainder_reg <= 0;
            threshold_reg <= 0;
            shifted_result_reg <= 0;

            final_value_reg <= 0;

            shift_reg <= 0;
            zero_point_reg <= 0;
            relu_reg <= 0;

            right_shift <= 0;

            result <= 0;
            done <= 0;

        end else begin

            // done is a one-clock-cycle pulse.
            done <= 0;

            case (state)

                // ============================================
                // IDLE: Capture inputs and calculate product
                // ============================================

                IDLE: begin

                    if (start) begin

                        shift_reg <= shift;
                        zero_point_reg <= output_zero_point;
                        relu_reg <= relu;

                        right_shift <= -$signed(shift);

                        // Preserve the original signed 32x32
                        // multiplication and its 64-bit result.
                        product_reg <= accumulator * multiplier;

                        state <= ROUND;

                    end

                end

                // ============================================
                // ROUND: Saturating rounding doubling high mul
                // ============================================

                ROUND: begin

                    if (product_reg >= 0) begin

                        high_result_reg <=
                            (product_reg + 64'sd1073741824) >>> 31;

                    end else begin

                        high_result_reg <=
                            (product_reg +
                             (64'sd1 - 64'sd1073741824)) >>> 31;

                    end

                    state <= SHIFT_PREP;

                end

                // ============================================
                // SHIFT_PREP: Calculate shift intermediates
                // ============================================

                SHIFT_PREP: begin

                    if (right_shift > 0) begin

                        mask_reg <=
                            (64'sd1 <<< right_shift) - 64'sd1;

                        remainder_reg <=
                            high_result_reg &
                            ((64'sd1 <<< right_shift) - 64'sd1);

                        threshold_reg <=
                            (((64'sd1 <<< right_shift) - 64'sd1) >>> 1)
                            + (high_result_reg < 0 ? 64'sd1 : 64'sd0);

                        shifted_result_reg <=
                            high_result_reg >>> right_shift;

                    end else begin

                        mask_reg <= 0;
                        remainder_reg <= 0;
                        threshold_reg <= 0;

                        shifted_result_reg <= high_result_reg;

                    end

                    state <= SHIFT_CORRECT;

                end

                // ============================================
                // SHIFT_CORRECT: Apply rounding correction
                // ============================================

                SHIFT_CORRECT: begin

                    if ((right_shift > 0) &&
                        (remainder_reg > threshold_reg)) begin

                        shifted_result_reg <=
                            shifted_result_reg + 64'sd1;

                    end

                    state <= ZERO_POINT;

                end

                // ============================================
                // ZERO_POINT: Add output zero point
                // ============================================

                ZERO_POINT: begin

                    final_value_reg <=
                        shifted_result_reg + $signed(zero_point_reg);

                    state <= APPLY_RELU;

                end

                // ============================================
                // APPLY_RELU: Apply quantized ReLU if enabled
                // ============================================

                APPLY_RELU: begin

                    if (relu_reg &&
                        (final_value_reg < $signed(zero_point_reg))) begin

                        final_value_reg <= $signed(zero_point_reg);

                    end

                    state <= CLAMP;

                end

                // ============================================
                // CLAMP: Saturate to signed INT8 range
                // ============================================

                CLAMP: begin

                    if (final_value_reg > 64'sd127) begin

                        result <= 8'sd127;

                    end else if (final_value_reg < -64'sd128) begin

                        result <= -8'sd128;

                    end else begin

                        result <= final_value_reg[7:0];

                    end

                    done <= 1;
                    state <= IDLE;

                end

                default: begin

                    state <= IDLE;

                end

            endcase

        end

    end

endmodule
