
`timescale 1ns / 1ps

// ============================================================
// Module: systolic_pe
//
// Purpose:
//   Configurable processing element for a systolic array.
//
// Arithmetic:
//   A = signed weight
//   B = unsigned activation
//   Product = signed weight * unsigned activation
//   Accumulator = signed running sum
//
// Parameters:
//   DATA_WIDTH = Width of weights and activations.
//   ACC_WIDTH  = Width of the accumulator.
//
// Data movement:
//   A moves horizontally.
//   B moves vertically.
//
// DSP INFERENCE:
//   The use_dsp attribute encourages Vivado to implement
//   the multiplication using a DSP48E1 slice instead of LUTs.
// ============================================================
module systolic_pe #(

    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32

)(

    // --------------------------------------------------------
    // Clock and control
    // --------------------------------------------------------

    input logic clk,
    input logic reset,
    input logic enable,
    input logic clear_acc,

    // --------------------------------------------------------
    // Data inputs
    // --------------------------------------------------------

    input logic signed [DATA_WIDTH-1:0] a_in,
    // Signed weight input.

    input logic [DATA_WIDTH-1:0] b_in,
    // Unsigned activation input.

    // --------------------------------------------------------
    // Data outputs
    // --------------------------------------------------------

    output logic signed [DATA_WIDTH-1:0] a_out,
    // Forwarded signed weight.

    output logic [DATA_WIDTH-1:0] b_out,
    // Forwarded unsigned activation.

    output logic signed [ACC_WIDTH-1:0] acc_out
    // Current accumulated result.

);

    // ========================================================
    // 1. Internal signals
    // ========================================================

    // Extend the unsigned activation by one bit.
    // This allows the unsigned value to be represented
    // correctly as a positive signed number.

    logic signed [DATA_WIDTH:0] activation_extended;

    // Full multiplication result.
    // A signed DATA_WIDTH-bit weight multiplied by a
    // signed (DATA_WIDTH+1)-bit activation requires
    // 2*DATA_WIDTH+1 result bits.

    // IMPORTANT:
    // Encourage Vivado to map this multiplication into
    // a DSP48E1 slice rather than LUT logic.

    (* use_dsp = "yes" *)
    logic signed [(2*DATA_WIDTH):0] product;

    // Extend the multiplication result to accumulator width.

    logic signed [ACC_WIDTH-1:0] product_extended;

    // Internal running accumulator.

    logic signed [ACC_WIDTH-1:0] accumulator;


    // ========================================================
    // 2. Combinational arithmetic
    // ========================================================

    // Convert the unsigned activation into a positive
    // signed number without changing its numerical value.

    assign activation_extended = {
        1'b0,
        b_in
    };

    // Signed weight multiplied by unsigned activation.
    // Both operands are interpreted as signed here.

    assign product = a_in * activation_extended;

    // Sign-extend the product before accumulation.
    // This preserves negative multiplication results.

    assign product_extended = {
        {(ACC_WIDTH-(2*DATA_WIDTH+1)){product[(2*DATA_WIDTH)]}},
        product
    };

    // Expose the internal accumulator.

    assign acc_out = accumulator;


    // ========================================================
    // 3. Sequential operation
    // ========================================================

    always_ff @(posedge clk) begin

        if (reset) begin

            // Reset the accumulated result.

            accumulator <= '0;

            // Clear forwarded data.

            a_out <= '0;
            b_out <= '0;

        end

        else begin

            // ------------------------------------------------
            // Clear accumulator
            // ------------------------------------------------

            if (clear_acc) begin

                accumulator <= '0;

            end

            // ------------------------------------------------
            // Multiply and accumulate
            // ------------------------------------------------

            else if (enable) begin

                accumulator <= accumulator
                             + product_extended;

            end

            // ------------------------------------------------
            // Forward data through the systolic array
            // ------------------------------------------------

            if (enable) begin

                a_out <= a_in;
                b_out <= b_in;

            end

        end

    end

endmodule
