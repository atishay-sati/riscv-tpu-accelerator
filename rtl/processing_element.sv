
`timescale 1ns / 1ps

// ============================================================
// Module: processing_element
// Purpose:
//   Multiply two signed 8-bit inputs and accumulate their
//   products into a signed 32-bit register.
//
// This is the basic arithmetic building block of our TPU.
// Multiple PEs can eventually be connected into a systolic
// array to perform neural-network matrix multiplication.
// ============================================================

module processing_element (

    // -------------------- Control inputs --------------------

    input logic clk,
    // Clock signal.
    // The accumulator updates on each rising edge of clk.

    input logic reset,
    // Synchronous reset.
    // When reset is high at a rising clock edge, the accumulator
    // is cleared to zero.

    input logic enable,
    // Accumulation enable.
    // When enable is high, the product is added to the accumulator.
    // When enable is low, the accumulator keeps its previous value.

    // -------------------- Data inputs ------------------------

    input logic signed [7:0] a,
    // First signed 8-bit input.
    // In a neural network, this could represent an activation
    // value from the previous layer.

    input logic signed [7:0] b,
    // Second signed 8-bit input.
    // In a neural network, this could represent a weight.

    // -------------------- Data output ------------------------

    output logic signed [31:0] acc_out
    // Current accumulated result.
    // This output continuously reflects the accumulator register.
);

    // -------------------- Internal signals ------------------

    logic signed [15:0] product;
    // Holds the signed multiplication result.
    // Two signed 8-bit values can produce a result requiring
    // up to 16 bits.

    logic signed [31:0] accumulator;
    // Stores the running sum of products.
    // A wider accumulator allows multiple products to be added
    // without immediately overflowing a 16-bit result.

    // Multiply the two inputs.
    // This is combinational logic, so product updates when a or b
    // changes; it does not wait for a clock edge.
    assign product = a * b;

    // Update the accumulator only at the rising clock edge.
    always_ff @(posedge clk) begin

        if (reset) begin
            // Reset takes priority over enable.
            accumulator <= 32'sd0;
        end

        else if (enable) begin
            // Add the product to the previous accumulated sum.
            //
            // product[15] is the sign bit.
            // Replicating it 16 times sign-extends product from
            // 16 bits to 32 bits, preserving negative values.
            accumulator <= accumulator
                         + {{16{product[15]}}, product};
        end

        // If reset is low and enable is low, no assignment occurs.
        // The accumulator retains its previous value.
    end

    // Connect the internal accumulator to the module output.
    assign acc_out = accumulator;

endmodule
