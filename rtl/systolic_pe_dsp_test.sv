
`timescale 1ns / 1ps

// ============================================================
// Module: systolic_pe_dsp_test
//
// Purpose:
//   Test DSP inference with the complete PE datapath.
//
// Includes:
//   1. Signed weight x unsigned activation multiplication.
//   2. 32-bit accumulator.
//   3. Accumulator reset and clear controls.
//   4. Horizontal weight forwarding (a_out).
//   5. Vertical activation forwarding (b_out).
// ============================================================

module systolic_pe_dsp_test (
    input logic clk,
    input logic reset,
    input logic enable,
    input logic clear_acc,

    // Signed INT8 weight input.
    input logic signed [7:0] a_in,

    // Unsigned INT8 activation input.
    input logic [7:0] b_in,

    // Forwarded weight for the next PE.
    output logic signed [7:0] a_out,

    // Forwarded activation for the next PE.
    output logic [7:0] b_out,

    // Signed 32-bit accumulated result.
    output logic signed [31:0] acc_out
);

    // Convert unsigned activation into positive signed 9-bit.
    logic signed [8:0] activation_extended;

    // Signed 17-bit multiplication result.
    (* use_dsp = "yes" *)
    logic signed [16:0] product;

    // Sign-extend multiplication result to 32 bits.
    logic signed [31:0] product_extended;

    // Running accumulator.
    logic signed [31:0] accumulator;

    assign activation_extended = {1'b0, b_in};
    assign product = a_in * activation_extended;

    assign product_extended = {
        {15{product[16]}},
        product
    };

    assign acc_out = accumulator;

    always_ff @(posedge clk) begin
        if (reset) begin
            accumulator <= '0;
            a_out       <= '0;
            b_out       <= '0;
        end
        else begin

            // Accumulator control.
            if (clear_acc) begin
                accumulator <= '0;
            end
            else if (enable) begin
                accumulator <= accumulator + product_extended;
            end

            // Forward weight and activation to neighboring PEs.
            if (enable) begin
                a_out <= a_in;
                b_out <= b_in;
            end

        end
    end

endmodule
