`timescale 1ns / 1ps

// ============================================================
// Module: matrix_engine
//
// Purpose:
//   Multiply two 2x2 matrices using four processing elements.
//
// Matrix A:
//       | a00  a01 |
//       | a10  a11 |
//
// Matrix B:
//       | b00  b01 |
//       | b10  b11 |
//
// Result C = A x B:
//
//       | c00  c01 |
//       | c10  c11 |
//
// The output equations are:
//
//   c00 = a00*b00 + a01*b10
//   c01 = a00*b01 + a01*b11
//   c10 = a10*b00 + a11*b10
//   c11 = a10*b01 + a11*b11
//
// IMPORTANT:
//   This is a simple matrix multiplication engine, not yet a
//   true systolic array. Data is selected centrally and is not
//   passed between neighboring PEs.
// ============================================================

module matrix_engine (

    // -------------------- Control inputs --------------------

    input logic clk,
    // Clock shared by the controller and all four PEs.
    // State changes and PE accumulation happen on rising edges.

    input logic reset,
    // Synchronous reset for the controller.
    // Also clears all four PE accumulators.

    input logic start,
    // Requests a new matrix multiplication.
    // Assert start while the controller is in IDLE.
    // Keep all matrix inputs stable during computation.

    // -------------------- Matrix A inputs -------------------

    input logic signed [7:0] a00,
    // Element at row 0, column 0 of matrix A.

    input logic signed [7:0] a01,
    // Element at row 0, column 1 of matrix A.

    input logic signed [7:0] a10,
    // Element at row 1, column 0 of matrix A.

    input logic signed [7:0] a11,
    // Element at row 1, column 1 of matrix A.

    // -------------------- Matrix B inputs -------------------

    input logic signed [7:0] b00,
    // Element at row 0, column 0 of matrix B.

    input logic signed [7:0] b01,
    // Element at row 0, column 1 of matrix B.

    input logic signed [7:0] b10,
    // Element at row 1, column 0 of matrix B.

    input logic signed [7:0] b11,
    // Element at row 1, column 1 of matrix B.

    // -------------------- Matrix C outputs ------------------

    output logic signed [31:0] c00,
    // Element at row 0, column 0 of the result.

    output logic signed [31:0] c01,
    // Element at row 0, column 1 of the result.

    output logic signed [31:0] c10,
    // Element at row 1, column 0 of the result.

    output logic signed [31:0] c11,
    // Element at row 1, column 1 of the result.

    output logic done
    // High while the controller is in DONE.
    // Indicates that all four matrix outputs are ready.
);

    // -------------------- Controller states -----------------

    // Four possible states, encoded using two bits.
    typedef enum logic [1:0] {
        IDLE,   // Wait for a start request.
        MAC0,   // Accumulate the first product.
        MAC1,   // Accumulate the second product.
        DONE    // Signal that the result is ready.
    } state_t;

    state_t state;
    // Stores the current controller state.

    // -------------------- PE control signals ----------------

    logic pe_enable;
    // Enables all four PEs during MAC0 and MAC1.
    // Each enabled rising edge adds one product to its accumulator.

    logic pe_reset;
    // Clears all four accumulators during external reset or
    // when a new request starts from IDLE.

    // -------------------- Selected PE inputs ----------------

    logic signed [7:0] x00, x01, x10, x11;
    // First operands selected for PEs 00, 01, 10, and 11.

    logic signed [7:0] y00, y01, y10, y11;
    // Second operands selected for PEs 00, 01, 10, and 11.

    // Enable accumulation only during the two MAC states.
    assign pe_enable = (state == MAC0) || (state == MAC1);

    // Clear the PE accumulators when reset is asserted or
    // when a new multiplication request is accepted.
    assign pe_reset = reset || ((state == IDLE) && start);

    // -------------------- Operand selection -----------------

    // Select which matrix elements each PE multiplies.
    // This block is combinational: the selected values change
    // when the state or matrix inputs change.

    always_comb begin

        // Default selection: first product of each dot product.
        //
        // PE00 calculates the first part of c00.
        // PE01 calculates the first part of c01.
        // PE10 calculates the first part of c10.
        // PE11 calculates the first part of c11.

        x00 = a00;
        y00 = b00;

        x01 = a00;
        y01 = b01;

        x10 = a10;
        y10 = b00;

        x11 = a10;
        y11 = b01;

        // In MAC1, select the second product for each output.
        if (state == MAC1) begin

            // c00 adds a01*b10.
            x00 = a01;
            y00 = b10;

            // c01 adds a01*b11.
            x01 = a01;
            y01 = b11;

            // c10 adds a11*b10.
            x10 = a11;
            y10 = b10;

            // c11 adds a11*b11.
            x11 = a11;
            y11 = b11;

        end
    end

    // -------------------- Processing elements ---------------

    // Each PE contains its own accumulator.
    // All four PEs operate in parallel.

    // PE00:
    // c00 = a00*b00 + a01*b10
    processing_element pe00 (
        .clk(clk),
        .reset(pe_reset),
        .enable(pe_enable),
        .a(x00),
        .b(y00),
        .acc_out(c00)
    );

    // PE01:
    // c01 = a00*b01 + a01*b11
    processing_element pe01 (
        .clk(clk),
        .reset(pe_reset),
        .enable(pe_enable),
        .a(x01),
        .b(y01),
        .acc_out(c01)
    );

    // PE10:
    // c10 = a10*b00 + a11*b10
    processing_element pe10 (
        .clk(clk),
        .reset(pe_reset),
        .enable(pe_enable),
        .a(x10),
        .b(y10),
        .acc_out(c10)
    );

    // PE11:
    // c11 = a10*b01 + a11*b11
    processing_element pe11 (
        .clk(clk),
        .reset(pe_reset),
        .enable(pe_enable),
        .a(x11),
        .b(y11),
        .acc_out(c11)
    );

    // -------------------- Controller ------------------------

    // Advance through IDLE -> MAC0 -> MAC1 -> DONE.
    // The controller changes state on each rising clock edge.

    always_ff @(posedge clk) begin

        if (reset) begin
            // Return to IDLE when external reset is high.
            state <= IDLE;
        end
        else begin

            case (state)

                IDLE: begin
                    // Wait for a new matrix multiplication request.
                    if (start)
                        state <= MAC0;
                end

                MAC0: begin
                    // The first product is accumulated at this edge.
                    // Next cycle selects the second product.
                    state <= MAC1;
                end

                MAC1: begin
                    // The second product is accumulated at this edge.
                    // All four matrix results are now complete.
                    state <= DONE;
                end

                DONE: begin
                    // Return to IDLE to allow another request.
                    state <= IDLE;
                end

                default: begin
                    // Recover to IDLE if the state is invalid.
                    state <= IDLE;
                end

            endcase
        end
    end

    // Assert done during the DONE state.
    assign done = (state == DONE);

endmodule