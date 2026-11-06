`timescale 1ns / 1ps

module uart_tpu_controller (

    // ========================================================
    // CLOCK AND RESET
    // ========================================================

    input logic clk,
    input logic reset,

    // ========================================================
    // UART RECEIVER INTERFACE
    // ========================================================

    // Byte received from uart_rx.
    input logic [7:0] rx_data,

    // One-clock-cycle pulse when rx_data contains a new byte.
    input logic rx_data_valid,

    // One-clock-cycle pulse if UART detects an invalid stop bit.
    input logic rx_framing_error,

    // ========================================================
    // TPU INPUT WRITE INTERFACE
    // ========================================================

    // These three signals connect directly to the matching
    // external input-write ports on tpu_top.

    output logic       input_write_enable,
    output logic [9:0] input_write_address,
    output logic [7:0] input_write_data,

    // ========================================================
    // TPU CONTROL INTERFACE
    // ========================================================

    // One-clock-cycle pulse to start TPU inference.
    output logic tpu_start,

    // TPU asserts done when inference has completed.
    input logic tpu_done,

    // TPU's predicted class, from 0 through 31.
    input logic [4:0] predicted_class,

    // ========================================================
    // UART TRANSMITTER INTERFACE
    // ========================================================

    // Byte to send through uart_tx.
    output logic [7:0] tx_data,

    // One-clock-cycle pulse requesting transmission.
    output logic tx_start,

    // One-clock-cycle pulse when uart_tx finishes.
    input logic tx_done,

    // ========================================================
    // STATUS OUTPUTS
    // ========================================================

    // High while receiving pixels, running inference,
    // or transmitting the result.
    output logic controller_busy,

    // One-clock-cycle pulse when a complete result is transmitted.
    output logic packet_done,

    // One-clock-cycle pulse when a UART framing error occurs.
    output logic packet_error

);

    // ========================================================
    // PACKET CONFIGURATION
    // ========================================================

    // Python sends this byte before sending the image.
    localparam logic [7:0] PACKET_HEADER = 8'hAA;

    // Each image contains 28 * 28 = 784 pixels.
    localparam integer IMAGE_SIZE = 784;

    // ========================================================
    // CONTROLLER STATES
    // ========================================================

    typedef enum logic [2:0] {

        IDLE,           // Wait for packet header.
        RECEIVE_PIXELS, // Receive and write 784 pixels.
        START_TPU,      // Pulse TPU start.
        WAIT_TPU,       // Wait for inference completion.
        START_TX,       // Request result transmission.
        WAIT_TX         // Wait for UART transmission completion.

    } state_t;

    state_t state;

    // ========================================================
    // PIXEL COUNTER
    // ========================================================

    // Tracks the destination address for the next pixel.
    //
    // Valid addresses:
    // 0 through 783.
    //
    // Ten bits are needed to represent this range.

    logic [9:0] pixel_count;

    // ========================================================
    // CONTROLLER FSM
    // ========================================================

    always_ff @(posedge clk) begin

        if (reset) begin

            // Reset the controller to its idle state.
            state <= IDLE;

            // Reset the pixel counter.
            pixel_count <= 10'd0;

            // Disable all output pulses.
            input_write_enable <= 1'b0;
            input_write_address <= 10'd0;
            input_write_data <= 8'd0;

            tpu_start <= 1'b0;

            tx_data <= 8'd0;
            tx_start <= 1'b0;

            packet_done <= 1'b0;
            packet_error <= 1'b0;

        end

        else begin

            // ------------------------------------------------
            // DEFAULT OUTPUT VALUES
            // ------------------------------------------------
            // These signals are pulses. Clear them every cycle
            // unless a state explicitly asserts them.
            // ------------------------------------------------

            input_write_enable <= 1'b0;
            tpu_start <= 1'b0;
            tx_start <= 1'b0;

            packet_done <= 1'b0;
            packet_error <= 1'b0;

            // ------------------------------------------------
            // UART FRAMING ERROR HANDLING
            // ------------------------------------------------
            // If a framing error happens during packet reception,
            // abandon the incomplete packet and return to IDLE.
            // A new packet must begin with 0xAA.
            // ------------------------------------------------

            if (rx_framing_error) begin

                if (state == RECEIVE_PIXELS) begin

                    state <= IDLE;
                    pixel_count <= 10'd0;

                    packet_error <= 1'b1;

                end

            end

            else begin

                case (state)

                    // ========================================
                    // IDLE: WAIT FOR PACKET HEADER
                    // ========================================

                    IDLE: begin

                        pixel_count <= 10'd0;

                        // Ignore incoming bytes until the header
                        // byte 0xAA arrives.

                        if (rx_data_valid &&
                            (rx_data == PACKET_HEADER)) begin

                            // A new image packet has started.
                            state <= RECEIVE_PIXELS;

                        end

                    end

                    // ========================================
                    // RECEIVE_PIXELS: RECEIVE 784 PIXELS
                    // ========================================

                    RECEIVE_PIXELS: begin

                        // Only process bytes when UART RX reports
                        // that a complete byte has arrived.

                        if (rx_data_valid) begin

                            // Write the received pixel into the
                            // TPU input memory.
                            //
                            // The TPU captures this registered
                            // write pulse on the next rising edge.

                            input_write_enable <= 1'b1;

                            input_write_address <= pixel_count;

                            input_write_data <= rx_data;

                            // Check whether this is pixel 783,
                            // the final pixel in the image.

                            if (pixel_count == IMAGE_SIZE - 1) begin

                                // All 784 pixels have been received.
                                //
                                // Move to START_TPU. The final
                                // registered memory write will be
                                // captured before tpu_start is
                                // observed by the TPU.

                                state <= START_TPU;

                            end

                            else begin

                                // Advance to the next pixel address.
                                pixel_count <= pixel_count + 1'b1;

                            end

                        end

                    end

                    // ========================================
                    // START_TPU: START INFERENCE
                    // ========================================

                    START_TPU: begin

                        // Pulse start for exactly one clock cycle.
                        tpu_start <= 1'b1;

                        // Move on to waiting for inference.
                        state <= WAIT_TPU;

                    end

                    // ========================================
                    // WAIT_TPU: WAIT FOR INFERENCE
                    // ========================================

                    WAIT_TPU: begin

                        // The TPU asserts done when all 32 output
                        // scores have been calculated and its
                        // predicted_class output is ready.

                        if (tpu_done) begin

                            state <= START_TX;

                        end

                    end

                    // ========================================
                    // START_TX: SEND THE PREDICTED CLASS
                    // ========================================

                    START_TX: begin

                        // The predicted class is between 0 and 31.
                        //
                        // Send it as one unsigned byte.
                        // The upper three bits are zero.

                        tx_data <= {3'b000, predicted_class};

                        // Request one UART transmission.
                        tx_start <= 1'b1;

                        state <= WAIT_TX;

                    end

                    // ========================================
                    // WAIT_TX: WAIT FOR UART TRANSMISSION
                    // ========================================

                    WAIT_TX: begin

                        // Wait until uart_tx reports that the
                        // complete result byte was transmitted.

                        if (tx_done) begin

                            // The complete image transaction is done.
                            packet_done <= 1'b1;

                            // Return to idle for the next image.
                            state <= IDLE;

                            pixel_count <= 10'd0;

                        end

                    end

                    // ========================================
                    // DEFAULT STATE RECOVERY
                    // ========================================

                    default: begin

                        state <= IDLE;
                        pixel_count <= 10'd0;

                    end

                endcase

            end

        end

    end

    // ========================================================
    // STATUS OUTPUT
    // ========================================================

    // The controller is busy whenever it is not waiting for
    // a new packet header.

    assign controller_busy = (state != IDLE);

endmodule