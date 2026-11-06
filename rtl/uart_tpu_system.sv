
`timescale 1ns / 1ps

module uart_tpu_system #(

    // ========================================================
    // UART CONFIGURATION
    // ========================================================

    // Basys 3 clock frequency.
    parameter integer CLOCK_FREQ_HZ = 100_000_000,

    // Actual hardware baud rate.
    parameter integer BAUD_RATE = 115200,

    // ========================================================
    // TPU CONFIGURATION
    // ========================================================

    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32,
    parameter ADDR_WIDTH = 32,

    // --------------------------------------------------------
    // MODEL MEMORY FILE LOCATIONS
    // --------------------------------------------------------

    parameter HIDDEN_WEIGHTS_FILE =
        "mem_files/dense_layer_test/hidden_weights.mem",

    parameter HIDDEN_BIASES_FILE =
        "mem_files/dense_layer_test/hidden_biases.mem",

    parameter HIDDEN_MULTIPLIERS_FILE =
        "mem_files/dense_layer_test/hidden_multipliers.mem",

    parameter HIDDEN_SHIFTS_FILE =
        "mem_files/dense_layer_test/hidden_shifts.mem",

    parameter INPUT_PIXELS_FILE =
        "mem_files/dense_layer_test/input_pixels.mem",

    parameter OUTPUT_WEIGHTS_FILE =
        "mem_files/output_layer_test/output_weights.mem",

    parameter OUTPUT_BIASES_FILE =
        "mem_files/output_layer_test/output_biases.mem",

    parameter OUTPUT_MULTIPLIERS_FILE =
        "mem_files/output_layer_test/output_multipliers.mem",

    parameter OUTPUT_SHIFTS_FILE =
        "mem_files/output_layer_test/output_shifts.mem"

)(

    // ========================================================
    // PHYSICAL FPGA INTERFACE
    // ========================================================

    input logic clk,
    input logic reset,

    // Connect this input to the Basys 3 USB-UART receive pin.
    // The computer sends the image through this signal.
    input logic uart_rx_pin,

    // Connect this output to the Basys 3 USB-UART transmit pin.
    // The FPGA sends the predicted class through this signal.
    output logic uart_tx_pin,

    // ========================================================
    // SYSTEM STATUS
    // ========================================================

    // High while receiving an image, running inference,
    // or transmitting the prediction.
    output logic controller_busy,

    // High while the TPU is performing inference.
    output logic tpu_busy,

    // One-cycle pulse after the result has been transmitted.
    output logic packet_done,

    // One-cycle pulse if a packet framing error is detected.
    output logic packet_error,

    // Predicted class from 0 through 31.
    output logic [4:0] predicted_class



);
    // ========================================================
    // UART RECEIVER SIGNALS
    // ========================================================

    logic [7:0] rx_data;
    logic rx_data_valid;
    logic rx_framing_error;

    // ========================================================
    // CONTROLLER-TO-TPU INPUT WRITE SIGNALS
    // ========================================================

    logic input_write_enable;
    logic [9:0] input_write_address;
    logic [7:0] input_write_data;

    // ========================================================
    // CONTROLLER-TO-TPU CONTROL SIGNALS
    // ========================================================

    logic tpu_start;
    logic tpu_done;

    // ========================================================
    // CONTROLLER-TO-UART TRANSMITTER SIGNALS
    // ========================================================

    logic [7:0] tx_data;
    logic tx_start;
    logic tx_done;

    // ========================================================
    // UART RECEIVER
    // ========================================================

    // Receives bytes sent by the computer.
    //
    // The receiver:
    // 1. Monitors uart_rx_pin.
    // 2. Decodes incoming serial data.
    // 3. Produces rx_data_valid for each received byte.
    // 4. Reports invalid stop bits through rx_framing_error.

    uart_rx #(
        .CLOCK_FREQ_HZ(CLOCK_FREQ_HZ),
        .BAUD_RATE(BAUD_RATE)
    ) uart_receiver (

        .clk(clk),
        .reset(reset),

        .rx(uart_rx_pin),

        .data_out(rx_data),
        .data_valid(rx_data_valid),
        .framing_error(rx_framing_error)

    );

    // ========================================================
    // UART-TO-TPU CONTROLLER
    // ========================================================

    // Controls the complete image classification transaction.
    //
    // 1. Waits for header 0xAA.
    // 2. Receives 784 pixel bytes.
    // 3. Writes pixels into TPU input memory.
    // 4. Starts TPU inference.
    // 5. Waits for TPU completion.
    // 6. Requests transmission of the predicted class.
    // 7. Reports packet completion.

    uart_tpu_controller controller (

        .clk(clk),
        .reset(reset),

        // UART receiver interface.
        .rx_data(rx_data),
        .rx_data_valid(rx_data_valid),
        .rx_framing_error(rx_framing_error),

        // External TPU input-memory write interface.
        .input_write_enable(input_write_enable),
        .input_write_address(input_write_address),
        .input_write_data(input_write_data),

        // TPU control interface.
        .tpu_start(tpu_start),
        .tpu_done(tpu_done),
        .predicted_class(predicted_class),

        // UART transmitter interface.
        .tx_data(tx_data),
        .tx_start(tx_start),
        .tx_done(tx_done),

        // System status.
        .controller_busy(controller_busy),
        .packet_done(packet_done),
        .packet_error(packet_error)

    );

    // ========================================================
    // ACTUAL TPU
    // ========================================================

    // This is your existing, fully implemented INT8 TPU.
    //
    // LOAD_INPUT_FROM_FILE = 0 is important:
    // The image must come from UART, not input_pixels.mem.
    //
    // All model weights, biases, multipliers, and shifts
    // are still loaded from their existing memory files.

    tpu_top #(

        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),

        .HIDDEN_WEIGHTS_FILE(HIDDEN_WEIGHTS_FILE),
        .HIDDEN_BIASES_FILE(HIDDEN_BIASES_FILE),
        .HIDDEN_MULTIPLIERS_FILE(HIDDEN_MULTIPLIERS_FILE),
        .HIDDEN_SHIFTS_FILE(HIDDEN_SHIFTS_FILE),

        .INPUT_PIXELS_FILE(INPUT_PIXELS_FILE),

        .LOAD_INPUT_FROM_FILE(1'b0),

        .OUTPUT_WEIGHTS_FILE(OUTPUT_WEIGHTS_FILE),
        .OUTPUT_BIASES_FILE(OUTPUT_BIASES_FILE),
        .OUTPUT_MULTIPLIERS_FILE(OUTPUT_MULTIPLIERS_FILE),
        .OUTPUT_SHIFTS_FILE(OUTPUT_SHIFTS_FILE)

    ) tpu (

        .clk(clk),
        .reset(reset),

        // Start inference when the UART controller requests it.
        .start(tpu_start),

        // Receive the 784 pixels from the UART controller.
        .input_write_enable(input_write_enable),
        .input_write_address(input_write_address),
        .input_write_data(input_write_data),

        // Return the predicted class to the controller.
        .predicted_class(predicted_class),

        // Inference status.
        .busy(tpu_busy),
        .done(tpu_done)

    );

    // ========================================================
    // UART TRANSMITTER
    // ========================================================

    // Sends the predicted class back to the computer.
    //
    // The controller places the class in tx_data and pulses
    // tx_start. The transmitter sends one complete UART byte.
    //
    // The upper three bits of tx_data are always zero because
    // the predicted class is between 0 and 31.

    uart_tx #(
        .CLOCK_FREQ_HZ(CLOCK_FREQ_HZ),
        .BAUD_RATE(BAUD_RATE)
    ) uart_transmitter (

        .clk(clk),
        .reset(reset),

        .data_in(tx_data),
        .tx_start(tx_start),

        .tx(uart_tx_pin),
        .tx_busy(),
        .tx_done(tx_done)

    );

        // ========================================================
    // ILA DEBUG CORE
    // ========================================================

    // Monitors the UART-to-TPU input path and TPU completion.
    // This lets us inspect what physically happens on the FPGA.

    ila_tpu_debug ila_debug (

        .clk(clk),

        // Pixel writes into TPU input memory.
        // probe0 combines write enable (1 bit) and address (10 bits).
        .probe0({input_write_enable, input_write_address}),
        .probe1(input_write_data),

        // TPU inference control and completion.
        .probe2(tpu_start),
        .probe3(tpu_done),

        // Final predicted class.
        .probe4(predicted_class)

    );

endmodule
