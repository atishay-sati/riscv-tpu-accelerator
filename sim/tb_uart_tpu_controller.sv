`timescale 1ns / 1ps

module tb_uart_tpu_controller;

    // ========================================================
    // SIMULATION CONFIGURATION
    // ========================================================

    localparam integer CLOCK_FREQ_HZ = 100_000_000;
    localparam integer TEST_BAUD_RATE = 1_000_000;
    localparam integer IMAGE_SIZE = 784;

    integer pixel_errors;
    integer packet_error_count;
    integer host_framing_error_count;

    // ========================================================
    // CLOCK AND RESET
    // ========================================================

    logic clk;
    logic reset;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    // ========================================================
    // SIMULATED COMPUTER UART TRANSMITTER
    // ========================================================

    logic [7:0] host_tx_data;
    logic host_tx_start;

    logic host_serial_tx;
    logic host_tx_busy;
    logic host_tx_done;

    // ========================================================
    // FPGA UART RECEIVER
    // ========================================================

    logic [7:0] rx_data;
    logic rx_data_valid;
    logic rx_framing_error;

    // ========================================================
    // CONTROLLER TO TPU INTERFACE
    // ========================================================

    logic input_write_enable;
    logic [9:0] input_write_address;
    logic [7:0] input_write_data;

    logic tpu_start;
    logic mock_tpu_done;
    logic [4:0] mock_predicted_class;

    // ========================================================
    // CONTROLLER TO UART TRANSMITTER INTERFACE
    // ========================================================

    logic [7:0] fpga_tx_data;
    logic fpga_tx_start;

    logic fpga_serial_tx;
    logic fpga_tx_busy;
    logic fpga_tx_done;

    // ========================================================
    // CONTROLLER STATUS
    // ========================================================

    logic controller_busy;
    logic packet_done;
    logic packet_error;

    // ========================================================
    // SIMULATED COMPUTER UART RECEIVER
    // ========================================================

    logic [7:0] host_rx_data;
    logic host_rx_valid;
    logic host_rx_framing_error;

    // ========================================================
    // MOCK TPU INPUT MEMORY
    // ========================================================

    logic [7:0] mock_input_memory [0:IMAGE_SIZE-1];

    integer writes_seen;
    integer tpu_start_count;

    // ========================================================
    // MOCK TPU CONTROL
    // ========================================================

    logic mock_tpu_busy;
    integer mock_tpu_counter;

    // ========================================================
    // EXPECTED TEST PIXEL GENERATOR
    // ========================================================

    function automatic logic [7:0] expected_pixel(
        input integer pixel_index
    );

        expected_pixel = (pixel_index * 7 + 13) % 256;

    endfunction

    // ========================================================
    // COMPUTER UART TRANSMITTER
    // ========================================================

    uart_tx #(
        .CLOCK_FREQ_HZ(CLOCK_FREQ_HZ),
        .BAUD_RATE(TEST_BAUD_RATE)
    ) host_uart_tx (

        .clk(clk),
        .reset(reset),

        .data_in(host_tx_data),
        .tx_start(host_tx_start),

        .tx(host_serial_tx),
        .tx_busy(host_tx_busy),
        .tx_done(host_tx_done)

    );

    // ========================================================
    // FPGA UART RECEIVER
    // ========================================================

    uart_rx #(
        .CLOCK_FREQ_HZ(CLOCK_FREQ_HZ),
        .BAUD_RATE(TEST_BAUD_RATE)
    ) fpga_uart_rx (

        .clk(clk),
        .reset(reset),

        .rx(host_serial_tx),

        .data_out(rx_data),
        .data_valid(rx_data_valid),
        .framing_error(rx_framing_error)

    );

    // ========================================================
    // UART-TO-TPU CONTROLLER
    // ========================================================

    uart_tpu_controller controller (

        .clk(clk),
        .reset(reset),

        .rx_data(rx_data),
        .rx_data_valid(rx_data_valid),
        .rx_framing_error(rx_framing_error),

        .input_write_enable(input_write_enable),
        .input_write_address(input_write_address),
        .input_write_data(input_write_data),

        .tpu_start(tpu_start),
        .tpu_done(mock_tpu_done),
        .predicted_class(mock_predicted_class),

        .tx_data(fpga_tx_data),
        .tx_start(fpga_tx_start),
        .tx_done(fpga_tx_done),

        .controller_busy(controller_busy),
        .packet_done(packet_done),
        .packet_error(packet_error)

    );

    // ========================================================
    // FPGA UART TRANSMITTER
    // ========================================================

    uart_tx #(
        .CLOCK_FREQ_HZ(CLOCK_FREQ_HZ),
        .BAUD_RATE(TEST_BAUD_RATE)
    ) fpga_uart_tx (

        .clk(clk),
        .reset(reset),

        .data_in(fpga_tx_data),
        .tx_start(fpga_tx_start),

        .tx(fpga_serial_tx),
        .tx_busy(fpga_tx_busy),
        .tx_done(fpga_tx_done)

    );

    // ========================================================
    // COMPUTER UART RECEIVER
    // ========================================================

    uart_rx #(
        .CLOCK_FREQ_HZ(CLOCK_FREQ_HZ),
        .BAUD_RATE(TEST_BAUD_RATE)
    ) host_uart_rx (

        .clk(clk),
        .reset(reset),

        .rx(fpga_serial_tx),

        .data_out(host_rx_data),
        .data_valid(host_rx_valid),
        .framing_error(host_rx_framing_error)

    );

    // ========================================================
    // MOCK TPU
    // ========================================================

    // Captures pixel writes, detects the start signal,
    // simulates inference, and returns class 30.

    always_ff @(posedge clk) begin

        if (reset) begin

            writes_seen <= 0;
            tpu_start_count <= 0;

            mock_tpu_busy <= 1'b0;
            mock_tpu_done <= 1'b0;

            mock_tpu_counter <= 0;

        end

        else begin

            mock_tpu_done <= 1'b0;

            // Capture incoming pixel writes.
            if (input_write_enable) begin

                if (input_write_address < IMAGE_SIZE) begin

                    mock_input_memory[input_write_address]
                        <= input_write_data;

                    writes_seen <= writes_seen + 1;

                end

            end

            // Detect TPU start.
            if (tpu_start && !mock_tpu_busy) begin

                mock_tpu_busy <= 1'b1;
                mock_tpu_counter <= 0;

                tpu_start_count <= tpu_start_count + 1;

            end

            // Simulate inference.
            else if (mock_tpu_busy) begin

                if (mock_tpu_counter == 20) begin

                    mock_tpu_busy <= 1'b0;
                    mock_tpu_done <= 1'b1;

                end

                else begin

                    mock_tpu_counter <= mock_tpu_counter + 1;

                end

            end

        end

    end

    assign mock_predicted_class = 5'd30;

    // ========================================================
    // ERROR PULSE COUNTERS
    // ========================================================

    // Count error pulses because they may disappear before
    // the main test reaches its final checks.

    always_ff @(posedge clk) begin

        if (reset) begin

            packet_error_count <= 0;
            host_framing_error_count <= 0;

        end

        else begin

            if (packet_error)
                packet_error_count <= packet_error_count + 1;

            if (host_rx_framing_error)
                host_framing_error_count <=
                    host_framing_error_count + 1;

        end

    end

    // ========================================================
    // TASK: SEND ONE UART BYTE
    // ========================================================

    task automatic send_uart_byte(
        input logic [7:0] byte_value
    );

        begin

            @(negedge clk);

            host_tx_data = byte_value;
            host_tx_start = 1'b1;

            @(negedge clk);

            host_tx_start = 1'b0;

            wait (host_tx_done == 1'b1);

            @(negedge clk);

        end

    endtask

    // ========================================================
    // MAIN TEST
    // ========================================================

    integer i;

    initial begin

        // Initialize testbench signals and counters.
        reset = 1'b1;

        host_tx_data = 8'd0;
        host_tx_start = 1'b0;

        pixel_errors = 0;

        $display("");
        $display("========================================");
        $display("UART-TO-TPU CONTROLLER TEST");
        $display("========================================");
        $display("");

        // Allow reset to propagate.
        repeat (10) @(negedge clk);

        reset = 1'b0;

        repeat (10) @(negedge clk);

        // ====================================================
        // STEP 1: SEND PACKET HEADER
        // ====================================================

        $display("STEP 1: Sending packet header 0xAA");

        send_uart_byte(8'hAA);

        $display("PASS: Header transmission completed.");
        $display("");

        // ====================================================
        // STEP 2: SEND ALL 784 PIXELS
        // ====================================================

        $display("STEP 2: Sending 784 image pixels");

        for (i = 0; i < IMAGE_SIZE; i = i + 1) begin

            send_uart_byte(expected_pixel(i));

        end

        $display("PASS: All 784 pixel bytes transmitted.");
        $display("");

        // ====================================================
        // STEP 3: WAIT FOR RESULT FROM FPGA
        // ====================================================

        $display("STEP 3: Waiting for FPGA prediction");

        wait (host_rx_valid == 1'b1);

        #1;

        $display("Received result byte: 0x%02h", host_rx_data);
        $display("Received predicted class: %0d", host_rx_data);

        if (host_rx_data == 8'd30)
            $display("PASS: Correct predicted class received.");

        else
            $display("FAIL: Incorrect predicted class received.");

        $display("");

        // ====================================================
        // STEP 4: WAIT FOR COMPLETE PACKET TRANSACTION
        // ====================================================

        wait (packet_done == 1'b1);

        #1;

        $display("STEP 4: Checking controller status");

        if (!controller_busy)
            $display("PASS: Controller returned to IDLE.");

        else
            $display("FAIL: Controller is still busy.");

        if (packet_error_count == 0)
            $display("PASS: No packet errors detected.");

        else
            $display(
                "FAIL: Detected %0d packet errors.",
                packet_error_count
            );

        if (host_framing_error_count == 0)
            $display("PASS: No UART framing errors detected.");

        else
            $display(
                "FAIL: Detected %0d UART framing errors.",
                host_framing_error_count
            );

        $display("");

        // ====================================================
        // STEP 5: CHECK ALL 784 MEMORY WRITES
        // ====================================================

        $display("STEP 5: Checking TPU input memory");

        $display("Expected writes: %0d", IMAGE_SIZE);
        $display("Actual writes:   %0d", writes_seen);

        if (writes_seen == IMAGE_SIZE)
            $display("PASS: Exactly 784 writes captured.");

        else
            $display("FAIL: Incorrect number of memory writes.");

        // Compare every stored pixel against its expected value.
        // Increment pixel_errors for every mismatch.

        for (i = 0; i < IMAGE_SIZE; i = i + 1) begin

            if (mock_input_memory[i] !== expected_pixel(i)) begin

                pixel_errors = pixel_errors + 1;

                $display(
                    "FAIL: Pixel %0d: actual=%0d expected=%0d",
                    i,
                    mock_input_memory[i],
                    expected_pixel(i)
                );

            end

        end

        if (pixel_errors == 0)
            $display("PASS: All 784 pixel values are correct.");

        else
            $display("FAIL: %0d pixel mismatches detected.", pixel_errors);

        // ====================================================
        // STEP 6: CHECK TPU START COUNT
        // ====================================================

        $display("");
        $display("STEP 6: Checking TPU start signal");

        $display("TPU start pulses: %0d", tpu_start_count);

        if (tpu_start_count == 1)
            $display("PASS: TPU started exactly once.");

        else
            $display("FAIL: Incorrect number of TPU start pulses.");

        // ====================================================
        // FINAL TEST SUMMARY
        // ====================================================

        $display("");
        $display("========================================");
        $display("UART-TO-TPU TEST SUMMARY");
        $display("========================================");

        if ((writes_seen == IMAGE_SIZE) &&
            (pixel_errors == 0) &&
            (tpu_start_count == 1) &&
            (host_rx_data == 8'd30) &&
            (!controller_busy) &&
            (packet_error_count == 0) &&
            (host_framing_error_count == 0)) begin

            $display("SUCCESS: UART-TO-TPU TEST PASSED!");
            $display("784 pixels transferred correctly.");
            $display("TPU started exactly once.");
            $display("Predicted class 30 returned to computer.");
            $display("No packet or UART framing errors.");

        end

        else begin

            $display("FAIL: UART-TO-TPU TEST FAILED.");

        end

        $display("========================================");

        $finish;

    end

    // ========================================================
    // SIMULATION TIMEOUT
    // ========================================================

    initial begin

        // Allow 20 ms for the complete test.

        #20_000_000;

        $display("");
        $display("ERROR: UART-TO-TPU TEST TIMED OUT.");

        $finish;

    end

endmodule