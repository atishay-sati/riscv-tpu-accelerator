`timescale 1ns / 1ps

module tb_uart_loopback;

    // =====================================================
    // TESTBENCH SIGNALS
    // =====================================================

    // Simulated 100 MHz FPGA clock.
    logic clk = 0;

    // Active-high reset.
    logic reset;

    // Signals going into the UART transmitter.
    logic [7:0] tx_data;
    logic tx_start;

    // Signals coming out of the UART transmitter.
    logic tx_serial;
    logic tx_busy;
    logic tx_done;

    // Signals coming out of the UART receiver.
    logic [7:0] rx_data;
    logic rx_valid;
    logic framing_error;

    // =====================================================
    // CLOCK GENERATION
    // =====================================================

    // 100 MHz clock:
    // 10 ns period = 100 MHz.
    always #5 clk = ~clk;

    // =====================================================
    // UART TRANSMITTER
    // =====================================================

    uart_tx #(
        .CLOCK_FREQ_HZ(100_000_000),
        .BAUD_RATE(115200)
    ) transmitter (
        .clk(clk),
        .reset(reset),
        .data_in(tx_data),
        .tx_start(tx_start),
        .tx(tx_serial),
        .tx_busy(tx_busy),
        .tx_done(tx_done)
    );

    // =====================================================
    // UART RECEIVER
    // =====================================================

    uart_rx #(
        .CLOCK_FREQ_HZ(100_000_000),
        .BAUD_RATE(115200)
    ) receiver (
        .clk(clk),
        .reset(reset),
        .rx(tx_serial),
        .data_out(rx_data),
        .data_valid(rx_valid),
        .framing_error(framing_error)
    );

    // =====================================================
    // TASK: TRANSMIT ONE BYTE AND CHECK THE RESULT
    // =====================================================

    task automatic send_and_check(input logic [7:0] test_byte);

        begin

            // Wait for a clock edge before starting.
            @(negedge clk);

            tx_data = test_byte;
            tx_start = 1'b1;

            // Keep tx_start HIGH for exactly one clock cycle.
            @(negedge clk);
            tx_start = 1'b0;

            // Wait until the receiver reports valid data.
            wait (rx_valid == 1'b1);

            // Compare received data against transmitted data.
            if (rx_data !== test_byte) begin

                $display("FAIL: Sent %h, received %h",
                         test_byte, rx_data);

                $fatal(1);

            end

            // Check that no framing error occurred.
            if (framing_error !== 1'b0) begin

                $display("FAIL: Framing error detected!");

                $fatal(1);

            end

            // Wait until the transmitter finishes.
            wait (tx_done == 1'b1);

            $display("PASS: Sent %h, received %h",
                     test_byte, rx_data);

            // Allow signals to settle before next test.
            @(negedge clk);

        end

    endtask

    // =====================================================
    // MAIN TEST SEQUENCE
    // =====================================================

    initial begin

        $display("======================================");
        $display("       UART LOOPBACK TEST");
        $display("======================================");

        // Initialize input signals.
        reset = 1'b1;
        tx_data = 8'h00;
        tx_start = 1'b0;

        // Hold reset for several clock cycles.
        repeat (10) @(negedge clk);

        reset = 1'b0;

        // Allow the modules to leave reset.
        repeat (5) @(negedge clk);

        // Test different byte patterns.
        send_and_check(8'hA5);
        send_and_check(8'h3C);
        send_and_check(8'h00);
        send_and_check(8'hFF);

        // All tests passed.
        $display("======================================");
        $display("SUCCESS: ALL UART TESTS PASSED!");
        $display("======================================");

        $finish;

    end

    // =====================================================
    // SIMULATION TIMEOUT
    // =====================================================

    // Prevent simulation from running forever if something
    // goes wrong with the transmitter or receiver.
    initial begin

        #2_000_000;

        $display("ERROR: UART TEST TIMED OUT!");

        $fatal(1);

    end

endmodule