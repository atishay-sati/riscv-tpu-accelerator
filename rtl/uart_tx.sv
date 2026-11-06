`timescale 1ns / 1ps

module uart_tx #(
    parameter integer CLOCK_FREQ_HZ = 100_000_000,
    parameter integer BAUD_RATE = 115200
)(
    input  logic       clk,
    input  logic       reset,

    // Byte that we want to transmit.
    input  logic [7:0] data_in,

    // One-clock-cycle pulse requesting transmission.
    input  logic       tx_start,

    // Serial output connected to the computer.
    output logic       tx,

    // HIGH while a byte is being transmitted.
    output logic       tx_busy,

    // One-clock-cycle pulse when transmission finishes.
    output logic       tx_done
);

    // Number of FPGA clock cycles per UART bit.
    localparam integer CLKS_PER_BIT = CLOCK_FREQ_HZ / BAUD_RATE;

    // Counter width needed to count one UART bit.
    localparam integer COUNT_WIDTH = $clog2(CLKS_PER_BIT + 1);

    // UART transmitter states.
    typedef enum logic [1:0] {
        IDLE,
        START_BIT,
        DATA_BITS,
        STOP_BIT
    } state_t;

    state_t state;

    // Counts clock cycles during each transmitted bit.
    logic [COUNT_WIDTH-1:0] clk_count;

    // Tracks which data bit is being transmitted.
    logic [2:0] bit_index;

    // Holds the byte during transmission.
    logic [7:0] tx_shift_reg;

    // =====================================================
    // UART TRANSMITTER STATE MACHINE
    // =====================================================

    always_ff @(posedge clk) begin

        // Reset all transmitter signals to their idle values.
        if (reset) begin
            state        <= IDLE;
            clk_count    <= 0;
            bit_index    <= 0;
            tx_shift_reg <= 0;

            tx           <= 1'b1;
            tx_busy      <= 1'b0;
            tx_done      <= 1'b0;
        end

        else begin

            // Default: tx_done is a one-clock-cycle pulse.
            tx_done <= 1'b0;

            case (state)

                // -----------------------------------------
                // IDLE: Wait for a transmission request.
                // -----------------------------------------
                IDLE: begin
                    tx       <= 1'b1;
                    tx_busy  <= 1'b0;
                    clk_count <= 0;
                    bit_index <= 0;

                    if (tx_start) begin
                        // Save the byte so it stays unchanged
                        // throughout the transmission.
                        tx_shift_reg <= data_in;

                        // UART start bit is always logic 0.
                        tx <= 1'b0;

                        tx_busy <= 1'b1;
                        state <= START_BIT;
                    end
                end

                // -----------------------------------------
                // START_BIT: Hold TX LOW for one bit period.
                // -----------------------------------------
                START_BIT: begin
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= 0;

                        // Begin transmitting data bit 0.
                        tx <= tx_shift_reg[0];

                        // Shift the next bit into position 0.
                        tx_shift_reg <= {1'b0, tx_shift_reg[7:1]};

                        state <= DATA_BITS;
                    end
                    else begin
                        clk_count <= clk_count + 1;
                    end
                end

                // -----------------------------------------
                // DATA_BITS: Transmit 8 bits, LSB first.
                // -----------------------------------------
                DATA_BITS: begin
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= 0;

                        if (bit_index == 7) begin
                            // All 8 data bits have been sent.
                            // Begin the stop bit.
                            tx <= 1'b1;
                            state <= STOP_BIT;
                        end
                        else begin
                            // Move to the next data bit.
                            bit_index <= bit_index + 1;

                            tx <= tx_shift_reg[0];

                            tx_shift_reg <= {
                                1'b0,
                                tx_shift_reg[7:1]
                            };
                        end
                    end
                    else begin
                        clk_count <= clk_count + 1;
                    end
                end

                // -----------------------------------------
                // STOP_BIT: Hold TX HIGH for one bit period.
                // -----------------------------------------
                STOP_BIT: begin
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= 0;

                        // Transmission is complete.
                        tx <= 1'b1;
                        tx_busy <= 1'b0;
                        tx_done <= 1'b1;

                        state <= IDLE;
                    end
                    else begin
                        clk_count <= clk_count + 1;
                    end
                end

                // Recover to idle if an invalid state occurs.
                default: begin
                    state <= IDLE;
                    tx <= 1'b1;
                    tx_busy <= 1'b0;
                    clk_count <= 0;
                    bit_index <= 0;
                end

            endcase
        end
    end

endmodule