
`timescale 1ns / 1ps

module uart_rx #(
    parameter integer CLOCK_FREQ_HZ = 100_000_000,
    parameter integer BAUD_RATE    = 115200
)(
    input  logic       clk,
    input  logic       reset,

    // Serial input from the computer's UART transmitter.
    input  logic       rx,

    // Received 8-bit byte.
    output logic [7:0] data_out,

    // Goes high for one clock cycle when a valid byte arrives.
    output logic       data_valid,

    // Goes high for one clock cycle if the stop bit is invalid.
    output logic       framing_error
);

    // Number of FPGA clock cycles in one UART bit.
    // At 100 MHz and 115200 baud, this is approximately 868.
    localparam integer CLKS_PER_BIT = CLOCK_FREQ_HZ / BAUD_RATE;

    // We first check the middle of the start bit.
    localparam integer HALF_BIT = CLKS_PER_BIT / 2;

    // Counter wide enough to count one complete UART bit.
    localparam integer COUNT_WIDTH = $clog2(CLKS_PER_BIT + 1);

    // UART receiver states.
    typedef enum logic [1:0] {
        IDLE,
        START_BIT,
        DATA_BITS,
        STOP_BIT
    } state_t;

    state_t state;

    // Clock counter used to determine when to sample each bit.
    logic [COUNT_WIDTH-1:0] clk_count;

    // Tracks which of the 8 data bits we are receiving.
    logic [2:0] bit_index;

    // Temporary storage for the incoming byte.
    logic [7:0] rx_shift_reg;

    // --------------------------------------------------------
    // INPUT SYNCHRONIZER
    // --------------------------------------------------------
    // The UART signal is asynchronous to our FPGA clock.
    // Two flip-flops reduce the risk of metastability.
    // --------------------------------------------------------

    logic rx_meta;
    logic rx_sync;

    always_ff @(posedge clk) begin
        if (reset) begin
            rx_meta <= 1'b1;
            rx_sync <= 1'b1;
        end
        else begin
            rx_meta <= rx;
            rx_sync <= rx_meta;
        end
    end

    // --------------------------------------------------------
    // UART RECEIVER STATE MACHINE
    // --------------------------------------------------------

    always_ff @(posedge clk) begin

        if (reset) begin
            state         <= IDLE;
            clk_count     <= '0;
            bit_index     <= '0;
            rx_shift_reg  <= '0;
            data_out      <= '0;
            data_valid    <= 1'b0;
            framing_error <= 1'b0;
        end
        else begin

            // These signals are pulses, so clear them by default.
            data_valid    <= 1'b0;
            framing_error <= 1'b0;

            case (state)

                // --------------------------------------------
                // IDLE: Wait for the start bit.
                // UART lines normally remain HIGH when idle.
                // --------------------------------------------

                IDLE: begin
                    clk_count <= '0;

                    if (rx_sync == 1'b0) begin
                        state <= START_BIT;
                    end
                end

                // --------------------------------------------
                // START_BIT: Check the middle of the start bit.
                // This helps reject short noise pulses.
                // --------------------------------------------

                START_BIT: begin
                    if (clk_count == HALF_BIT - 1) begin

                        clk_count <= '0;

                        // A valid start bit must still be LOW.
                        if (rx_sync == 1'b0) begin
                            bit_index <= 3'd0;
                            state     <= DATA_BITS;
                        end
                        else begin
                            state <= IDLE;
                        end

                    end
                    else begin
                        clk_count <= clk_count + 1'b1;
                    end
                end

                // --------------------------------------------
                // DATA_BITS: Receive the 8 data bits.
                // UART sends the least significant bit first.
                // --------------------------------------------

                DATA_BITS: begin
                    if (clk_count == CLKS_PER_BIT - 1) begin

                        clk_count <= '0;

                        // Store the sampled serial bit.
                        rx_shift_reg[bit_index] <= rx_sync;

                        if (bit_index == 3'd7) begin
                            state <= STOP_BIT;
                        end
                        else begin
                            bit_index <= bit_index + 1'b1;
                        end

                    end
                    else begin
                        clk_count <= clk_count + 1'b1;
                    end
                end

                // --------------------------------------------
                // STOP_BIT: Wait until the center of the stop bit.
                // A valid UART stop bit must be HIGH.
                // --------------------------------------------

                STOP_BIT: begin
                    if (clk_count == CLKS_PER_BIT - 1) begin

                        clk_count <= '0;
                        state     <= IDLE;

                        if (rx_sync == 1'b1) begin
                            data_out   <= rx_shift_reg;
                            data_valid <= 1'b1;
                        end
                        else begin
                            framing_error <= 1'b1;
                        end

                    end
                    else begin
                        clk_count <= clk_count + 1'b1;
                    end
                end

                default: begin
                    state <= IDLE;
                end

            endcase
        end
    end

endmodule
