
import serial
import time

# ==========================================
# Basys 3 UART configuration
# ==========================================

COM_PORT = "COM6"
BAUD_RATE = 115200

# The FPGA expects 784 pixels (28 x 28).
IMAGE_SIZE = 28 * 28

# Packet header expected by the FPGA.
PACKET_HEADER = 0xAA


def main():

    print("Connecting to Basys 3...")

    try:
        # Open the USB UART connection.
        ser = serial.Serial(
            port=COM_PORT,
            baudrate=BAUD_RATE,
            timeout=10
        )

        # Give the serial connection a moment to initialize.
        time.sleep(0.5)

        # Clear any old incoming serial data.
        ser.reset_input_buffer()

        print("Connected to", COM_PORT)

        # ==========================================
        # Create a test image
        # ==========================================

        # For now, send an all-zero image.
        # Each pixel is represented by one unsigned byte.
        pixels = bytes([0] * IMAGE_SIZE)

        # ==========================================
        # Send packet to FPGA
        # ==========================================

        print("Sending image packet...")

        # Send the packet header first.
        ser.write(bytes([PACKET_HEADER]))

        # Then send all 784 pixel values.
        ser.write(pixels)

        # Wait until all transmitted bytes leave the PC buffer.
        ser.flush()

        print("Sent 784 pixels.")

        # ==========================================
        # Receive prediction from FPGA
        # ==========================================

        print("Waiting for TPU prediction...")

        result = ser.read(1)

        if len(result) == 1:

            # The FPGA returns the predicted class in bits [4:0].
            predicted_class = result[0] & 0x1F

            print("Prediction received!")
            print("Predicted class ID:", predicted_class)

        else:
            print("ERROR: Timed out waiting for FPGA response.")

    except serial.SerialException as error:
        print("UART connection error:", error)

    finally:
        if "ser" in locals() and ser.is_open:
            ser.close()
            print("Serial connection closed.")


if __name__ == "__main__":
    main()
