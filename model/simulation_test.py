import serial
import time
from pathlib import Path

PROJECT = Path(
    r"C:\Users\Atish\VerliogProjects\TPU_Integration"
)

INPUT_FILE = PROJECT / "mem_files/dense_layer_test/input_pixels.mem"
EXPECTED_FILE = PROJECT / "mem_files/output_layer_test/expected_outputs.mem"

# Load the exact 784-byte input used by the passing testbench.
pixels = bytes(
    int(line.strip(), 16)
    for line in INPUT_FILE.read_text().splitlines()
    if line.strip()
)

# Calculate the expected class from the simulation's expected scores.
scores = [
    int(line.strip(), 16)
    for line in EXPECTED_FILE.read_text().splitlines()
    if line.strip()
]

scores = [value - 256 if value >= 128 else value for value in scores]

assert len(pixels) == 784, f"Expected 784 pixels, got {len(pixels)}"
assert len(scores) == 32, f"Expected 32 scores, got {len(scores)}"

expected_class = max(range(32), key=lambda i: scores[i])

print("Expected FPGA class:", expected_class)
print("Sending synthetic input:", len(pixels), "pixels")

with serial.Serial("COM6", 115200, timeout=30, write_timeout=5) as ser:
    time.sleep(0.2)
    ser.reset_input_buffer()

    sent = ser.write(bytes([0xAA]) + pixels)
    ser.flush()

    print("Bytes sent:", sent)

    result = ser.read(1)

    if result:
        print("Physical FPGA class:", result[0])
        print("MATCH:", result[0] == expected_class)
    else:
        print("FPGA timed out.")