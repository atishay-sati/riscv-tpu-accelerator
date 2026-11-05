
import numpy as np
from pathlib import Path

# Project paths.
ROOT = Path(__file__).resolve().parent
PARAMS_DIR = ROOT / "fpga_params"
MEM_DIR = ROOT / "mem_files"

# Load the exact arrays used by the FPGA.
weights = np.load(PARAMS_DIR / "hidden_weights_int8.npy")
biases = np.load(PARAMS_DIR / "hidden_biases_int32.npy")

# Read the same test vector used by the MAC testbench.
pixels = []
with open("test_pixels.mem", "r") as f:
    for line in f:
        value = int(line.strip(), 16)
        if value >= 128:
            value -= 256
        pixels.append(value)

pixels = np.array(pixels, dtype=np.int8)

# Check dimensions before calculating.
assert pixels.shape == (784,), f"Unexpected pixel shape: {pixels.shape}"
assert weights.shape == (128, 784), f"Unexpected weight shape: {weights.shape}"
assert biases.shape == (128,), f"Unexpected bias shape: {biases.shape}"

# Calculate neuron 0's dot product using integer arithmetic.
mac_sum = int(
    np.sum(
        pixels.astype(np.int32) * weights[0].astype(np.int32),
        dtype=np.int64
    )
)

# Add the neuron's INT32 bias.
biased_sum = mac_sum + int(biases[0])

print("=== HIDDEN NEURON 0 REFERENCE ===")
print("MAC sum:    ", mac_sum)
print("Bias:       ", int(biases[0]))
print("MAC + bias: ", biased_sum)

# The MAC result from our verified Vivado test.
expected_mac = -102019

assert mac_sum == expected_mac, (
    f"MAC mismatch: Python={mac_sum}, expected={expected_mac}"
)

print("\nPASS: Python MAC matches the verified Vivado result.")
