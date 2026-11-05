
from pathlib import Path
import numpy as np
import json

PARAM_DIR = Path("fpga_params")


def save_int8_mem(filename, array):
    """Write each INT8 value as one byte of hexadecimal."""
    values = np.asarray(array, dtype=np.int8).flatten(order="C")

    with open(PARAM_DIR / filename, "w") as f:
        for value in values:
            f.write(f"{int(value) & 0xFF:02x}\n")

    print(f"{filename}: {len(values):,} bytes")


def save_int32_mem(filename, array):
    """Write each INT32 value as eight hex digits."""
    values = np.asarray(array, dtype=np.int32).flatten(order="C")

    with open(PARAM_DIR / filename, "w") as f:
        for value in values:
            f.write(f"{int(value) & 0xFFFFFFFF:08x}\n")

    print(f"{filename}: {len(values):,} words")


def save_float32_bits_mem(filename, values):
    """Save float32 values as their raw IEEE-754 bit patterns."""
    values = np.asarray(values, dtype=np.float32).flatten(order="C")
    bits = values.view(np.uint32)

    with open(PARAM_DIR / filename, "w") as f:
        for value in bits:
            f.write(f"{int(value):08x}\n")

    print(f"{filename}: {len(values):,} scale values")


# --------------------------------------------------
# 1. Load the original TFLite quantized parameters.
# Do not modify the original .npy files.
# --------------------------------------------------

hidden_w_tflite = np.load(
    PARAM_DIR / "hidden_weights_int8.npy"
)
hidden_b = np.load(
    PARAM_DIR / "hidden_biases_int32.npy"
)
output_w_tflite = np.load(
    PARAM_DIR / "output_weights_int8.npy"
)
output_b = np.load(
    PARAM_DIR / "output_biases_int32.npy"
)


# --------------------------------------------------
# 2. Check the original TFLite dimensions.
#
# Hidden weights: [128 neurons, 784 input features]
# Output weights: [32 classes, 128 hidden neurons]
# --------------------------------------------------

assert hidden_w_tflite.shape == (128, 784), hidden_w_tflite.shape
assert hidden_b.shape == (128,), hidden_b.shape
assert output_w_tflite.shape == (32, 128), output_w_tflite.shape
assert output_b.shape == (32,), output_b.shape


# --------------------------------------------------
# 3. Transpose weights for the FPGA matrix layout.
#
# The array computes C = A x B.
# B[row][column] = weight[input_feature][neuron]
#
# Hidden: [784 input features, 128 neurons]
# Output: [128 hidden neurons, 32 classes]
# --------------------------------------------------

hidden_w = hidden_w_tflite.T.copy()
output_w = output_w_tflite.T.copy()

assert hidden_w.shape == (784, 128), hidden_w.shape
assert output_w.shape == (128, 32), output_w.shape


# --------------------------------------------------
# 4. Verify the transpose preserves every weight.
# --------------------------------------------------

assert np.array_equal(hidden_w.T, hidden_w_tflite)
assert np.array_equal(output_w.T, output_w_tflite)

print("Weight dimensions verified.")
print(f"Hidden weights for FPGA: {hidden_w.shape}")
print(f"Output weights for FPGA: {output_w.shape}")


# --------------------------------------------------
# 5. Export weights in row-major order.
#
# Hidden address = feature_index * 128 + neuron_index
# Output address = hidden_index * 32 + class_index
#
# Bias arrays remain in their original order.
# --------------------------------------------------

save_int8_mem("hidden_weights.mem", hidden_w)
save_int32_mem("hidden_biases.mem", hidden_b)

save_int8_mem("output_weights.mem", output_w)
save_int32_mem("output_biases.mem", output_b)


# --------------------------------------------------
# 6. Export the original quantization scales.
#
# These are reference values, not final hardware
# requantization multipliers.
# --------------------------------------------------

metadata = json.loads(
    (PARAM_DIR / "quantization_metadata.json").read_text()
)

for key, filename in [
    ("hidden_weights_int8", "hidden_weight_scales.mem"),
    ("hidden_biases_int32", "hidden_bias_scales.mem"),
    ("output_weights_int8", "output_weight_scales.mem"),
    ("output_biases_int32", "output_bias_scales.mem"),
]:
    scales = metadata[key]["scales"]
    save_float32_bits_mem(filename, scales)


print("\nFPGA memory files created successfully.")
print("Weight ordering: transposed matrices, NumPy C-order.")
print("Original .npy files were not modified.")
