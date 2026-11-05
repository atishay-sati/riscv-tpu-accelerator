
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf


# ============================================================
# 1. FILE LOCATIONS
# ============================================================

MODEL_PATH = Path("quickdraw_dense_32_int8.tflite")
PARAM_DIR = Path("fpga_params")
TEST_DIR = PARAM_DIR / "output_layer_test"

TEST_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD TFLITE MODEL
# ============================================================

interpreter = tf.lite.Interpreter(
    model_path=str(MODEL_PATH),
    experimental_preserve_all_tensors=True
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero_point = input_details["quantization"]
output_scale, output_zero_point = output_details["quantization"]


# ============================================================
# 3. FIND HIDDEN ACTIVATION AND OUTPUT WEIGHT TENSORS
# ============================================================

def find_tensor(name_part):
    matches = [
        tensor
        for tensor in interpreter.get_tensor_details()
        if name_part in tensor["name"]
    ]

    if not matches:
        raise ValueError(f"Could not find tensor: {name_part}")

    return matches[0]


hidden_activation_info = find_tensor(
    "sequential_1/hidden_1/MatMul;sequential_1/hidden_1/Relu;sequential_1/hidden_1/BiasAdd"
)

output_weight_info = find_tensor(
    "sequential_1/class_logits_1/MatMul"
)

hidden_scale = (
    hidden_activation_info["quantization_parameters"]["scales"][0]
)

hidden_zero_point = (
    hidden_activation_info["quantization_parameters"]["zero_points"][0]
)

output_weight_scales = (
    output_weight_info["quantization_parameters"]["scales"]
)


# ============================================================
# 4. LOAD OUTPUT LAYER WEIGHTS AND BIASES
# ============================================================

weights = np.load(
    PARAM_DIR / "output_weights_int8.npy"
).astype(np.int8)

biases = np.load(
    PARAM_DIR / "output_biases_int32.npy"
).astype(np.int64)

assert weights.shape == (32, 128), weights.shape
assert biases.shape == (32,), biases.shape
assert len(output_weight_scales) == 32


# ============================================================
# 5. GENERATE FIXED-POINT MULTIPLIERS AND SHIFTS
# ============================================================

def quantize_multiplier(real_multiplier):
    if real_multiplier == 0.0:
        return 0, 0

    significand, exponent = np.frexp(real_multiplier)

    multiplier = int(
        np.rint(significand * (1 << 31))
    )

    if multiplier == (1 << 31):
        multiplier //= 2
        exponent += 1

    return multiplier, exponent


# The output layer converts hidden activations into output scores.
real_multipliers = (
    float(hidden_scale)
    * output_weight_scales.astype(np.float64)
    / float(output_scale)
)

fixed_parameters = [
    quantize_multiplier(float(value))
    for value in real_multipliers
]

multipliers = np.array(
    [parameter[0] for parameter in fixed_parameters],
    dtype=np.int64
)

shifts = np.array(
    [parameter[1] for parameter in fixed_parameters],
    dtype=np.int32
)

assert len(multipliers) == 32
assert len(shifts) == 32

assert np.all(shifts >= -128)
assert np.all(shifts <= 127)


# ============================================================
# 6. GENERATE THE SAME TEST IMAGE
# ============================================================

test_pixels = (
    (np.arange(784) * 37 + 13) % 256
).astype(np.uint8)

test_input = (
    test_pixels.astype(np.int16) + input_zero_point
).astype(np.int8)

interpreter.set_tensor(
    input_details["index"],
    test_input.reshape(input_details["shape"])
)

interpreter.invoke()


# ============================================================
# 7. EXTRACT ACTUAL TFLITE HIDDEN ACTIVATIONS AND OUTPUTS
# ============================================================

tflite_hidden = interpreter.get_tensor(
    hidden_activation_info["index"]
).reshape(-1).astype(np.int8)

tflite_output = interpreter.get_tensor(
    output_details["index"]
).reshape(-1).astype(np.int8)

assert len(tflite_hidden) == 128
assert len(tflite_output) == 32


# Convert signed INT8 activations into adjusted unsigned values.
# The FPGA controller expects values in the range 0 to 255.

adjusted_activations = (
    tflite_hidden.astype(np.int16) - int(hidden_zero_point)
).astype(np.uint8)

assert np.all(adjusted_activations <= 255)


# ============================================================
# 8. WRITE HEX FILES
# ============================================================

def save_hex(filename, values, bits):
    values = np.asarray(values).flatten()

    mask = (1 << bits) - 1
    digits = bits // 4

    with open(TEST_DIR / filename, "w") as file:
        for value in values:
            file.write(
                f"{int(value) & mask:0{digits}X}\n"
            )

    print(f"{filename}: {len(values)} entries")


# Weights remain neuron-major:
# address = class_index * 128 + hidden_neuron_index

save_hex("output_weights.mem", weights, 8)
save_hex("output_biases.mem", biases, 32)

save_hex("output_multipliers.mem", multipliers, 32)
save_hex("output_shifts.mem", shifts, 8)

save_hex("input_activations.mem", adjusted_activations, 8)

# These are the actual TensorFlow Lite final outputs.
save_hex("expected_outputs.mem", tflite_output, 8)


# ============================================================
# 9. VERIFY ALL GENERATED FILES
# ============================================================

expected_counts = {
    "output_weights.mem": 32 * 128,
    "output_biases.mem": 32,
    "output_multipliers.mem": 32,
    "output_shifts.mem": 32,
    "input_activations.mem": 128,
    "expected_outputs.mem": 32,
}

print("\n=== VERIFYING OUTPUT LAYER FILES ===")

for filename, expected_count in expected_counts.items():
    path = TEST_DIR / filename

    assert path.exists(), f"Missing {filename}"

    actual_count = sum(
        1 for line in path.open()
        if line.strip()
    )

    assert actual_count == expected_count, (
        f"{filename}: expected {expected_count}, got {actual_count}"
    )

    print(f"VERIFIED: {filename}: {actual_count} entries")


# ============================================================
# 10. DISPLAY TEST INFORMATION
# ============================================================

print("\n------------------------------------------")
print("OUTPUT LAYER TEST FILES READY")
print("------------------------------------------")

print("Input features:", len(adjusted_activations))
print("Output classes:", len(tflite_output))

print("Hidden activation scale:", hidden_scale)
print("Hidden activation zero point:", hidden_zero_point)

print("Output scale:", output_scale)
print("Output zero point:", output_zero_point)

print("Adjusted activation range:",
      int(np.min(adjusted_activations)),
      "to",
      int(np.max(adjusted_activations)))

print("TFLite predicted class:", int(np.argmax(tflite_output)))

print("\nOutput layer multipliers:", len(multipliers))
print("Output layer shifts:", len(shifts))

print("\nALL OUTPUT LAYER TEST FILES GENERATED.")
