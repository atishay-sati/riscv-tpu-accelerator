
import os

# Reduce TensorFlow startup messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*tf.lite.Interpreter is deprecated.*",
    category=UserWarning
)

import numpy as np
import tensorflow as tf


# ============================================================
# 1. LOAD THE TFLITE MODEL
# ============================================================

MODEL_PATH = "quickdraw_dense_32_int8.tflite"
PARAMS_DIR = "fpga_params"

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH,
    experimental_preserve_all_tensors=True
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero_point = input_details["quantization"]
output_scale, output_zero_point = output_details["quantization"]

print("=== MODEL INFORMATION ===")
print("Input shape:", input_details["shape"])
print("Input dtype:", input_details["dtype"])
print("Input scale:", input_scale)
print("Input zero point:", input_zero_point)
print("Output shape:", output_details["shape"])
print("Output scale:", output_scale)
print("Output zero point:", output_zero_point)


# ============================================================
# 2. LOAD EXPORTED WEIGHTS AND BIASES
# ============================================================

hidden_weights = np.load(
    os.path.join(PARAMS_DIR, "hidden_weights_int8.npy")
).astype(np.int8)

hidden_biases = np.load(
    os.path.join(PARAMS_DIR, "hidden_biases_int32.npy")
).astype(np.int64)

output_weights = np.load(
    os.path.join(PARAMS_DIR, "output_weights_int8.npy")
).astype(np.int8)

output_biases = np.load(
    os.path.join(PARAMS_DIR, "output_biases_int32.npy")
).astype(np.int64)

print("\n=== PARAMETER SHAPES ===")
print("Hidden weights:", hidden_weights.shape)
print("Hidden biases:", hidden_biases.shape)
print("Output weights:", output_weights.shape)
print("Output biases:", output_biases.shape)


# ============================================================
# 3. FIND TENSOR QUANTIZATION INFORMATION
# ============================================================

def find_tensor(name_part):
    """Find a TFLite tensor by part of its name."""
    matches = [
        tensor
        for tensor in interpreter.get_tensor_details()
        if name_part in tensor["name"]
    ]

    if not matches:
        raise ValueError(
            f"Could not find a tensor containing: {name_part}"
        )

    # Prefer the exact tensor name if there are multiple matches.
    exact_matches = [
        tensor for tensor in matches
        if tensor["name"] == name_part
    ]

    return exact_matches[0] if exact_matches else matches[0]


hidden_weight_info = find_tensor(
    "sequential_1/hidden_1/MatMul"
)

hidden_bias_info = find_tensor(
    "sequential_1/hidden_1/Relu;sequential_1/hidden_1/BiasAdd"
)

hidden_activation_info = find_tensor(
    "sequential_1/hidden_1/MatMul;sequential_1/hidden_1/Relu;sequential_1/hidden_1/BiasAdd"
)

output_weight_info = find_tensor(
    "sequential_1/class_logits_1/MatMul"
)

output_bias_info = find_tensor(
    "sequential_1/class_logits_1/BiasAdd"
)

hidden_weight_scales = (
    hidden_weight_info["quantization_parameters"]["scales"]
)

output_weight_scales = (
    output_weight_info["quantization_parameters"]["scales"]
)

hidden_scale = (
    hidden_activation_info["quantization_parameters"]["scales"][0]
)

hidden_zero_point = (
    hidden_activation_info["quantization_parameters"]["zero_points"][0]
)

print("\n=== QUANTIZATION INFORMATION ===")
print("Hidden weight scale count:", len(hidden_weight_scales))
print("Hidden bias scale count:",
      len(hidden_bias_info["quantization_parameters"]["scales"]))
print("Hidden activation scale:", hidden_scale)
print("Hidden activation zero point:", hidden_zero_point)
print("Output weight scale count:", len(output_weight_scales))
print("Output bias scale count:",
      len(output_bias_info["quantization_parameters"]["scales"]))

# Bias scales should approximately equal:
# input activation scale * weight scale for each output channel.
expected_hidden_bias_scales = input_scale * hidden_weight_scales
actual_hidden_bias_scales = (
    hidden_bias_info["quantization_parameters"]["scales"]
)

expected_output_bias_scales = hidden_scale * output_weight_scales
actual_output_bias_scales = (
    output_bias_info["quantization_parameters"]["scales"]
)

print(
    "Hidden bias scales match expected:",
    np.allclose(
        actual_hidden_bias_scales,
        expected_hidden_bias_scales,
        rtol=1e-5,
        atol=1e-12
    )
)

print(
    "Output bias scales match expected:",
    np.allclose(
        actual_output_bias_scales,
        expected_output_bias_scales,
        rtol=1e-5,
        atol=1e-12
    )
)


# ============================================================
# 3B. EXTRACT FIXED-POINT REQUANTIZATION PARAMETERS
# ============================================================

# TensorFlow Lite represents the real multiplier using
# an integer multiplier and a shift.

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

    # TensorFlow Lite convention:
    # positive exponent = left shift
    # negative exponent = right shift.
    return multiplier, exponent


# ------------------------------------------------------------
# Calculate parameters for all 128 hidden neurons.
# ------------------------------------------------------------

hidden_real_multipliers = (
    float(input_scale)
    * hidden_weight_scales.astype(np.float64)
    / float(hidden_scale)
)

hidden_fixed_parameters = [
    quantize_multiplier(float(value))
    for value in hidden_real_multipliers
]

hidden_multipliers = np.array(
    [parameter[0] for parameter in hidden_fixed_parameters],
    dtype=np.int64
)

hidden_shifts = np.array(
    [parameter[1] for parameter in hidden_fixed_parameters],
    dtype=np.int32
)

# Verify the number of parameters.
assert len(hidden_multipliers) == 128
assert len(hidden_shifts) == 128

# Verify that shifts fit in signed 8-bit values.
assert np.all(hidden_shifts >= -128)
assert np.all(hidden_shifts <= 127)

# Verify neuron 0 against our previous test.
assert hidden_multipliers[0] == 1608709942
assert hidden_shifts[0] == -13

# Export the parameters.
os.makedirs(PARAMS_DIR, exist_ok=True)

with open(
    os.path.join(PARAMS_DIR, "hidden_multipliers.mem"),
    "w"
) as file:
    for multiplier in hidden_multipliers:
        file.write(
            f"{int(multiplier) & 0xFFFFFFFF:08X}\n"
        )

with open(
    os.path.join(PARAMS_DIR, "hidden_shifts.mem"),
    "w"
) as file:
    for shift in hidden_shifts:
        file.write(
            f"{int(shift) & 0xFF:02X}\n"
        )

print("\n=== HIDDEN LAYER REQUANTIZATION PARAMETERS ===")
print("Number of multipliers:", len(hidden_multipliers))
print("Number of shifts:", len(hidden_shifts))

print("\nNeuron 0:")
print("Real multiplier:", hidden_real_multipliers[0])
print("Integer multiplier:", hidden_multipliers[0])
print("Hex multiplier:", hex(int(hidden_multipliers[0])))
print("Shift:", hidden_shifts[0])

print("\nExported:")
print(os.path.join(PARAMS_DIR, "hidden_multipliers.mem"))
print(os.path.join(PARAMS_DIR, "hidden_shifts.mem"))


# ============================================================
# 4. SIMULATE ONE QUANTIZED DENSE LAYER
# ============================================================

def dense_layer(
    input_q,
    weights,
    biases,
    input_zp,
    input_scale,
    weight_scales,
    output_scale,
    output_zp,
    relu=False
):
    """
    Approximate a quantized fully connected layer.

    input_q: quantized input values
    weights: INT8 weights, shaped (output_neurons, input_neurons)
    biases: INT32 biases
    input_zp: input zero point
    input_scale: scale of THIS layer's input
    weight_scales: one scale per output neuron
    output_scale: scale of the output activation
    output_zp: output zero point
    relu: whether to apply ReLU
    """

    # Remove the input zero point before multiplication.
    input_adjusted = (
        input_q.astype(np.int64) - int(input_zp)
    )

    # Integer dot products plus integer biases.
    accumulators = (
        weights.astype(np.int64) @ input_adjusted
        + biases
    )

    # IMPORTANT:
    # Each output channel has its own weight scale.
    # Use this layer's input scale, not a global scale.
    real_values = (
        accumulators.astype(np.float64)
        * float(input_scale)
        * weight_scales.astype(np.float64)
    )

    # First layer uses ReLU.
    if relu:
        real_values = np.maximum(real_values, 0.0)

    # Convert real values to the output's quantized format.
    quantized = np.rint(
        real_values / float(output_scale) + int(output_zp)
    )

    # INT8 values must be between -128 and 127.
    quantized = np.clip(quantized, -128, 127).astype(np.int8)

    return quantized, accumulators


# ============================================================
# 5. CREATE A TEST IMAGE
# ============================================================

# Generate a deterministic 28x28 grayscale test image.
# Pixel values range from 0 to 255.
test_pixels = (
    (np.arange(784) * 37 + 13) % 256
).astype(np.uint8)

# Quantize pixels using the model's input zero point.
test_input = (
    test_pixels.astype(np.int16) + input_zero_point
).astype(np.int8)

# TFLite expects the input shape [1, 784].
tflite_input = test_input.reshape(input_details["shape"])

interpreter.set_tensor(input_details["index"], tflite_input)
interpreter.invoke()

# Read the actual hidden-layer output from TensorFlow Lite.
tflite_hidden = interpreter.get_tensor(
    hidden_activation_info["index"]
).reshape(-1)

# Read the final output.
tflite_output = interpreter.get_tensor(
    output_details["index"]
).reshape(-1)

print("\n=== TFLITE HIDDEN ACTIVATIONS ===")
print("First 20:", tflite_hidden[:20])
print("Neuron 0:", int(tflite_hidden[0]))


# ============================================================
# 6. RUN OUR TWO-LAYER REFERENCE CALCULATION
# ============================================================

# Layer 1: 784 inputs -> 128 hidden neurons.
hidden_output, hidden_accumulators = dense_layer(
    input_q=test_input,
    weights=hidden_weights,
    biases=hidden_biases,
    input_zp=input_zero_point,
    input_scale=input_scale,
    weight_scales=hidden_weight_scales,
    output_scale=hidden_scale,
    output_zp=hidden_zero_point,
    relu=True
)

# Layer 2: 128 hidden neurons -> 32 output classes.
output_q, output_accumulators = dense_layer(
    input_q=hidden_output,
    weights=output_weights,
    biases=output_biases,
    input_zp=hidden_zero_point,

    # The second layer uses the HIDDEN activation scale.
    input_scale=hidden_scale,

    weight_scales=output_weight_scales,
    output_scale=output_scale,
    output_zp=output_zero_point,
    relu=False
)


# ============================================================
# 7. COMPARE OUR RESULT AGAINST TFLITE
# ============================================================

differences = np.abs(
    tflite_output.astype(np.int16)
    - output_q.astype(np.int16)
)

print("\n=== FIRST LAYER ===")
print("Hidden output shape:", hidden_output.shape)
print("First 20 hidden activations:", hidden_output[:20])
print("First 10 biased accumulators:", hidden_accumulators[:10])

print("\n=== FINAL OUTPUT COMPARISON ===")
print("TFLite output:", tflite_output)
print("Our output:   ", output_q)
print("Maximum absolute difference:", np.max(differences))
print("Matching output values:", np.sum(differences == 0), "/ 32")
print("TFLite predicted index:", np.argmax(tflite_output))
print("Our predicted index:   ", np.argmax(output_q))


print("\n=== NEURON 0 COMPARISON ===")

print("TFLite activation:", int(tflite_hidden[0]))
print("Our activation:   ", int(hidden_output[0]))

print(
    "Match:",
    int(tflite_hidden[0]) == int(hidden_output[0])
)


# ============================================================
# 8. EXPORT NEURON 0 FPGA TEST FILES
# ============================================================

FPGA_TEST_DIR = "fpga_params/neuron0_test"
os.makedirs(FPGA_TEST_DIR, exist_ok=True)


def write_hex_file(filename, values, bits=8):
    """Write integer values as fixed-width hexadecimal."""
    digits = bits // 4
    mask = (1 << bits) - 1

    with open(filename, "w") as file:
        for value in values:
            file.write(f"{int(value) & mask:0{digits}X}\n")


# Convert quantized input pixels into unsigned 0-255 values.
adjusted_pixels = (
    test_input.astype(np.int16) - int(input_zero_point)
).astype(np.uint8)

# Export the input pixels and neuron 0 weights.
write_hex_file(
    os.path.join(FPGA_TEST_DIR, "pixels.mem"),
    adjusted_pixels
)

write_hex_file(
    os.path.join(FPGA_TEST_DIR, "weights.mem"),
    hidden_weights[0]
)

# Export the neuron 0 bias.
write_hex_file(
    os.path.join(FPGA_TEST_DIR, "bias.mem"),
    [hidden_biases[0]],
    bits=32
)

# Export the expected accumulator.
write_hex_file(
    os.path.join(FPGA_TEST_DIR, "expected_accumulator.mem"),
    [hidden_accumulators[0]],
    bits=32
)

# Export the expected TFLite activation.
write_hex_file(
    os.path.join(FPGA_TEST_DIR, "expected_activation.mem"),
    [tflite_hidden[0]]
)

print("\n=== FPGA TEST FILES EXPORTED ===")
print("Directory:", FPGA_TEST_DIR)
print("Expected accumulator:", int(hidden_accumulators[0]))
print("Expected activation:", int(tflite_hidden[0]))
print("Adjusted pixel range:",
      int(np.min(adjusted_pixels)),
      "to",
      int(np.max(adjusted_pixels)))


# ============================================================
# 9. EXPORT ALL 128 HIDDEN NEURON TEST VECTORS
# ============================================================

# Directory for the complete hidden-layer test.
ALL_NEURONS_DIR = "fpga_params/all_neurons_test"
os.makedirs(ALL_NEURONS_DIR, exist_ok=True)

# Export all 128 expected accumulators.
write_hex_file(
    os.path.join(ALL_NEURONS_DIR, "accumulators.mem"),
    hidden_accumulators,
    bits=32
)

# Export all 128 expected TFLite activations.
write_hex_file(
    os.path.join(ALL_NEURONS_DIR, "expected_activations.mem"),
    tflite_hidden,
    bits=8
)

# Export all 128 fixed-point multipliers.
write_hex_file(
    os.path.join(ALL_NEURONS_DIR, "multipliers.mem"),
    hidden_multipliers,
    bits=32
)

# Export all 128 fixed-point shifts.
write_hex_file(
    os.path.join(ALL_NEURONS_DIR, "shifts.mem"),
    hidden_shifts,
    bits=8
)

print("\n=== ALL HIDDEN NEURON TEST FILES EXPORTED ===")
print("Directory:", ALL_NEURONS_DIR)
print("Accumulators:", len(hidden_accumulators))
print("Expected activations:", len(tflite_hidden))
print("Multipliers:", len(hidden_multipliers))
print("Shifts:", len(hidden_shifts))


print("\nReference test completed.")
