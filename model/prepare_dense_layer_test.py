from pathlib import Path
import numpy as np

PARAM_DIR = Path("fpga_params")
TEST_DIR = PARAM_DIR / "dense_layer_test"
TEST_DIR.mkdir(parents=True, exist_ok=True)


def save_hex(filename, values, bits):
    values = np.asarray(values).flatten()
    mask = (1 << bits) - 1
    digits = bits // 4

    with open(TEST_DIR / filename, "w") as f:
        for value in values:
            f.write(f"{int(value) & mask:0{digits}x}\n")

    print(f"{filename}: {len(values)} entries")


# Load original TFLite parameters.
weights = np.load(PARAM_DIR / "hidden_weights_int8.npy")
biases = np.load(PARAM_DIR / "hidden_biases_int32.npy")

assert weights.shape == (128, 784)
assert biases.shape == (128,)

# Preserve neuron-major ordering:
# address = neuron * 784 + feature
save_hex("hidden_weights.mem", weights, 8)
save_hex("hidden_biases.mem", biases, 32)


# Copy fixed-point quantization parameters.
for filename in [
    "hidden_multipliers.mem",
    "hidden_shifts.mem",
]:
    source = PARAM_DIR / filename
    destination = TEST_DIR / filename

    assert source.exists(), f"Missing {source}"
    destination.write_bytes(source.read_bytes())


# Generate the same deterministic input as the Python reference.
pixels = ((np.arange(784) * 37 + 13) % 256).astype(np.uint8)

# Input zero point is -128.
# The controller expects adjusted unsigned values.
adjusted_pixels = pixels.astype(np.uint8)

save_hex("input_pixels.mem", adjusted_pixels, 8)


# Copy the TFLite golden activations.
source_dir = PARAM_DIR / "all_neurons_test"

for filename in [
    "expected_activations.mem",
]:
    source = source_dir / filename
    destination = TEST_DIR / filename

    assert source.exists(), f"Missing {source}"
    destination.write_bytes(source.read_bytes())


# Verify the generated files.
expected_counts = {
    "hidden_weights.mem": 128 * 784,
    "hidden_biases.mem": 128,
    "hidden_multipliers.mem": 128,
    "hidden_shifts.mem": 128,
    "input_pixels.mem": 784,
    "expected_activations.mem": 128,
}

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

print("\nAll real-model test files are ready.")
