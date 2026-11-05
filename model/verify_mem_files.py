
from pathlib import Path
import numpy as np

PARAM_DIR = Path("fpga_params")


def verify_int8_mem(filename, expected_array):
    """Verify INT8 memory contents and ordering."""
    expected = np.asarray(expected_array, dtype=np.int8).flatten(order="C")
    path = PARAM_DIR / filename
    lines = path.read_text().splitlines()

    assert len(lines) == len(expected), (
        f"{filename}: expected {len(expected)} entries, "
        f"found {len(lines)}"
    )

    expected_hex = [
        f"{int(value) & 0xFF:02x}" for value in expected
    ]

    assert lines == expected_hex, f"{filename}: data mismatch"
    print(f"PASS: {filename} ({len(lines):,} entries)")


def verify_int32_mem(filename, expected_array):
    """Verify INT32 memory contents and ordering."""
    expected = np.asarray(expected_array, dtype=np.int32).flatten(order="C")
    path = PARAM_DIR / filename
    lines = path.read_text().splitlines()

    assert len(lines) == len(expected), (
        f"{filename}: expected {len(expected)} entries, "
        f"found {len(lines)}"
    )

    expected_hex = [
        f"{int(value) & 0xFFFFFFFF:08x}" for value in expected
    ]

    assert lines == expected_hex, f"{filename}: data mismatch"
    print(f"PASS: {filename} ({len(lines):,} entries)")


# Load the original quantized parameters.
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

# Apply the same FPGA layout transformation as the exporter.
hidden_w_fpga = hidden_w_tflite.T.copy()
output_w_fpga = output_w_tflite.T.copy()

# Verify dimensions.
assert hidden_w_fpga.shape == (784, 128)
assert output_w_fpga.shape == (128, 32)

# Verify file contents and exact memory ordering.
verify_int8_mem("hidden_weights.mem", hidden_w_fpga)
verify_int32_mem("hidden_biases.mem", hidden_b)
verify_int8_mem("output_weights.mem", output_w_fpga)
verify_int32_mem("output_biases.mem", output_b)

print("\nAll FPGA parameter memory files passed validation.")
