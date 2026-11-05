import json
from pathlib import Path

import numpy as np
import tflite

# Model and output locations
MODEL_PATH = Path("quickdraw_dense_32_int8.tflite")
OUTPUT_DIR = Path("fpga_params")
OUTPUT_DIR.mkdir(exist_ok=True)

# Exact tensor names from our TFLite model
TARGETS = {
    "hidden_weights_int8":
        "sequential_1/hidden_1/MatMul",
    "hidden_biases_int32":
        "sequential_1/hidden_1/Relu;sequential_1/hidden_1/BiasAdd",
    "output_weights_int8":
        "sequential_1/class_logits_1/MatMul",
    "output_biases_int32":
        "sequential_1/class_logits_1/BiasAdd",
}


def read_tensor_data(model, tensor, dtype):
    """Read a constant tensor directly from its TFLite buffer."""
    shape = [
        tensor.Shape(i)
        for i in range(tensor.ShapeLength())
    ]

    buffer = model.Buffers(tensor.Buffer())
    raw_data = buffer.DataAsNumpy()

    if raw_data is None or len(raw_data) == 0:
        raise RuntimeError(
            f"No constant data found for {tensor.Name().decode()}"
        )

    values = np.frombuffer(raw_data.tobytes(), dtype=dtype).copy()

    expected_size = int(np.prod(shape))
    if values.size != expected_size:
        raise RuntimeError(
            f"Shape mismatch: expected {expected_size} values, "
            f"got {values.size}"
        )

    return values.reshape(shape)


# Load the FlatBuffer model without importing TensorFlow
model_bytes = MODEL_PATH.read_bytes()
model = tflite.Model.GetRootAsModel(model_bytes, 0)
subgraph = model.Subgraphs(0)

found = set()
quantization = {}

for i in range(subgraph.TensorsLength()):
    tensor = subgraph.Tensors(i)
    name = tensor.Name().decode("utf-8")

    for filename, target_name in TARGETS.items():
        if name != target_name:
            continue

        dtype = (
            np.int8 if filename.endswith("int8")
            else np.int32
        )

        values = read_tensor_data(model, tensor, dtype)
        output_path = OUTPUT_DIR / f"{filename}.npy"
        np.save(output_path, values)

        print(f"\nSaved: {output_path}")
        print(f"  Shape: {values.shape}")
        print(f"  Type:  {values.dtype}")
        print(f"  Size:  {values.nbytes:,} bytes")

        # Save the quantization parameters associated with this tensor.
        q = tensor.Quantization()
        scales = (
            q.ScaleAsNumpy().astype(float).tolist()
            if q and q.ScaleLength() > 0 else []
        )
        zero_points = (
            q.ZeroPointAsNumpy().astype(int).tolist()
            if q and q.ZeroPointLength() > 0 else []
        )

        quantization[filename] = {
            "tensor_name": name,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "scales": scales,
            "zero_points": zero_points,
        }

        print(f"  Scale count: {len(scales)}")
        print(f"  Zero-point count: {len(zero_points)}")

        found.add(filename)

missing = set(TARGETS) - found

if missing:
    print("\nERROR: These tensors were not found:")
    for name in sorted(missing):
        print(f"  {name}")
    raise RuntimeError("Extraction incomplete. Check tensor names.")

# --------------------------------------------------
# NEW: Activation quantization parameters
#
# For each FULLY_CONNECTED layer, record the scale and
# zero point of its input and output activation tensors.
# fc0_output should be the same tensor as fc1_input.
# --------------------------------------------------

activations = {}
fc_index = 0

for op_i in range(subgraph.OperatorsLength()):
    op = subgraph.Operators(op_i)
    code = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
    # If this line errors on your tflite package version,
    # use DeprecatedBuiltinCode() instead of BuiltinCode().
    if code != tflite.BuiltinOperator.FULLY_CONNECTED:
        continue

    for label, idx in [("input", op.Inputs(0)), ("output", op.Outputs(0))]:
        t = subgraph.Tensors(idx)
        q = t.Quantization()
        scales = (
            q.ScaleAsNumpy().astype(float).tolist()
            if q and q.ScaleLength() > 0 else []
        )
        zps = (
            q.ZeroPointAsNumpy().astype(int).tolist()
            if q and q.ZeroPointLength() > 0 else []
        )

        activations[f"fc{fc_index}_{label}"] = {
            "tensor_name": t.Name().decode("utf-8"),
            "scales": scales,
            "zero_points": zps,
        }
        print(
            f"fc{fc_index} {label}: {t.Name().decode('utf-8')}  "
            f"scale={scales}  zero_point={zps}"
        )

    fc_index += 1

quantization["activations"] = activations

metadata_path = OUTPUT_DIR / "quantization_metadata.json"
metadata_path.write_text(json.dumps(quantization, indent=2))

print(f"\nQuantization metadata saved: {metadata_path}")
print("\nAll four quantized parameter arrays extracted successfully.")