
import numpy as np
from pathlib import Path
import tflite

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = Path("quickdraw_dense_32_int8.tflite")
PARAM_DIR = Path("fpga_params")

TARGETS = {
    "hidden_weights_int8": "sequential_1/hidden_1/MatMul",
    "hidden_biases_int32":
        "sequential_1/hidden_1/Relu;sequential_1/hidden_1/BiasAdd",
    "output_weights_int8": "sequential_1/class_logits_1/MatMul",
    "output_biases_int32": "sequential_1/class_logits_1/BiasAdd",
}

DTYPES = {
    "hidden_weights_int8": np.int8,
    "hidden_biases_int32": np.int32,
    "output_weights_int8": np.int8,
    "output_biases_int32": np.int32,
}


def read_tensor(model, tensor, dtype):
    """Independently read a constant tensor from the TFLite buffer."""
    shape = [
        tensor.Shape(i)
        for i in range(tensor.ShapeLength())
    ]

    buffer = model.Buffers(tensor.Buffer())
    raw = buffer.DataAsNumpy()

    if raw is None or len(raw) == 0:
        raise ValueError(
            f"Tensor {tensor.Name().decode()} has no constant data"
        )

    values = np.frombuffer(raw.tobytes(), dtype=dtype).copy()

    if values.size != int(np.prod(shape)):
        raise ValueError(
            f"Incorrect buffer size for {tensor.Name().decode()}"
        )

    return values.reshape(shape)


def get_quantization(tensor):
    """Return a tensor's quantization scales and zero points."""
    q = tensor.Quantization()

    scales = (
        q.ScaleAsNumpy().astype(float).tolist()
        if q and q.ScaleLength() > 0 else []
    )

    zero_points = (
        q.ZeroPointAsNumpy().astype(int).tolist()
        if q and q.ZeroPointLength() > 0 else []
    )

    return scales, zero_points


def main():
    # Check required files exist.
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    if not PARAM_DIR.is_dir():
        raise FileNotFoundError(f"Missing directory: {PARAM_DIR}")

    model_bytes = MODEL_PATH.read_bytes()
    model = tflite.Model.GetRootAsModel(model_bytes, 0)
    subgraph = model.Subgraphs(0)

    # Find the original tensors by exact name.
    tensors_by_name = {}

    for i in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(i)
        name = tensor.Name().decode("utf-8")
        tensors_by_name[name] = tensor

    errors = 0

    print("VALIDATING EXTRACTED PARAMETERS")
    print("=" * 55)

    for filename, tensor_name in TARGETS.items():
        print(f"\nChecking {filename}...")

        extracted_path = PARAM_DIR / f"{filename}.npy"

        if not extracted_path.is_file():
            print(f"  FAIL: Missing {extracted_path}")
            errors += 1
            continue

        if tensor_name not in tensors_by_name:
            print(f"  FAIL: Tensor not found in model: {tensor_name}")
            errors += 1
            continue

        original_tensor = tensors_by_name[tensor_name]

        # Read original values directly from the TFLite file.
        original = read_tensor(
            model, original_tensor, DTYPES[filename]
        )

        # Load the extraction script's saved values.
        extracted = np.load(extracted_path, allow_pickle=False)

        checks_passed = True

        if extracted.dtype != original.dtype:
            print(
                f"  FAIL: dtype mismatch: "
                f"{extracted.dtype} vs {original.dtype}"
            )
            checks_passed = False

        if extracted.shape != original.shape:
            print(
                f"  FAIL: shape mismatch: "
                f"{extracted.shape} vs {original.shape}"
            )
            checks_passed = False
        elif not np.array_equal(extracted, original):
            mismatch_count = int(np.count_nonzero(extracted != original))
            print(f"  FAIL: {mismatch_count} values differ")
            checks_passed = False

        # Check quantization metadata saved by the extractor.
        metadata_path = PARAM_DIR / "quantization_metadata.json"

        if not metadata_path.is_file():
            print("  FAIL: Missing quantization_metadata.json")
            checks_passed = False
        else:
            import json

            metadata = json.loads(metadata_path.read_text())

            if filename not in metadata:
                print("  FAIL: Missing tensor metadata")
                checks_passed = False
            else:
                saved_meta = metadata[filename]
                original_scales, original_zps = get_quantization(
                    original_tensor
                )

                if saved_meta.get("tensor_name") != tensor_name:
                    print("  FAIL: Tensor name in metadata differs")
                    checks_passed = False

                if saved_meta.get("shape") != list(original.shape):
                    print("  FAIL: Metadata shape differs")
                    checks_passed = False

                if saved_meta.get("dtype") != str(original.dtype):
                    print("  FAIL: Metadata dtype differs")
                    checks_passed = False

                if not np.allclose(
                    saved_meta.get("scales", []),
                    original_scales,
                    rtol=1e-6,
                    atol=0,
                ):
                    print("  FAIL: Quantization scales differ")
                    checks_passed = False

                if saved_meta.get("zero_points", []) != original_zps:
                    print("  FAIL: Quantization zero points differ")
                    checks_passed = False

        if checks_passed:
            print("  PASS: Values, shape, dtype, and metadata match")
            print(f"  Shape: {original.shape}")
            print(f"  Values checked: {original.size:,}")
            print(f"  Bytes: {original.nbytes:,}")
        else:
            errors += 1

    print("\n" + "=" * 55)

    if errors == 0:
        print("SUCCESS: All four extracted tensors passed validation.")
    else:
        print(f"FAILED: {errors} tensor(s) had validation errors.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
