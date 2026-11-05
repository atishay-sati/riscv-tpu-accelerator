
import tensorflow as tf

MODEL_PATH = "quickdraw_dense_32_int8.tflite"

# Load the model and inspect its quantized tensors.
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

details = interpreter.get_tensor_details()

print("\n=== MODEL INPUTS ===")
for tensor in interpreter.get_input_details():
    print(tensor["name"], tensor["shape"], tensor["dtype"])
    print("Quantization:", tensor["quantization"])

print("\n=== MODEL OUTPUTS ===")
for tensor in interpreter.get_output_details():
    print(tensor["name"], tensor["shape"], tensor["dtype"])
    print("Quantization:", tensor["quantization"])

print("\n=== WEIGHT AND BIAS TENSORS ===")
for tensor in details:
    name = tensor["name"].lower()

    if "weight" in name or "bias" in name or "matmul" in name:
        q = tensor["quantization_parameters"]

        print(f"\nName: {tensor['name']}")
        print("Shape:", tensor["shape"])
        print("Dtype:", tensor["dtype"])
        print("Scales count:", len(q["scales"]))
        print("Zero points count:", len(q["zero_points"]))
        print("Quantized dimension:", q["quantized_dimension"])

# Show the actual operators used by the model.
print("\n=== MODEL OPERATORS ===")

for op in interpreter._get_ops_details():
    print("Operator:", op["op_name"])
    print("Inputs:", op["inputs"])
    print("Outputs:", op["outputs"])

print("\nInspection complete.")
