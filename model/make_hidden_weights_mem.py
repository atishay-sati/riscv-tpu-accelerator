import numpy as np
from pathlib import Path

folder = Path("fpga_params")
weights = np.load(folder / "hidden_weights_int8.npy")

assert weights.shape == (128, 784), f"Unexpected shape: {weights.shape}"
assert weights.dtype == np.int8, f"Unexpected dtype: {weights.dtype}"

# Convert signed INT8 values to their 8-bit hexadecimal bit patterns.
# For example, -13 becomes F3, and 18 becomes 12.
unsigned = weights.reshape(-1).view(np.uint8)
lines = [f"{int(value):02X}" for value in unsigned]

output_path = folder / "hidden_weights.mem"
output_path.write_text("\n".join(lines) + "\n", encoding="ascii")

# Read back the generated file and verify every entry.
mem = np.array(
    [int(line, 16) for line in output_path.read_text().splitlines()],
    dtype=np.uint8
)
recovered = mem.view(np.int8).reshape(weights.shape)

print("Generated:", output_path)
print("Entries:", len(lines))
print("Nonzero weights:", int(np.count_nonzero(weights)))
print("Exact match:", np.array_equal(weights, recovered))
print("Neuron 0, weights 28-39:", weights[0, 28:40].tolist())
print("Neuron 0, MEM 28-39:", lines[28:40])
