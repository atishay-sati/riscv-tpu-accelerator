import numpy as np
from pathlib import Path

folder = Path("fpga_params")
weights = np.load(folder / "output_weights_int8.npy")

print("Shape:", weights.shape)
print("Dtype:", weights.dtype)

assert weights.shape == (32, 128), f"Unexpected shape: {weights.shape}"
assert weights.dtype == np.int8, f"Unexpected dtype: {weights.dtype}"

# Preserve each signed INT8 value's exact 8-bit pattern.
unsigned = weights.reshape(-1).view(np.uint8)
lines = [f"{int(value):02X}" for value in unsigned]

path = folder / "output_weights.mem"
path.write_text("\n".join(lines) + "\n", encoding="ascii")

# Verify the saved file against the original array.
mem = np.array(
    [int(line, 16) for line in path.read_text().splitlines()],
    dtype=np.uint8
)
recovered = mem.view(np.int8).reshape(weights.shape)

print("Entries:", len(lines))
print("Nonzero weights:", int(np.count_nonzero(weights)))
print("Exact match:", np.array_equal(weights, recovered))
print("First 12 weights:", weights[0, :12].tolist())
print("First 12 MEM entries:", lines[:12])
