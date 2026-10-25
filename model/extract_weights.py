import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import numpy as np
from pathlib import Path

# Output directory
output_dir = Path("fpga_params")
output_dir.mkdir(exist_ok=True)

# Load trained model
model = tf.keras.models.load_model("quickdraw_dense_32.keras")

# Find the two Dense layers
dense_layers = [
    layer for layer in model.layers
    if isinstance(layer, tf.keras.layers.Dense)
]

assert len(dense_layers) == 2, (
    f"Expected 2 Dense layers, found {len(dense_layers)}"
)

hidden, output = dense_layers

# Extract and save weights and biases
np.save(
    output_dir / "hidden_weights_float.npy",
    hidden.get_weights()[0]
)
np.save(
    output_dir / "hidden_biases_float.npy",
    hidden.get_weights()[1]
)
np.save(
    output_dir / "output_weights_float.npy",
    output.get_weights()[0]
)
np.save(
    output_dir / "output_biases_float.npy",
    output.get_weights()[1]
)

# Confirm saved files
print("Saved parameter files to:", output_dir.resolve())

for path in sorted(output_dir.glob("*.npy")):
    data = np.load(path)
    print(
        f"{path.name}: "
        f"shape={data.shape}, dtype={data.dtype}"
    )

print("\nWeight extraction complete.")
