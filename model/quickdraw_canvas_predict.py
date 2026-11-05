import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Reduce TensorFlow startup messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tkinter as tk
from tkinter import messagebox
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

MODEL_PATH = Path("quickdraw_dense_32_int8.tflite")
LABELS_PATH = Path("labels.txt")
CANVAS_SIZE = 280
MODEL_SIZE = 28
BRUSH_WIDTH = 10
TOP_K = 5

# False: black background with white strokes.
# Set True if your training images used white backgrounds with black strokes.
INVERT_IMAGE = False


def load_labels():
    # Keep this order exactly the same as the order used during training.
    return [
        "apple", "tree", "banana", "strawberry", "cake",
        "bicycle", "airplane", "chair", "table", "clock",
        "house", "umbrella", "flower", "sun", "star", "cloud",
        "guitar", "mug", "eyeglasses", "ladder", "pencil",
        "candle", "rainbow", "lightning", "envelope", "scissors",
        "sailboat", "fork", "door", "mountain", "crown", "sword"
    ]


class QuickDrawApp:
    def __init__(self, root):
        self.root = root
        root.title("Quick, Draw! Model Tester")
        root.resizable(False, False)

        if not MODEL_PATH.exists():
            messagebox.showerror(
                "Model not found",
                f"Could not find {MODEL_PATH.resolve()}\n\n"
                "Save this script in the same folder as your .tflite model."
            )
            root.destroy()
            return

        self.labels = load_labels()
        self.interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
        self.interpreter.allocate_tensors()
        self.input_info = self.interpreter.get_input_details()[0]
        self.output_info = self.interpreter.get_output_details()[0]

        input_shape = tuple(self.input_info["shape"])
        if input_shape != (1, 784):
            messagebox.showerror(
                "Unexpected input shape",
                f"This script expects (1, 784), but the model uses {input_shape}."
            )
            root.destroy()
            return

        self.input_scale, self.input_zero_point = self.input_info["quantization"]
        self.output_scale, self.output_zero_point = self.output_info["quantization"]

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)

        tk.Label(
            root,
            text="Draw one object, then click Predict.",
            padx=10,
            pady=8
        ).pack()

        self.canvas = tk.Canvas(
            root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            cursor="cross",
            highlightthickness=1
        )
        self.canvas.pack(padx=12, pady=4)

        self.last_x = None
        self.last_y = None
        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)

        buttons = tk.Frame(root)
        buttons.pack(pady=8)
        tk.Button(buttons, text="Predict", width=12, command=self.predict).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(buttons, text="Clear", width=12, command=self.clear).pack(
            side=tk.LEFT, padx=5
        )

        self.status = tk.Label(
            root,
            text="Draw something to begin.",
            justify=tk.LEFT,
            anchor="w",
            font=("Consolas", 10),
            padx=12,
            pady=8
        )
        self.status.pack(fill=tk.X)

        tk.Label(
            root,
            text="If predictions look wrong, verify the background/stroke colors "
                 "and preprocessing match your training images.",
            wraplength=CANVAS_SIZE,
            justify=tk.LEFT,
            fg="#555555",
            padx=12,
            pady=6
        ).pack()

    def start_stroke(self, event):
        self.last_x, self.last_y = event.x, event.y
        radius = BRUSH_WIDTH // 2
        self.canvas.create_oval(
            event.x-radius, event.y-radius, event.x+radius, event.y+radius,
            fill="white", outline="white"
        )
        self.draw.ellipse(
            (event.x-radius, event.y-radius, event.x+radius, event.y+radius),
            fill=255
        )

    def paint(self, event):
        if self.last_x is None:
            self.start_stroke(event)
            return

        self.canvas.create_line(
            self.last_x, self.last_y, event.x, event.y,
            fill="white", width=BRUSH_WIDTH, capstyle=tk.ROUND, smooth=True
        )
        self.draw.line(
            (self.last_x, self.last_y, event.x, event.y),
            fill=255, width=BRUSH_WIDTH
        )
        radius = BRUSH_WIDTH / 2
        self.draw.ellipse(
            (event.x-radius, event.y-radius, event.x+radius, event.y+radius),
            fill=255
        )
        self.last_x, self.last_y = event.x, event.y

    def end_stroke(self, _event):
        self.last_x = None
        self.last_y = None

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.status.config(text="Canvas cleared. Draw something.")

    def prepare_input(self):
        small = self.image.resize(
            (MODEL_SIZE, MODEL_SIZE),
            Image.Resampling.LANCZOS
        )
        pixels = np.asarray(small, dtype=np.float32) / 255.0

        if INVERT_IMAGE:
            pixels = 1.0 - pixels

        quantized = np.rint(
            pixels / float(self.input_scale) + int(self.input_zero_point)
        )
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
        return quantized.reshape(1, 784)

    def predict(self):
        try:
            model_input = self.prepare_input()
            self.interpreter.set_tensor(self.input_info["index"], model_input)
            self.interpreter.invoke()
            output_q = self.interpreter.get_tensor(
                self.output_info["index"]
            ).reshape(-1)

            scores = (
                output_q.astype(np.float32) - int(self.output_zero_point)
            ) * float(self.output_scale)

            top_indices = np.argsort(scores)[::-1][:TOP_K]
            lines = ["Top predictions (score is dequantized output):"]
            for rank, index in enumerate(top_indices, start=1):
                lines.append(
                    f"{rank}. {self.labels[index]} "
                    f"(class {index}, score {scores[index]:.3f})"
                )
            self.status.config(text="\n".join(lines))
        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc))


if __name__ == "__main__":
    root = tk.Tk()
    app = QuickDrawApp(root)
    if root.winfo_exists():
        root.mainloop()
