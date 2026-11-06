import os



# Reduce TensorFlow startup messages.

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"



import tkinter as tk

from tkinter import messagebox

from pathlib import Path

import time

import serial



import numpy as np

from PIL import Image, ImageDraw

import tensorflow as tf



INPUT_FILE = Path(

    r"C:\Users\Atish\VerliogProjects\TPU_Integration"

    r"\mem_files\dense_layer_test\input_pixels.mem"

)



with open(INPUT_FILE) as f:

    pixels = bytes(

        int(line.strip(), 16)

        for line in f

        if line.strip()

    )



# ============================================================

# CONFIGURATION

# ============================================================



MODEL_PATH = Path("quickdraw_dense_32_int8.tflite")



# Basys 3 USB-UART settings.

SERIAL_PORT = "COM6"

BAUD_RATE = 115200

IMAGE_SIZE = 28 * 28
# UART uses 8 data bits, no parity, and 1 stop bit (10 bits per byte).
# One transaction sends 785 bytes (header + 784 pixels) and receives 1 byte.
UART_BYTES_PER_TRANSACTION = 1 + IMAGE_SIZE + 1
UART_BITS_PER_BYTE = 10
UART_WIRE_TIME_MS = (
    UART_BYTES_PER_TRANSACTION * UART_BITS_PER_BYTE
    / BAUD_RATE * 1000
)



# FPGA packet protocol.

PACKET_HEADER = 0xAA

IMAGE_SIZE = 28 * 28

# UART uses 8 data bits, no parity, and 1 stop bit (10 bits per byte).
# A transaction sends 785 bytes (header + 784 pixels) and receives 1 byte.
UART_BYTES_PER_TRANSACTION = 1 + IMAGE_SIZE + 1
UART_BITS_PER_BYTE = 10
UART_WIRE_TIME_MS = (
    UART_BYTES_PER_TRANSACTION * UART_BITS_PER_BYTE
    / BAUD_RATE * 1000
)



# Drawing interface settings.

CANVAS_SIZE = 280

MODEL_SIZE = 28

BRUSH_WIDTH = 10



# Wait up to 30 seconds for the FPGA prediction.

UART_TIMEOUT = 30



# False: black background with white strokes.

# Set True if your training images used white backgrounds with black strokes.

INVERT_IMAGE = False





# ============================================================

# CLASS LABELS

# ============================================================



# The ordering MUST match the ordering used during training.

# The FPGA returns a class index from 0 through 31.

def load_labels():

    return [

        "apple", "tree", "banana", "strawberry", "cake",

        "bicycle", "airplane", "chair", "table", "clock",

        "house", "umbrella", "flower", "sun", "star", "cloud",

        "guitar", "mug", "eyeglasses", "ladder", "pencil",

        "candle", "rainbow", "lightning", "envelope", "scissors",

        "sailboat", "fork", "door", "mountain", "crown", "sword"

    ]





# ============================================================

# QUICK DRAW APPLICATION

# ============================================================



class QuickDrawApp:



    def __init__(self, root):



        self.root = root

        root.title("Quick, Draw! FPGA TPU")

        root.resizable(False, False)



        # ----------------------------------------------------

        # Load the TFLite model.

        # It provides the input quantization parameters and

        # runs local reference inference for comparison.

        # ----------------------------------------------------



        if not MODEL_PATH.exists():

            messagebox.showerror(

                "Model not found",

                f"Could not find {MODEL_PATH.resolve()}\n\n"

                "Keep the .tflite model in the same folder "

                "as this Python script."

            )

            root.destroy()

            return



        self.labels = load_labels()



        self.interpreter = tf.lite.Interpreter(

            model_path=str(MODEL_PATH)

        )

        self.interpreter.allocate_tensors()



        self.input_info = self.interpreter.get_input_details()[0]

        self.output_info = self.interpreter.get_output_details()[0]



        # ----------------------------------------------------

        # Check that the loaded model matches the FPGA design.

        # ----------------------------------------------------



        input_shape = tuple(self.input_info["shape"])

        output_shape = tuple(self.output_info["shape"])



        if input_shape != (1, IMAGE_SIZE):

            messagebox.showerror(

                "Unexpected input shape",

                f"Expected (1, {IMAGE_SIZE}), but model uses {input_shape}."

            )

            root.destroy()

            return



        if output_shape != (1, len(self.labels)):

            messagebox.showerror(

                "Unexpected output shape",

                f"Expected (1, {len(self.labels)}), "

                f"but model uses {output_shape}."

            )

            root.destroy()

            return



        # The FPGA interface is designed for INT8 model inputs

        # and INT8 output scores.

        if self.input_info["dtype"] != np.int8:

            messagebox.showerror(

                "Unexpected input type",

                f"Expected INT8 input, but model uses "

                f"{self.input_info['dtype']}."

            )

            root.destroy()

            return



        if self.output_info["dtype"] != np.int8:

            messagebox.showerror(

                "Unexpected output type",

                f"Expected INT8 output, but model uses "

                f"{self.output_info['dtype']}."

            )

            root.destroy()

            return



        # These parameters convert normalized pixels into INT8.

        self.input_scale, self.input_zero_point = (

            self.input_info["quantization"]

        )



        if self.input_scale <= 0:

            messagebox.showerror(

                "Invalid quantization",

                "The model has an invalid input quantization scale."

            )

            root.destroy()

            return



        # ----------------------------------------------------

        # Create the drawing image.

        # ----------------------------------------------------



        self.image = Image.new(

            "L",

            (CANVAS_SIZE, CANVAS_SIZE),

            0

        )



        self.draw = ImageDraw.Draw(self.image)



        tk.Label(

            root,

            text="Draw one object, then click Predict.",

            padx=10,

            pady=8

        ).pack()



        # ----------------------------------------------------

        # Drawing canvas.

        # ----------------------------------------------------



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



        # ----------------------------------------------------

        # Buttons.

        # ----------------------------------------------------



        buttons = tk.Frame(root)

        buttons.pack(pady=8)



        tk.Button(

            buttons,

            text="Predict",

            width=12,

            command=self.predict

        ).pack(side=tk.LEFT, padx=5)



        tk.Button(

            buttons,

            text="Clear",

            width=12,

            command=self.clear

        ).pack(side=tk.LEFT, padx=5)



        # ----------------------------------------------------

        # Prediction/status display.

        # ----------------------------------------------------



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

            text="Compare local TensorFlow inference against FPGA inference.",

            wraplength=CANVAS_SIZE,

            justify=tk.LEFT,

            fg="#555555",

            padx=12,

            pady=6

        ).pack()





    # ========================================================

    # DRAWING FUNCTIONS

    # ========================================================



    def start_stroke(self, event):



        self.last_x = event.x

        self.last_y = event.y



        radius = BRUSH_WIDTH // 2



        self.canvas.create_oval(

            event.x - radius,

            event.y - radius,

            event.x + radius,

            event.y + radius,

            fill="white",

            outline="white"

        )



        self.draw.ellipse(

            (

                event.x - radius,

                event.y - radius,

                event.x + radius,

                event.y + radius

            ),

            fill=255

        )





    def paint(self, event):



        if self.last_x is None:

            self.start_stroke(event)

            return



        self.canvas.create_line(

            self.last_x,

            self.last_y,

            event.x,

            event.y,

            fill="white",

            width=BRUSH_WIDTH,

            capstyle=tk.ROUND,

            smooth=True

        )



        self.draw.line(

            (

                self.last_x,

                self.last_y,

                event.x,

                event.y

            ),

            fill=255,

            width=BRUSH_WIDTH

        )



        radius = BRUSH_WIDTH / 2



        self.draw.ellipse(

            (

                event.x - radius,

                event.y - radius,

                event.x + radius,

                event.y + radius

            ),

            fill=255

        )



        self.last_x = event.x

        self.last_y = event.y





    def end_stroke(self, _event):



        self.last_x = None

        self.last_y = None





    def clear(self):



        self.canvas.delete("all")



        self.image = Image.new(

            "L",

            (CANVAS_SIZE, CANVAS_SIZE),

            0

        )



        self.draw = ImageDraw.Draw(self.image)



        self.status.config(

            text="Canvas cleared. Draw something."

        )





    # ========================================================

    # IMAGE PREPROCESSING

    # ========================================================



    def prepare_input(self):



        # Resize the 280x280 drawing into the model's 28x28 input.

        small = self.image.resize(

            (MODEL_SIZE, MODEL_SIZE),

            Image.Resampling.LANCZOS

        )



        # Convert pixel values from 0-255 into floating-point 0-1.

        pixels = np.asarray(

            small,

            dtype=np.float32

        ) / 255.0



        if INVERT_IMAGE:

            pixels = 1.0 - pixels



        # Apply the same INT8 quantization used by the trained model.

        quantized = np.rint(

            pixels / float(self.input_scale)

            + int(self.input_zero_point)

        )



        quantized = np.clip(

            quantized,

            -128,

            127

        ).astype(np.int8)



        return quantized.reshape(IMAGE_SIZE)





    # ========================================================

    # CONVERT MODEL INPUT INTO THE FORMAT THE FPGA EXPECTS

    # ========================================================



    def to_fpga_bytes(self, pixels):



        # TensorFlow Lite computes weight * (q - input_zero_point).

        # The TPU stores the already-adjusted unsigned activation

        # (q - input_zero_point).

        #

        # For this model, the zero point is -128, so this converts

        # signed INT8 values into unsigned bytes from 0 to 255.

        adjusted = (

            pixels.astype(np.int16)

            - int(self.input_zero_point)

        )



        # Ensure that every adjusted activation fits in one byte.

        if adjusted.min() < 0 or adjusted.max() > 255:

            raise ValueError(

                "Adjusted pixels fall outside 0-255: "

                f"min={adjusted.min()}, max={adjusted.max()}"

            )



        return adjusted.astype(np.uint8)





    # ========================================================

    # LOCAL TFLITE REFERENCE INFERENCE

    # ========================================================



    def run_local_inference(self, pixels):



        # TFLite expects the original signed INT8 values,

        # not the adjusted unsigned bytes sent to the FPGA.

        model_input = pixels.reshape(1, IMAGE_SIZE)



        self.interpreter.set_tensor(

            self.input_info["index"],

            model_input

        )



        self.interpreter.invoke()



        # Retrieve the 32 raw INT8 output scores.

        local_output = self.interpreter.get_tensor(

            self.output_info["index"]

        ).reshape(-1)



        # The output uses one shared quantization scale and

        # zero point, so the largest raw score has the same

        # class index as the largest dequantized score.

        return int(np.argmax(local_output))





    # ========================================================

    # UART COMMUNICATION

    # ========================================================



    def send_to_fpga(self, fpga_pixels):



        with serial.Serial(

            port=SERIAL_PORT,

            baudrate=BAUD_RATE,

            timeout=UART_TIMEOUT,

            write_timeout=5

        ) as ser:



            # Give the serial connection a moment to settle.

            time.sleep(0.1)



            # Discard stale incoming bytes from an earlier transaction.

            ser.reset_input_buffer()



            # Convert the 784 adjusted unsigned pixels into bytes.

            pixel_bytes = fpga_pixels.tobytes()



            # Packet:

            # Byte 0: 0xAA header

            # Bytes 1-784: adjusted image pixels

            packet = bytes([PACKET_HEADER]) + pixel_bytes



            # Start timing immediately before transmission. This excludes
            # serial-port opening, the 100 ms settling delay, buffer clearing,
            # and packet construction.
            transaction_start = time.perf_counter()

            bytes_sent = ser.write(packet)



            if bytes_sent != len(packet):

                raise RuntimeError(

                    f"Only sent {bytes_sent} of {len(packet)} packet bytes."

                )



            # Wait until the packet has finished transmitting.

            ser.flush()



            # Wait for one prediction byte from the FPGA.

            result = ser.read(1)



            if len(result) != 1:

                raise TimeoutError(

                    "No prediction received from FPGA. "

                    "Check the board and UART connection."

                )



            # The FPGA sends the class index in bits [4:0].

            predicted_class = result[0] & 0x1F



            # Includes UART transfer, FPGA processing, and USB/driver overhead.
            transaction_time_ms = (
                time.perf_counter() - transaction_start
            ) * 1000.0

            return predicted_class, transaction_time_ms





    # ========================================================

    # FPGA + LOCAL PREDICTION COMPARISON

    # ========================================================



    def predict(self):



        try:



            self.status.config(

                text="Preparing image and running predictions..."

            )



            self.root.update_idletasks()



            # ------------------------------------------------

            # STEP 1: Preprocess the drawing.

            # ------------------------------------------------



            # Signed INT8 values for local TFLite inference.

            pixels = self.prepare_input()



            # Adjusted unsigned bytes for the FPGA.

            fpga_pixels = self.to_fpga_bytes(pixels)



            # ------------------------------------------------

            # STEP 2: Run local TensorFlow Lite inference.

            # ------------------------------------------------



            cpu_start = time.perf_counter()
            local_class = self.run_local_inference(pixels)
            cpu_time_ms = (time.perf_counter() - cpu_start) * 1000.0



            # ------------------------------------------------

            # STEP 3: Send the adjusted pixels to the FPGA.

            # ------------------------------------------------



            fpga_class, fpga_transaction_ms = self.send_to_fpga(fpga_pixels)

            # Estimate compute time by removing the theoretical UART wire
            # time. This is not an exact hardware measurement because USB,
            # serial-driver, and Python overhead are still included.
            fpga_inference_estimate_ms = max(
                0.0,
                fpga_transaction_ms - UART_WIRE_TIME_MS
            )



            if fpga_class >= len(self.labels):

                raise ValueError(

                    f"FPGA returned invalid class ID: {fpga_class}"

                )



            # ------------------------------------------------

            # STEP 4: Compare both predictions.

            # ------------------------------------------------



            local_label = self.labels[local_class]

            fpga_label = self.labels[fpga_class]



            if local_class == fpga_class:

                comparison = "MATCH"

            else:

                comparison = "MISMATCH"



            # Keep a concise terminal log for debugging.

            print(

                f"Local: {local_label} ({local_class}) | "

                f"FPGA: {fpga_label} ({fpga_class}) | "

                f"{comparison} | "
                f"CPU: {cpu_time_ms:.3f} ms | "
                f"FPGA transaction: {fpga_transaction_ms:.3f} ms | "
                f"FPGA inference estimate: {fpga_inference_estimate_ms:.3f} ms"

            )



            # ------------------------------------------------

            # STEP 5: Display both predictions in the GUI.

            # ------------------------------------------------



            self.status.config(

                text=(

                    f"Local TFLite: {local_label} (class {local_class})\n"

                    f"FPGA TPU:     {fpga_label} (class {fpga_class})\n"

                    f"Comparison:   {comparison}\n"
                    f"CPU TFLite:   {cpu_time_ms:.3f} ms\n"
                    f"FPGA total:   {fpga_transaction_ms:.3f} ms\n"
                    f"UART estimate:{UART_WIRE_TIME_MS:.3f} ms\n"
                    f"FPGA compute: ~{fpga_inference_estimate_ms:.3f} ms"

                )

            )



        except Exception as exc:



            self.status.config(

                text="Prediction failed."

            )



            messagebox.showerror(

                "FPGA Prediction Failed",

                str(exc)

            )





# ============================================================

# MAIN PROGRAM

# ============================================================



if __name__ == "__main__":



    root = tk.Tk()



    app = QuickDrawApp(root)



    if root.winfo_exists():

        root.mainloop()
