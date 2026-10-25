import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = Path("quickdraw_data")
MODEL_PATH = "quickdraw_dense_32.keras"
TFLITE_PATH = "quickdraw_dense_32_int8.tflite"

# Output neuron 0 corresponds to apple, neuron 1 to car, etc.
CLASSES = [
    # Keep these 32 existing classes
    "apple", "tree", "banana", "strawberry", "cake",
    "bicycle", "airplane",
    "chair", "table", "clock", "house", "umbrella",
    "flower", "sun", "star", "cloud",
    "guitar", "mug", "eyeglasses",
    "ladder", "pencil", "candle",
    "rainbow", "lightning", "envelope", "scissors",
    "sailboat", "fork", "door", "mountain",
    "crown", "sword"
]

NUM_CLASSES = len(CLASSES)

SAMPLES_PER_CLASS = 10000
BATCH_SIZE = 128
EPOCHS = 100

# ==========================================
# 2. LOAD DATA
# ==========================================
# Expected files:
# quickdraw_data/cat.npy
# quickdraw_data/dog.npy
# ...
# quickdraw_data/cloud.npy
#
# Each file must contain at least 10,000 images,
# each representing a 28x28 image or 784 pixels.

images = []
labels = []

for class_id, class_name in enumerate(CLASSES):
    path = DATA_DIR / f"{class_name}.npy"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {path}\n"
            f"Download or create the Quick Draw dataset for "
            f"'{class_name}', then save it at this path."
        )

    class_images = np.load(path)

    if len(class_images) < SAMPLES_PER_CLASS:
        raise ValueError(
            f"{path} has only {len(class_images)} samples; "
            f"need at least {SAMPLES_PER_CLASS}."
        )

    class_images = class_images[:SAMPLES_PER_CLASS]

    images.append(class_images)
    labels.append(
        np.full(SAMPLES_PER_CLASS, class_id, dtype=np.int32)
    )

X = np.concatenate(images, axis=0)
y = np.concatenate(labels, axis=0)

# Flatten images and normalize pixels to [0, 1].
X = X.reshape(-1, 784).astype(np.float32) / 255.0

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)
print("Number of classes:", NUM_CLASSES)

# ==========================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# ==========================================
# 80% training, 10% validation, 10% testing.

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))

# ==========================================
# 4. BUILD THE MODEL
# ==========================================

model = keras.Sequential([
    keras.Input(shape=(784,), name="pixels"),

    layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(0.001),
        name="hidden"
    ),

    # 128 output logits
    layers.Dense(
        NUM_CLASSES,
        activation=None,
        name="class_logits"
    )
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# 5. TRAIN
# ==========================================

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=10,
        restore_best_weights=True
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=callbacks
)

model.save(MODEL_PATH)
print(f"\nSaved float model to: {MODEL_PATH}")

# ==========================================
# 6. EVALUATE FLOAT MODEL
# ==========================================

test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

print(f"\nFloat test loss: {test_loss:.4f}")
print(f"Float test accuracy: {test_accuracy * 100:.2f}%")

logits = model.predict(X_test, verbose=0)
float_predictions = np.argmax(logits, axis=1)

print("\nFloat classification report:")
print(classification_report(
    y_test,
    float_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASSES,
    digits=4,
    zero_division=0
))

print("Float confusion matrix:")
print(confusion_matrix(
    y_test,
    float_predictions,
    labels=list(range(NUM_CLASSES))
))

# ==========================================
# 7. CONVERT TO FULL INT8 TFLITE
# ==========================================

def representative_dataset():
    # Use representative samples from training data only.
    rng = np.random.default_rng(SEED)

    for class_id in range(NUM_CLASSES):
        class_indices = np.flatnonzero(y_train == class_id)

        chosen = rng.choice(
            class_indices,
            size=min(200, len(class_indices)),
            replace=False
        )

        for index in chosen:
            yield [X_train[index:index + 1].astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Require integer-only inference operators.
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"\nSaved INT8 model to: {TFLITE_PATH}")
print(f"TFLite file size: {len(tflite_model):,} bytes")
print(f"TFLite file size: {len(tflite_model) / 1024:.2f} KiB")

# ==========================================
# 8. EVALUATE INT8 TFLITE MODEL
# ==========================================

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero_point = input_details["quantization"]
output_scale, output_zero_point = output_details["quantization"]

print("\nINT8 input details:")
print(input_details)

print("\nINT8 output details:")
print(output_details)

if input_scale <= 0 or output_scale <= 0:
    raise ValueError("Invalid TFLite quantization scale.")

int8_predictions = []

for sample in X_test:
    # Convert normalized float pixels to quantized INT8 inputs.
    quantized_input = np.round(
        sample / input_scale + input_zero_point
    )
    quantized_input = np.clip(
        quantized_input, -128, 127
    ).astype(np.int8)

    interpreter.set_tensor(
        input_details["index"],
        quantized_input.reshape(1, 784)
    )

    interpreter.invoke()

    quantized_output = interpreter.get_tensor(
        output_details["index"]
    )[0]

    predicted_class = int(np.argmax(quantized_output))
    int8_predictions.append(predicted_class)

int8_predictions = np.asarray(int8_predictions)
int8_accuracy = np.mean(int8_predictions == y_test)

print(f"\nINT8 test accuracy: {int8_accuracy * 100:.2f}%")

print("\nINT8 classification report:")
print(classification_report(
    y_test,
    int8_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASSES,
    digits=4,
    zero_division=0
))

print("INT8 confusion matrix:")
print(confusion_matrix(
    y_test,
    int8_predictions,
    labels=list(range(NUM_CLASSES))
))
