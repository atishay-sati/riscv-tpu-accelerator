
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError

DATA_DIR = Path("quickdraw_data")
DATA_DIR.mkdir(exist_ok=True)

# All 32 classes used by model.py.
# Keep this order consistent with the model's CLASSES list.
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

BASE_URL = (
    "https://storage.googleapis.com/quickdraw_dataset/"
    "full/numpy_bitmap/{}.npy"
)

failed = []

for name in CLASSES:
    destination = DATA_DIR / f"{name}.npy"

    if destination.exists():
        print(f"Already exists: {name}")
        continue

    url = BASE_URL.format(name)

    try:
        print(f"Downloading {name}...")
        urlretrieve(url, destination)
        print(f"Saved: {destination}")

    except (HTTPError, URLError, OSError) as error:
        # Remove an incomplete download, if one was created.
        if destination.exists():
            destination.unlink()

        failed.append(name)
        print(f"FAILED: {name}: {error}")

print("\nDownload process finished.")
print(f"Available files: {len(list(DATA_DIR.glob('*.npy')))}/{len(CLASSES)}")

if failed:
    print("Failed classes:", ", ".join(failed))
    print("Run this script again after checking your internet connection.")
else:
    print("All 32 classes are available.")
