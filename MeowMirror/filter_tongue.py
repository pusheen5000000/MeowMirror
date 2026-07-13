import shutil
from pathlib import Path

ANNOTATION_FILE = "Selfie-dataset/selfie_dataset.txt"
IMAGES_DIR = Path("Selfie-dataset/images")
OUTPUT_DIR = Path("data/tongue_out")

ATTRIBUTES = [
    "partial_faces", "is_female", "baby", "child", "teenager", "youth",
    "middle_age", "senior", "white", "black", "asian", "oval_face",
    "round_face", "heart_face", "smiling", "mouth_open", "frowning",
    "wearing_glasses", "wearing_sunglasses", "wearing_lipstick",
    "tongue_out", "duck_face", "black_hair", "blond_hair", "brown_hair",
    "red_hair", "curly_hair", "straight_hair", "braid_hair",
    "showing_cellphone", "using_earphone", "using_mirror", "braces",
    "wearing_hat", "harsh_lighting", "dim_lighting"
]

TONGUE_OUT_INDEX = ATTRIBUTES.index("tongue_out")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

copied = 0
skipped_missing = 0

with open(ANNOTATION_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2 + len(ATTRIBUTES):
            continue  # skip header/malformed lines

        image_name = parts[0]
        attr_values = parts[2:2 + len(ATTRIBUTES)]  # parts[1] is popularity score

        tongue_val = attr_values[TONGUE_OUT_INDEX]

        if tongue_val == "1":   # 1 = tongue out; -1 = not
            src = IMAGES_DIR / f"{image_name}.jpg"
            if not src.exists():
                for ext in [".jpeg", ".png"]:
                    alt = IMAGES_DIR / f"{image_name}{ext}"
                    if alt.exists():
                        src = alt
                        break

            if src.exists():
                shutil.copy(src, OUTPUT_DIR / src.name)
                copied += 1
            else:
                skipped_missing += 1

print(f"Copied {copied} tongue-out images to {OUTPUT_DIR}")
print(f"Skipped {skipped_missing} (image file not found)")