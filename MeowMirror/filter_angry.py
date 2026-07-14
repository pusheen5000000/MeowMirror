import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ANNOTATION_FILE = SCRIPT_DIR / "Selfie-dataset" / "selfie_dataset.txt"
IMAGES_DIR = SCRIPT_DIR / "Selfie-dataset" / "images"
OUTPUT_DIR = SCRIPT_DIR / "data" / "angry"

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

MOUTH_OPEN_INDEX = ATTRIBUTES.index("mouth_open")
SMILING_INDEX = ATTRIBUTES.index("smiling")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

copied = 0
skipped_missing = 0

with open(ANNOTATION_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2 + len(ATTRIBUTES):
            continue

        image_name = parts[0]
        attr_values = parts[2:2 + len(ATTRIBUTES)]

        mouth_open = attr_values[MOUTH_OPEN_INDEX]
        smiling = attr_values[SMILING_INDEX]

        # angry = mouth open + frowning + NOT smiling
        if mouth_open == "1" and smiling == "-1":
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

print(f"Copied {copied} angry (open-mouth) images to {OUTPUT_DIR}")
print(f"Skipped {skipped_missing} (image file not found)")