import cv2
import mediapipe as mp
import os
import time

# ---- CONFIG ----
LABEL = "tongue_out"   # change this each time you run the script (e.g. "confused")
SAVE_DIR = f"data/{LABEL}"
TARGET_COUNT = 200      # how many images to collect
CAPTURE_MODE = "space"  # "space" = press spacebar to capture, "auto" = auto-capture every second
IMG_SIZE = 48           # resize saved images to IMG_SIZE x IMG_SIZE (use 48 to match FER2013 exactly)

os.makedirs(SAVE_DIR, exist_ok=True)

face_detection = mp.solutions.face_detection.FaceDetection()
cap = cv2.VideoCapture(0)

count = len(os.listdir(SAVE_DIR))  # resume from existing count if you stopped partway
last_capture_time = 0

print(f"Collecting images for label: '{LABEL}'")
print(f"Already have {count} images in {SAVE_DIR}")
if CAPTURE_MODE == "space":
    print("Press SPACE to capture a face, press 'q' to quit.")
else:
    print("Auto-capturing every 1 second. Press 'q' to quit.")

while count < TARGET_COUNT:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    face_crop = None

    if results.detections:
        detection = results.detections[0]  # just use the first face found
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape

        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        width = int(bbox.width * w)
        height = int(bbox.height * h)

        # Draw box for visual feedback
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Crop the face region (clamp to frame bounds)
        face_crop = frame[y:y + height, x:x + width]

    # Show status on screen
    cv2.putText(frame, f"Label: {LABEL} | Count: {count}/{TARGET_COUNT}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    should_capture = False
    if CAPTURE_MODE == "space" and key == ord(" "):
        should_capture = True
    elif CAPTURE_MODE == "auto" and time.time() - last_capture_time >= 1.0:
        should_capture = True

    if should_capture and face_crop is not None and face_crop.size > 0:
        # Convert to grayscale (to match FER2013) and resize to a consistent size
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

        filename = os.path.join(SAVE_DIR, f"{LABEL}_{count}.jpg")
        cv2.imwrite(filename, resized)
        count += 1
        last_capture_time = time.time()
        print(f"Saved {filename}")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done. Collected {count} images in {SAVE_DIR}")