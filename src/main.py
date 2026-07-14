import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from pathlib import Path

model = tf.keras.models.load_model('src/emotion_model.keras')

face_detection = mp.solutions.face_detection.FaceDetection()

CLASS_NAMES = ["angry", "happy", "neutral", "sad"]
NEUTRAL_INDEX = CLASS_NAMES.index("neutral")
NEUTRAL_PENALTY = 0.15  # tweak this — higher = neutral shows up less
ANGRY_INDEX = CLASS_NAMES.index("angry")
ANGRY_BOOST = 0.15  # tweak this — higher = angry shows up more

SCRIPT_DIR = Path(__file__).resolve().parent
CAT_DIR = SCRIPT_DIR / "cat_faces"

PANEL_SIZE = 480  # each square panel (camera / cat) is PANEL_SIZE x PANEL_SIZE

cat_images = {}
for name in CLASS_NAMES:
    img = cv2.imread(str(CAT_DIR / f"{name}.jpg"))
    if img is None:
        print(f"Couldn't load: {CAT_DIR / f'{name}.jpg'}")
    cat_images[name] = cv2.resize(img, (PANEL_SIZE, PANEL_SIZE))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    emotion_label = "neutral"  # default if no face detected yet

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            x, y = max(0, x), max(0, y)
            width, height = min(w - x, width), min(h - y, height)

            if width > 0 and height > 0:
                face_crop = frame[y:y+height, x:x+width]

                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, (48, 48))

                img_array = np.expand_dims(resized_face, axis=-1)
                img_array = np.expand_dims(img_array, axis=0)

                predictions = model.predict(img_array, verbose=0)[0]

                # penalize neutral, boost angry
                adjusted = predictions.copy()
                adjusted[NEUTRAL_INDEX] -= NEUTRAL_PENALTY
                adjusted[ANGRY_INDEX] += ANGRY_BOOST

                class_idx = np.argmax(adjusted)
                emotion_label = CLASS_NAMES[class_idx]

                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, emotion_label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # --- crop the camera frame to a centered square ---
    if w > h:
        offset = (w - h) // 2
        square_frame = frame[:, offset:offset + h]
    else:
        offset = (h - w) // 2
        square_frame = frame[offset:offset + w, :]

    camera_panel = cv2.resize(square_frame, (PANEL_SIZE, PANEL_SIZE))

    # --- build the cat panel ---
    cat_panel = cat_images[emotion_label].copy()
    cv2.putText(cat_panel, emotion_label, (5, PANEL_SIZE - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- combine camera (left) + cat (right) into one wide rectangle ---
    combined = np.hstack([camera_panel, cat_panel])

    cv2.imshow("Face", combined)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()