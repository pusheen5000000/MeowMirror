import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('src/emotion_model.keras')

face_detection = mp.solutions.face_detection.FaceDetection()

CLASS_NAMES = ["angry", "happy", "neutral", "sad"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            # Calculate pixel coordinates
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Prevent coordinates from going outside the image boundaries (avoids crashes)
            x, y = max(0, x), max(0, y)
            width, height = min(w - x, width), min(h - y, height)

            if width > 0 and height > 0:
                # 3. Crop the face from the frame
                face_crop = frame[y:y+height, x:x+width]

                # 4. Preprocess the crop to match training data (Grayscale -> 48x48)
                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, (48, 48))
                
                # 5. Reshape from (48, 48) to (1, 48, 48, 1) for the batch format the model expects
                img_array = np.expand_dims(resized_face, axis=-1)  # adds channel -> (48, 48, 1)
                img_array = np.expand_dims(img_array, axis=0)     # adds batch size -> (1, 48, 48, 1)

                # 6. Predict!
                predictions = model.predict(img_array, verbose=0)
                score = tf.nn.softmax(predictions[0]) # Turns raw numbers into clean probabilities
                
                class_idx = np.argmax(predictions[0])
                emotion_label = CLASS_NAMES[class_idx]

                # Draw the rectangle and the text label on the screen
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, emotion_label, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()