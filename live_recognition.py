import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

base_options = mp_python.BaseOptions(
    model_asset_path='pose_landmarker.task',
    delegate=mp_python.BaseOptions.Delegate.CPU,
)
options = vision.PoseLandmarkerOptions(base_options=base_options)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

import joblib
model = joblib.load("gesture_model.joblib")

while True:
    bo, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])

        prediction = model.predict([features])
        gesture = prediction[0]

        cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("live gesture recognition", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
