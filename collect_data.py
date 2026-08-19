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

LABELS = ["wave", "nodd", "shake_head"]
current_label_index = 0
recording = False
data = []

while True:
    bo, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)
    
    for c in result.pose_landmarks:
        for d in c:
            h, w, _ = frame.shape
            px = int(d.x * w)
            py = int(d.y * h)
            cv2.circle(frame, (px, py), 5, (0,255,0), -1)
            if recording:
                data.append([LABELS[current_label_index], d.x, d.y, d.z])
            
    status = f"Gest: {LABELS[current_label_index]} | Opptak: {recording}"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("vindusnavn", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('1'):
        current_label_index = 0
    elif key & 0xFF == ord('2'):
        current_label_index = 1
    elif key & 0xFF == ord('3'):
        current_label_index = 2
    elif key & 0xFF == ord(' '):
        recording = not recording

import csv

with open("landmarks.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "x", "y", "z"])
    writer.writerows(data)

cap.release()
cv2.destroyAllWindows()
