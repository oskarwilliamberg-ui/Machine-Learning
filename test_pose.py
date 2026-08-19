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
            

    cv2.imshow("vindusnavn", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()