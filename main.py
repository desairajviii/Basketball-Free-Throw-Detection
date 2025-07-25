import cv2 as cv
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import time

# Initializing Mediapipe Pose and Drawing
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Initializing YOLO model for ball detection
ball_model = YOLO('yolov8n.pt')

def get_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def extract_keypoints(landmarks, frame_shape):
    h, w, _ = frame_shape
    get_point = lambda i: [int(landmarks[i].x * w ), int(landmarks[i].y * h)] 
    return {
        "right_shoulder": get_point(12),
        "right_elbow": get_point(14),   
        "right_wrist": get_point(16),
    }

def is_release(angles_history, threshold=20):
    if len(angles_history) < 5:
        return False
    recent = angles_history[-5:]
    delta = max(recent) - min(recent) 
    return delta > threshold

def detect_ball(frame):
    results = ball_model.predict(source=frame, verbose=False)[0]
    boxes = results.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == 32 and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv.putText(frame, f'Ball {conf:.2f}', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
            return True
    return False

cap = cv.VideoCapture(0)
angle_history = []
release_count = 0
start_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    ball_detected = detect_ball(frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        keypoints = extract_keypoints(landmarks, frame.shape)

        angle = get_angle(
            keypoints['right_shoulder'],
            keypoints['right_elbow'],
            keypoints['right_wrist']
        )
        angle_history.append(angle)

        # Display elbow angle
        cv.putText(frame, f"Elbow Angle: {int(angle)}", (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Check for shot release
        if is_release(angle_history):
            release_count += 1
            cv.putText(frame, f"Shot Released! Count: {release_count}", (10, 70),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            angle_history.clear()

    # FPS display
    fps = int(1 / (time.time() - start_time + 1e-5))
    start_time = time.time()
    cv.putText(frame, f"FPS: {fps}", (10, frame.shape[0] - 10),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    cv.imshow("Free Throw Tracker", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()