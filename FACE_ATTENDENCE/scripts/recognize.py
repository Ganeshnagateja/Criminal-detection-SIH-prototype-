import os
import pickle
import cv2
import dlib
import face_recognition
import numpy as np
from scipy.spatial import distance as dist
from datetime import datetime
import time
import csv

# ------------------ Base directories ------------------ #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENC_DIR = os.path.join(BASE_DIR, "..", "data", "encodings")
ENC_PATH = os.path.join(ENC_DIR, "encodings.pickle")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
PREDICTOR_PATH = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
ATT_DIR = os.path.join(BASE_DIR, "criminal_logs")
os.makedirs(ATT_DIR, exist_ok=True)

# ------------------ Load known encodings ------------------ #
if not os.path.exists(ENC_PATH):
    print("[ERROR] Encodings file not found! Run train_encodings.py first.")
    exit()

with open(ENC_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = []
known_names = []
for student in data:
    for enc in student['encodings']:
        known_encodings.append(np.array(enc))
        known_names.append(f"{student['student_id']} - {student['name']}")

# ------------------ Dlib detector & predictor ------------------ #
detector = dlib.get_frontal_face_detector()
if not os.path.exists(PREDICTOR_PATH):
    print("[ERROR] Shape predictor model not found! Place 'shape_predictor_68_face_landmarks.dat' in models folder.")
    exit()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# ------------------ Eye landmarks ------------------ #
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# ------------------ Blink & Liveness Parameters ------------------ #
EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 3
REQUIRED_BLINKS = 2
TIME_LIMIT = 10  # seconds

# ------------------ Attendance CSV ------------------ #
date_str = datetime.now().strftime("%Y-%m-%d")
csv_path = os.path.join(ATT_DIR, f"attendance_{date_str}.csv")
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID-Name", "Date", "Time"])

# ------------------ Video Capture ------------------ #
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot access webcam!")
    exit()

print("[INFO] Please blink at least twice within 10 seconds...")

# ------------------ EAR Calculation ------------------ #
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ------------------ Blink tracking dictionary ------------------ #
blink_data = {}

start_time = None

# ------------------ Main Loop ------------------ #
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if start_time is None:
        start_time = time.time()

    for i, rect in enumerate(rects):
        # Use face index as a simple ID
        face_key = i

        if face_key not in blink_data:
            blink_data[face_key] = {'blink_count': 0, 'frame_counter': 0, 'attendance_marked': False}

        shape = predictor(gray, rect)
        shape = [(shape.part(j).x, shape.part(j).y) for j in range(68)]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EAR_THRESHOLD:
            blink_data[face_key]['frame_counter'] += 1
        else:
            if blink_data[face_key]['frame_counter'] >= CONSEC_FRAMES:
                blink_data[face_key]['blink_count'] += 1
                print(f"[BLINK] Face {face_key} Count: {blink_data[face_key]['blink_count']}")
            blink_data[face_key]['frame_counter'] = 0

        # Draw eye landmarks
        for (x, y) in leftEye + rightEye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Draw rectangle around face
        left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

        # Mark attendance if blink count threshold reached
        if blink_data[face_key]['blink_count'] >= REQUIRED_BLINKS and not blink_data[face_key]['attendance_marked']:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            enc = face_recognition.face_encodings(rgb, [(top, right, bottom, left)])
            if enc:
                distances = face_recognition.face_distance(known_encodings, enc[0])
                best_idx = np.argmin(distances)
                if distances[best_idx] < 0.5:  # match threshold
                    name = known_names[best_idx]
                    time_str = datetime.now().strftime("%H:%M:%S")
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([name, date_str, time_str])
                    print(f"[ATTENDANCE] {name} marked at {time_str}")
                    blink_data[face_key]['attendance_marked'] = True

    elapsed = time.time() - start_time
    total_blinks = sum(face['blink_count'] for face in blink_data.values())
    cv2.putText(frame, f"Blinks: {total_blinks}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if elapsed > TIME_LIMIT and total_blinks < REQUIRED_BLINKS * len(rects):
        print("[FAILED] Liveness check failed ❌")
        break

    cv2.imshow("Face + Blink Detection", frame)

    # Stop recognition on any key press
    if cv2.waitKey(1) != -1:
        print("[INFO] Key pressed, stopping recognition.")
        break

cap.release()
cv2.destroyAllWindows()
