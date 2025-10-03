import streamlit as st
import os
import pickle
import numpy as np
from datetime import datetime
from PIL import Image
import cv2
import dlib
import face_recognition
import csv
from scipy.spatial import distance as dist
import time
import pandas as pd

st.set_page_config(page_title="Virtual Police", layout="wide")
st.markdown("""
<style>
.sidebar .sidebar-content {
    background-color: #34495e;
    color: white;
    font-family: 'Arial';
}
.stButton>button {
    background-color: #1abc9c;
    color: white;
    font-weight: bold;
    height: 50px;
    width: 220px;
    border-radius: 10px;
    border: 0px;
    font-size: 16px;
    margin-bottom: 10px;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #16a085;
    color: white;
}
h1 {
    color: #2c3e50;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "criminal_images")
ENC_DIR = os.path.join(BASE_DIR, "data", "encodings")
ENC_PATH = os.path.join(ENC_DIR, "encodings.pickle")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PREDICTOR_PATH = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
ATT_DIR = os.path.join(BASE_DIR, "criminal_logs")
os.makedirs(ATT_DIR, exist_ok=True)

# Sidebar menu
st.sidebar.title("📋 Virtual Police")
st.sidebar.markdown("---")

if "action" not in st.session_state:
    st.session_state['action'] = "home"

if st.sidebar.button("📝 Register Criminal"):
    st.session_state['action'] = "register"
if st.sidebar.button("⚡ Train Encodings"):
    st.session_state['action'] = "train"
if st.sidebar.button("📸 Mark Attendance"):
    st.session_state['action'] = "attendance"
if st.sidebar.button("📂 View Attendance Logs"):
    st.session_state['action'] = "view_logs"

st.markdown("<h1>📸 Virtual Police</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


if st.session_state['action'] == "home":
    st.write("✅ Welcome to Virtual Police")
    st.info("Use the left panel to register criminals, train encodings, mark attendance, or view logs.")

elif st.session_state['action'] == "register":
    st.subheader("📝 Register a New Criminal")
    sid = st.text_input("Criminal ID")
    name = st.text_input("Criminal Name")
    uploaded_files = st.file_uploader("Upload Images (optional)", type=["jpg", "png"], accept_multiple_files=True)
    cam_img = st.camera_input("Capture Photo via Webcam (optional)")

    if st.button("Save Criminal"):
        if not sid or not name:
            st.error("Enter both Criminal ID and Name")
        else:
            folder = os.path.join(DATA_DIR, f"{sid}_{name}")
            os.makedirs(folder, exist_ok=True)
            if uploaded_files:
                for file in uploaded_files:
                    img = Image.open(file).convert("RGB")
                    img.save(os.path.join(folder, file.name))
            if cam_img:
                img = Image.open(cam_img).convert("RGB")
                img.save(os.path.join(folder, f"{sid}_{name}_cam.jpg"))
            st.success(f"✅ Criminal {name} ({sid}) registered successfully!")

elif st.session_state['action'] == "train":
    st.subheader("⚡ Train Face Encodings")
    if st.button("Train Now"):
        if not os.path.exists(DATA_DIR):
            st.error("No criminal images found!")
        else:
            os.makedirs(ENC_DIR, exist_ok=True)
            data_enc = []
            criminal_dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
            for cdir in criminal_dirs:
                sid_name = os.path.basename(cdir)
                sid, name = sid_name.split("_",1)
                encs = []
                for img_file in os.listdir(cdir):
                    if img_file.endswith((".jpg",".png")):
                        path = os.path.join(cdir, img_file)
                        image = face_recognition.load_image_file(path)
                        boxes = face_recognition.face_locations(image, model='hog')
                        enc_list = face_recognition.face_encodings(image, boxes)
                        encs.extend(enc_list)
                if encs:
                    data_enc.append({
                        "student_id": sid,
                        "name": name,
                        "encodings": [e.tolist() for e in encs]
                    })
            with open(ENC_PATH, "wb") as f:
                pickle.dump(data_enc, f)
            st.success("✅ Face encodings trained successfully!")

elif st.session_state['action'] == "attendance":
    st.subheader("📸 Mark Attendance (Blink + Face Recognition)")
    st.info("Ensure your face is visible. Blink twice to mark attendance.")

    if not os.path.exists(ENC_PATH):
        st.warning("⚠️ Train encodings first!")
    else:
        with open(ENC_PATH, "rb") as f:
            data = pickle.load(f)

        known_encodings = []
        known_names = []
        for criminal in data:
            for enc in criminal["encodings"]:
                known_encodings.append(np.array(enc))
                known_names.append(f"{criminal['student_id']} - {criminal['name']}")

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
        (lStart, lEnd) = (42, 48)
        (rStart, rEnd) = (36, 42)

        EAR_THRESHOLD = 0.22
        CONSEC_FRAMES = 3
        REQUIRED_BLINKS = 2
        TIME_LIMIT = 10

        stframe = st.empty()
        run = st.button("Start Detection")

        attendance_marked_faces = set()

        if run:
            cap = cv2.VideoCapture(0)
            blink_counters = {}
            blink_counts = {}
            start_time = None
            frame_count = 0

            date_str = datetime.now().strftime("%Y-%m-%d")
            csv_path = os.path.join(ATT_DIR, f"attendance_{date_str}.csv")
            if not os.path.exists(csv_path):
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["ID-Name", "Date", "Time"])

            def eye_aspect_ratio(eye):
                A = dist.euclidean(eye[1], eye[5])
                B = dist.euclidean(eye[2], eye[4])
                C = dist.euclidean(eye[0], eye[3])
                return (A + B) / (2.0 * C)

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Cannot access webcam!")
                    break

                frame_count += 1
                # Process every 3rd frame for performance
                if frame_count % 3 != 0:
                    continue

                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize for faster processing
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 0)

                if start_time is None:
                    start_time = time.time()

                for i, rect in enumerate(rects):
                    # Scale face coords back to original frame size
                    left = rect.left() * 2
                    top = rect.top() * 2
                    right_ = rect.right() * 2
                    bottom = rect.bottom() * 2

                    shape = predictor(cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY), rect)
                    shape = [(shape.part(j).x * 2, shape.part(j).y * 2) for j in range(68)]
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0

                    if i not in blink_counters:
                        blink_counters[i] = 0
                        blink_counts[i] = 0

                    if ear < EAR_THRESHOLD:
                        blink_counters[i] += 1
                    else:
                        if blink_counters[i] >= CONSEC_FRAMES:
                            blink_counts[i] += 1
                        blink_counters[i] = 0

                    for (x, y) in leftEye + rightEye:
                        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

                    cv2.rectangle(frame, (left, top), (right_, bottom), (0, 255, 255), 2)
                    cv2.putText(frame, "Detected Face", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                    if blink_counts[i] >= REQUIRED_BLINKS and i not in attendance_marked_faces:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        enc = face_recognition.face_encodings(rgb, [(top, right_, bottom, left)])
                        if enc:
                            distances = face_recognition.face_distance(known_encodings, enc[0])
                            best_idx = np.argmin(distances)
                            if distances[best_idx] < 0.5:
                                name = known_names[best_idx]
                                time_str = datetime.now().strftime("%H:%M:%S")
                                with open(csv_path, "a", newline="") as f:
                                    writer = csv.writer(f)
                                    writer.writerow([name, date_str, time_str])
                                st.success(f"✅ Detection marked for {name} at {time_str}")
                                attendance_marked_faces.add(i)

                elapsed = time.time() - start_time
                cv2.putText(frame, f"Blinks: {sum(blink_counts.values())}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB")

                if elapsed > TIME_LIMIT and len(attendance_marked_faces) < len(rects):
                    st.warning("❌ Liveness check failed!")
                    break
            cap.release()

elif st.session_state['action'] == "view_logs":
    st.subheader("📂 View Detected Logs")
    csv_files = [f for f in os.listdir(ATT_DIR) if f.endswith(".csv")]
    if csv_files:
        selected_csv = st.selectbox("Select Date", csv_files)
        df = pd.read_csv(os.path.join(ATT_DIR, selected_csv))
        st.dataframe(df)
    else:
        st.info("No attendance logs found yet.")
