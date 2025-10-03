from pathlib import Path
import pickle
import face_recognition
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
IMG_ROOT = BASE_DIR / 'data' / 'criminal_images'
ENC_DIR = BASE_DIR / 'data' / 'encodings'
ENC_PATH = ENC_DIR / 'encodings.pickle'


def _parse_label_from_dir(dirname: str):
    # expects folder like "101_Alice" -> ("101", "Alice")
    base = Path(dirname).name
    parts = base.split('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return base, base


def build_encodings():
    ENC_DIR.mkdir(parents=True, exist_ok=True)
    data = []  # list of dicts: {student_id, name, encodings: [np.ndarray,...]}

    student_dirs = [d for d in IMG_ROOT.iterdir() if d.is_dir()]
    if not student_dirs:
        print("[WARN] No student folders found. Run register_criminal first.")
        return

    for sdir in student_dirs:
        sid, name = _parse_label_from_dir(str(sdir))
        print(f"[INFO] Processing {sid} - {name}")

        encs = []
        image_paths = list(sdir.glob("*.jpg")) + list(sdir.glob("*.png"))
        for ip in image_paths:
            image = face_recognition.load_image_file(str(ip))
            boxes = face_recognition.face_locations(image, model='hog')  # fast; use 'cnn' if GPU
            if not boxes:
                continue
            enc_list = face_recognition.face_encodings(image, boxes)
            encs.extend(enc_list)

        if encs:
            data.append({
                'student_id': sid,
                'name': name,
                'encodings': [e.tolist() for e in encs],  # to make it pickle/json friendly
            })
            print(f" -> {len(encs)} encodings")
        else:
            print(" -> No faces found in images; skip")

    with open(ENC_PATH, 'wb') as f:
        pickle.dump(data, f)

    print(f"[OK] Saved encodings to {ENC_PATH}")


if __name__ == "__main__":
    build_encodings()
