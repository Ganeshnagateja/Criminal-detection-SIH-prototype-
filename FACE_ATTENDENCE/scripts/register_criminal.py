import cv2
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
IMG_ROOT = BASE_DIR / 'data' / 'criminal_images'
IMG_ROOT.mkdir(parents=True, exist_ok=True)

class StudentRegisterApp:
    def __init__(self, root, num_images=2):
        self.root = root
        self.root.title("🔪Criminal Registration")
        self.root.geometry("950x550")
        self.root.configure(bg="#f5f6fa")

        self.num_images = num_images
        self.count = 0
        self.current_frame = None

        # --- Left panel: Form ---
        form_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="groove", padx=15, pady=15)
        form_frame.pack(side="left", fill="y", padx=20, pady=20)

        tk.Label(form_frame, text="Criminal Registration", font=("Arial", 16, "bold"), bg="#ffffff").pack(pady=10)

        tk.Label(form_frame, text="Criminal ID:", font=("Arial", 12), bg="#ffffff").pack(anchor="w", pady=(20, 5))
        self.student_id = tk.Entry(form_frame, font=("Arial", 12), bd=2, relief="solid")
        self.student_id.pack(fill="x", pady=5)

        tk.Label(form_frame, text="Criminal Name:", font=("Arial", 12), bg="#ffffff").pack(anchor="w", pady=(20, 5))
        self.name = tk.Entry(form_frame, font=("Arial", 12), bd=2, relief="solid")
        self.name.pack(fill="x", pady=5)

        self.status_label = tk.Label(form_frame, text="Status: Waiting...", font=("Arial", 10), fg="gray", bg="#ffffff")
        self.status_label.pack(pady=15)

        btn_frame = tk.Frame(form_frame, bg="#ffffff")
        btn_frame.pack(pady=20)

        self.save_btn = tk.Button(btn_frame, text="💾 Save Photo", font=("Arial", 12, "bold"),
                                  bg="#4CAF50", fg="white", padx=15, pady=8, command=self.save_photo)
        self.save_btn.grid(row=0, column=0, padx=10)

        self.upload_btn = tk.Button(btn_frame, text="📂 Upload Photo", font=("Arial", 12, "bold"),
                                    bg="#2980b9", fg="white", padx=15, pady=8, command=self.upload_photo)
        self.upload_btn.grid(row=0, column=1, padx=10)

        self.quit_btn = tk.Button(btn_frame, text="❌ Quit", font=("Arial", 12, "bold"),
                                  bg="#e74c3c", fg="white", padx=15, pady=8, command=self.quit_app)
        self.quit_btn.grid(row=1, column=0, columnspan=2, pady=15)

        # --- Right panel: Camera Preview ---
        preview_frame = tk.Frame(root, bg="#dcdde1", bd=2, relief="groove")
        preview_frame.pack(side="right", expand=True, fill="both", padx=20, pady=20)

        self.video_label = tk.Label(preview_frame, bg="#2f3640")
        self.video_label.pack(expand=True, fill="both")

        # --- Initialize Camera ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showwarning("⚠️ Warning", "Could not access webcam. You can still upload photos.")
            self.cap = None
        else:
            self.update_frame()

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img.resize((600, 450)))
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                self.current_frame = frame
            self.root.after(10, self.update_frame)

    def save_photo(self):
        sid = self.student_id.get().strip()
        name = self.name.get().strip()
        if not sid or not name:
            messagebox.showwarning("⚠️ Missing Info", "Please enter both Criminal ID and Name.")
            return

        if self.count >= self.num_images:
            messagebox.showinfo("✅ Done", f"Already saved {self.num_images} images for this criminal.")
            return

        if self.current_frame is None:
            messagebox.showerror("❌ Error", "No live camera frame available.")
            return

        safe_name = f"{sid}_{name}".replace(" ", "_")
        student_dir = IMG_ROOT / safe_name
        student_dir.mkdir(parents=True, exist_ok=True)

        img_name = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + ".jpg"
        try:
            cv2.imwrite(str(student_dir / img_name), cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))
            self.count += 1
            self.status_label.config(text=f"Status: Saved {self.count}/{self.num_images} images", fg="green")
            print(f"[OK] Saved {self.count}/{self.num_images} -> {img_name}")
        except Exception as e:
            messagebox.showerror("❌ Error", f"Failed to save image: {e}")

    def upload_photo(self):
        sid = self.student_id.get().strip()
        name = self.name.get().strip()
        if not sid or not name:
            messagebox.showwarning("⚠️ Missing Info", "Please enter both Criminal ID and Name.")
            return

        if self.count >= self.num_images:
            messagebox.showinfo("✅ Done", f"Already saved {self.num_images} images for this criminal.")
            return

        file_paths = filedialog.askopenfilenames(
            title="Select Photo(s)",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if not file_paths:
            return

        safe_name = f"{sid}_{name}".replace(" ", "_")
        student_dir = IMG_ROOT / safe_name
        student_dir.mkdir(parents=True, exist_ok=True)

        for fpath in file_paths:
            if self.count >= self.num_images:
                break
            img = cv2.imread(fpath)
            if img is None:
                continue
            img_name = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + ".jpg"
            try:
                cv2.imwrite(str(student_dir / img_name), img)
                self.count += 1
                print(f"[OK] Uploaded {self.count}/{self.num_images} -> {img_name}")
            except Exception as e:
                messagebox.showerror("❌ Error", f"Failed to upload image: {e}")

        self.status_label.config(text=f"Status: Saved {self.count}/{self.num_images} images", fg="blue")

    def quit_app(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = StudentRegisterApp(root, num_images=2)   # Only 2 photos required
    root.mainloop()
