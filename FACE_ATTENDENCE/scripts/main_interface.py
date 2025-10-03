import os
import sys
import tkinter as tk
from tkinter import messagebox
from register_criminal import StudentRegisterApp
from train_encodings import build_encodings
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MainInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Police")
        self.root.geometry("1200x700")
        self.root.configure(bg="#ecf0f1")

        # ------------------ Header ------------------ #
        header = tk.Frame(root, bg="#2c3e50", height=80)
        header.pack(fill="x")

        title = tk.Label(
            header,
            text="VIRTUAL POLICE\nAI video analyzer",
            font=("Arial", 22, "bold"),
            bg="#2c3e50",
            fg="white",
            justify="center"
        )
        title.pack(expand=True)

        # ------------------ Sidebar ------------------ #
        sidebar = tk.Frame(root, bg="#34495e", width=260)
        sidebar.pack(side="left", fill="y")

        # Button style
        btn_style = {
            "font": ("Arial", 14, "bold"),
            "bg": "#1abc9c",
            "fg": "white",
            "activebackground": "#16a085",
            "activeforeground": "white",
            "relief": "flat",
            "bd": 0,
            "pady": 12,
            "width": 22
        }

        self.add_sidebar_button(sidebar, "📝 Register new entry", self.open_register, btn_style)
        self.add_sidebar_button(sidebar, "⚡ Train Encodings", self.run_encoding, btn_style)
        self.add_sidebar_button(sidebar, "📸 Start detection", self.run_recognition, btn_style)

        # Exit button (bottom)
        exit_btn = tk.Button(sidebar, text="❌ Exit", command=self.root.destroy, **btn_style)
        exit_btn.pack(side="bottom", pady=30)

        exit_btn.bind("<Enter>", lambda e: exit_btn.config(bg="#e74c3c"))
        exit_btn.bind("<Leave>", lambda e: exit_btn.config(bg="#1abc9c"))

        # ------------------ Main Content ------------------ #
        self.content = tk.Frame(root, bg="white")
        self.content.pack(side="right", expand=True, fill="both")

        self.status_label = tk.Label(
            self.content,
            text="✅Virtual Police\nChoose an option from the left panel.",
            font=("Arial", 16),
            bg="white",
            fg="#2c3e50",
            pady=40,
            justify="center"
        )
        self.status_label.pack(expand=True)

        # ------------------ Footer / Status Bar ------------------ #
        footer = tk.Frame(root, bg="#2c3e50", height=30)
        footer.pack(fill="x", side="bottom")

        self.footer_label = tk.Label(
            footer,
            text="Ready...",
            font=("Arial", 10),
            bg="#2c3e50",
            fg="white",
            anchor="w",
            padx=15
        )
        self.footer_label.pack(fill="x")

    # ------------------ Sidebar Buttons with Hover ------------------ #
    def add_sidebar_button(self, parent, text, command, style):
        btn = tk.Button(parent, text=text, command=command, **style)
        btn.pack(pady=15)

        def on_enter(e):
            btn.config(bg="#16a085")

        def on_leave(e):
            btn.config(bg="#1abc9c")

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

    # ------------------ Functions ------------------ #
    def open_register(self):
        reg_win = tk.Toplevel(self.root)
        reg_win.grab_set()  # modal window
        StudentRegisterApp(reg_win, num_images=2)
        self.set_status("Opened registration window.")

    def run_encoding(self):
        try:
            build_encodings()
            messagebox.showinfo("✅ Success", "Encodings trained and saved successfully!")
            self.set_status("Encodings updated successfully.")
        except Exception as e:
            messagebox.showerror("❌ Error", f"Failed to encode faces: {e}")
            self.set_status("Encoding failed!")

    def run_recognition(self):
        recog_path = os.path.join(BASE_DIR, "recognize.py")
        if not os.path.exists(recog_path):
            messagebox.showerror("❌ Error", "recognize.py not found!")
            return
        subprocess.Popen([sys.executable, recog_path])
        self.set_status("Recognition started...")

    def set_status(self, msg):
        self.status_label.config(text=msg)
        self.footer_label.config(text=msg)


if __name__ == "__main__":
    root = tk.Tk()
    app = MainInterface(root)
    root.mainloop()
