import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import queue
import cv2
import os
from camera_thread import CameraThread


def on_closing(root, camera_thread):
    """Closes the application window and camera thread."""
    if messagebox.askokcancel("Exit", "Do you want to quit?"):
        camera_thread.stop()
        root.destroy()


class FaceUpApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FaceUp – Face Detection")

        # Frame queue
        self.frame_queue = queue.Queue(maxsize=3)

        # Paths to Haar cascades
        xml_folder = 'xml'
        face_cascade_file = 'haarcascade_frontalface_default.xml'
        eye_cascade_file = 'haarcascade_eye.xml'
        face_cascade_path = os.path.join(xml_folder, face_cascade_file)
        eye_cascade_path = os.path.join(xml_folder, eye_cascade_file)

        # Camera thread
        self.camera_thread = CameraThread(
            frame_queue=self.frame_queue,
            face_xml_path=face_cascade_path,
            eye_xml_path=eye_cascade_path,
            hat_path='example_overlays/szczur.png',
            glasses_path='example_overlays/glasses.png'
        )

        self.camera_thread.start()

        # Left and right frames
        self.frame_left = tk.Frame(root)
        self.frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.frame_right = tk.Frame(root)
        self.frame_right.pack(side=tk.RIGHT, fill=tk.Y)

        # Video label
        self.label_video = tk.Label(self.frame_left)
        self.label_video.pack(fill=tk.BOTH, expand=True)

        # Buttons on the right side
        btn_hat = tk.Button(self.frame_right, text="Select Hat", command=self.pick_hat)
        btn_hat.pack(pady=5)

        btn_glasses = tk.Button(self.frame_right, text="Select Glasses", command=self.pick_glasses)
        btn_glasses.pack(pady=5)

        btn_none = tk.Button(self.frame_right, text="Remove Overlay",
                             command=lambda: (self.camera_thread.set_hat(None), self.camera_thread.set_glasses(None)))
        btn_none.pack(pady=5)

        self.show_face_box = tk.BooleanVar(value=True)  # Default: enabled

        btn_toggle_face_box = tk.Checkbutton(self.frame_right, text="Show Face Outline",
                                             variable=self.show_face_box,
                                             command=self.toggle_face_box)
        btn_toggle_face_box.pack(pady=5)

        btn_exit = tk.Button(self.frame_right, text="Exit",
                             command=lambda: on_closing(root, self.camera_thread))
        btn_exit.pack(pady=20)

        # Frame updates via after()
        self.update_frame()

    def pick_hat(self):
        path = filedialog.askopenfilename(
            title="Select a Hat File (PNG)",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All Files", "*.*")]
        )
        if path:
            self.camera_thread.set_hat(path)

    def pick_glasses(self):
        path = filedialog.askopenfilename(
            title="Select a Glasses File (PNG)",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All Files", "*.*")]
        )
        if path:
            self.camera_thread.set_glasses(path)

    def toggle_face_box(self):
        self.camera_thread.toggle_face_box(self.show_face_box.get())

    def update_frame(self):
        """Fetches the latest frame from the queue and displays it in the label."""
        try:
            frame = self.frame_queue.get_nowait()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=pil_img)
            self.label_video.imgtk = imgtk
            self.label_video.config(image=imgtk)
        except queue.Empty:
            pass

        # Refresh every 30 ms
        self.root.after(30, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceUpApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root, app.camera_thread))
    root.mainloop()
