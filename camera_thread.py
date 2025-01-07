import threading
import queue
import cv2
import numpy as np
import os
from face_detection import detect_and_draw
from PIL import Image

class CameraThread(threading.Thread):
    def __init__(self, frame_queue, face_xml_path, eye_xml_path, hat_path=None, glasses_path=None):
        super().__init__()
        self.frame_queue = frame_queue
        self.daemon = True

        # Load Haar cascades
        self.face_cascade = cv2.CascadeClassifier(face_xml_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_xml_path)

        # Detection state
        self.face_detected = False
        self.face_detection_time = None

        # Overlays
        self.hat_img = None
        self.glasses_img = None
        self.load_hat(hat_path)
        self.load_glasses(glasses_path)

        # Face box outline
        self.draw_face_box = True  # Default: enabled

        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[CameraThread] Error: Unable to open camera.")

        self.running = True

    def load_hat(self, hat_path):
        self.hat_img = self._load_image(hat_path)

    def load_glasses(self, glasses_path):
        self.glasses_img = self._load_image(glasses_path)

    def _load_image(self, path):
        if path is None:
            return None
        try:
            pil_img = Image.open(path).convert("RGBA")
            overlay = np.array(pil_img)
            # Swap channels from RGBA to BGRA
            overlay = overlay[:, :, [2, 1, 0, 3]]  # Swap R and B
            return overlay
        except Exception as e:
            print(f"[CameraThread] Failed to load file: {path} - {e}")
            return None

    def toggle_face_box(self, enable):
        self.draw_face_box = enable

    def set_hat(self, hat_path):
        self.load_hat(hat_path)

    def set_glasses(self, glasses_path):
        self.load_glasses(glasses_path)

    def run(self):
        while self.running:
            ret, img = self.cap.read()
            if not ret:
                continue

            # Detect and draw
            processed, self.face_detected, self.face_detection_time = detect_and_draw(
                img,
                self.face_cascade,
                self.eye_cascade,
                hat_img=self.hat_img,
                glasses_img=self.glasses_img,
                face_detected_flag=self.face_detected,
                face_detected_time=self.face_detection_time,
                draw_face_box=self.draw_face_box  # Pass the flag value
            )

            # Put into queue if space is available
            if not self.frame_queue.full():
                self.frame_queue.put(processed)

        # Release camera
        self.cap.release()
        print("[CameraThread] Camera released.")

    def stop(self):
        self.running = False
