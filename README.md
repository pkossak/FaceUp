# **FaceUp – Real-Time Face Detection and Image Overlaying**
## Overview
FaceUp is a Python-based application for real-time face detection and image overlaying. Built using OpenCV and Tkinter, the application detects faces and eyes in real-time using Haar cascades and allows users to apply fun overlays like hats and glasses to their live video feed. The goal is to provide an intuitive and interactive tool for face detection with customizable visual augmentations.

## Features
1. Real-Time Face Detection: Uses OpenCV's Haar cascades to detect faces and eyes in live video streams.
2. Custom Overlays:
- Add hats or glasses by selecting image files.
- Seamlessly blend overlays onto detected faces using alpha transparency.
3. Interactive Controls:
- Enable/disable face outlines for detected regions.
- Reset overlays to none.
4. Efficient Performance: Multithreaded processing ensures smooth video playback and responsive UI.
  
## How It Works
1. Face and Eye Detection:
- Haar cascades detect faces and eyes in each video frame.
- Detected faces are outlined (optional).
  
2. Overlay Placement:
- User-selected PNG images (hats or glasses) are resized and positioned based on detected facial features.
- Alpha channels in PNG files are used for blending overlays into the video feed seamlessly.

3. Interactive GUI:
- Built with Tkinter, the app provides buttons and controls for:
  - Selecting hat or glasses images.
  - Enabling/disabling face outlines.
  - Exiting the application.

4. Threaded Video Processing:
- Video feed is handled in a separate thread for better performance and UI responsiveness.

## Project Structure
- `main.py`: Entry point of the application, handles GUI and user interaction.
- `camera_thread.py:` Contains the thread for capturing video frames and applying detection/overlays.
- `face_detection.py`: Handles face and eye detection, and overlay placement using OpenCV.
- `xml/`: Directory containing Haar cascade files for face (haarcascade_frontalface_default.xml) and eye (haarcascade_eye.xml) detection.

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/FaceUp.git
cd FaceUp
```

2. Install the required Python libraries:
```
pip install -r requirements.txt
```

3. Ensure you have a working webcam connected.

4. Run the application:
```
python main.py
```

## Requirements
- Python 3.7+
- OpenCV
- NumPy
- Pillow (PIL)
- Tkinter (pre-installed with Python on most systems)

## Usage
1. Launch the application.
2. Use the "Select Hat" or "Select Glasses" buttons to choose overlay images.
3. Toggle "Show Face Outline" to enable or disable the face detection box.
4. Click "Remove Overlay" to clear any selected overlays.
5. Exit the application via the "Exit" button or close the window.

## Future Enhancements
- Add support for custom AR filters and effects.
- Optimize detection using deep learning-based face detection models (e.g., DNN or YOLO).

## Contributions
Contributions, issues, and feature requests are welcome! Feel free to fork this repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
