import cv2
import time
import numpy as np


def filter_eyes(eyes, face_width, face_height):
    """
    Filters detected eyes based on:
    - Eyes must be in the upper half of the face.
    - Eye size must be reasonable relative to the face size.
    Returns a list of valid eyes (ex, ey, ew, eh).
    """
    valid_eyes = []
    for (ex, ey, ew, eh) in eyes:
        # Eyes should be in the upper half of the face
        if (ey + eh // 2) > (face_height // 2):
            continue

        # Eye width must be reasonable compared to face width
        if ew < face_width / 10 or ew > face_width / 2:
            continue

        valid_eyes.append((ex, ey, ew, eh))

    return valid_eyes


def place_overlay_safely(img, overlay, x, y):
    h, w = overlay.shape[:2]
    img_h, img_w = img.shape[:2]

    # Calculate overlay coordinates (considering image boundaries)
    x1, x2 = max(0, x), min(img_w, x + w)
    y1, y2 = max(0, y), min(img_h, y + h)

    overlay_x1, overlay_x2 = max(0, -x), w - max(0, (x + w) - img_w)
    overlay_y1, overlay_y2 = max(0, -y), h - max(0, (y + h) - img_h)

    if x1 >= x2 or y1 >= y2:
        return img  # Overlay is completely outside the image

    alpha_overlay = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
    overlay_bgr = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, :3]
    img_region = img[y1:y2, x1:x2]

    # Blend pixels considering alpha channel
    for c in range(3):  # Iterate over BGR channels
        img_region[:, :, c] = np.where(
            alpha_overlay > 0,
            (1 - alpha_overlay) * img_region[:, :, c] + alpha_overlay * overlay_bgr[:, :, c],
            img_region[:, :, c]  # Preserve original pixels for alpha = 0
        )

    img[y1:y2, x1:x2] = img_region
    return img


def detect_and_draw(img, face_cascade, eye_cascade, hat_img=None, glasses_img=None,
                    face_detected_flag=None, face_detected_time=None, draw_face_box=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        if face_detected_flag is not True:
            face_detected_flag = True
            face_detected_time = time.strftime("%H:%M:%S", time.localtime())
    else:
        face_detected_flag = False

    for (x, y, w, h) in faces:
        if draw_face_box:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # Add hat overlay
        if hat_img is not None and y - int(0.5 * h) >= 0:
            overlay_resized = cv2.resize(hat_img, (w, int(0.5 * h)), interpolation=cv2.INTER_AREA)
            img = place_overlay_safely(img, overlay_resized, x, y - int(0.5 * h))

        # Detect eyes
        face_roi_gray = gray[y:y + h, x:x + w]
        raw_eyes = eye_cascade.detectMultiScale(
            face_roi_gray,
            scaleFactor=1.2,
            minNeighbors=2,
            minSize=(20, 20)
        )
        valid_eyes = filter_eyes(raw_eyes, w, h)
        valid_eyes = sorted(valid_eyes, key=lambda e: e[2], reverse=True)[:2]

        if glasses_img is not None and len(valid_eyes) == 2:
            (ex1, ey1, ew1, eh1) = valid_eyes[0]
            (ex2, ey2, ew2, eh2) = valid_eyes[1]

            cx1 = x + ex1 + ew1 // 2
            cy1 = y + ey1 + eh1 // 2
            cx2 = x + ex2 + ew2 // 2
            cy2 = y + ey2 + eh2 // 2

            cx_mid = (cx1 + cx2) // 2
            cy_mid = (cy1 + cy2) // 2

            overlay_width = int(0.6 * w)
            overlay_height = int(overlay_width * glasses_img.shape[0] / glasses_img.shape[1])
            overlay_resized = cv2.resize(glasses_img, (overlay_width, overlay_height), interpolation=cv2.INTER_AREA)

            gx1 = cx_mid - (overlay_width // 2)
            gy1 = cy_mid - (overlay_height // 2)

            img = place_overlay_safely(img, overlay_resized, gx1, gy1)

    return img, face_detected_flag, face_detected_time
