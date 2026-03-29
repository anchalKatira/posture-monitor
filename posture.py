import cv2
import numpy as np
import os
import dlib

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = None


def load_predictor():
    global predictor
    if not os.path.exists(PREDICTOR_PATH):
        raise FileNotFoundError(
            f"Missing: {PREDICTOR_PATH}\n"
            "Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        )
    predictor = dlib.shape_predictor(PREDICTOR_PATH)


def get_landmarks(gray_frame):
    if gray_frame is None:
        return None, None
    if not isinstance(gray_frame, np.ndarray):
        return None, None
    if gray_frame.size == 0:
        return None, None
    if len(gray_frame.shape) != 2:
        return None, None
    if gray_frame.dtype != np.uint8:
        gray_frame = gray_frame.astype(np.uint8)

    # KEY FIX: histogram equalization for dark faces
    gray_frame = cv2.equalizeHist(gray_frame)

    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return None, None

    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

    try:
        shape = predictor(gray_frame, dlib_rect)
        landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
        return landmarks, dlib_rect
    except Exception:
        return None, None


def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx))


def calculate_head_tilt(landmarks):
    nose_tip = landmarks[30]
    chin     = landmarks[8]
    angle    = calculate_angle(nose_tip, chin)
    deviation = abs(angle - (-90))
    if deviation > 180:
        deviation = 360 - deviation
    return deviation


def calculate_ear_shoulder_ratio(landmarks, frame_height):
    left_ear  = landmarks[0]
    right_ear = landmarks[16]
    ear_y     = (left_ear[1] + right_ear[1]) / 2
    return ear_y / frame_height


def calculate_eye_level_tilt(landmarks):
    left_eye  = landmarks[36]
    right_eye = landmarks[45]
    angle = abs(calculate_angle(left_eye, right_eye))
    if angle > 90:
        angle = 180 - angle
    return angle


def get_posture_state(landmarks, frame_height, thresholds):
    head_tilt = calculate_head_tilt(landmarks)
    ear_ratio = calculate_ear_shoulder_ratio(landmarks, frame_height)
    eye_tilt  = calculate_eye_level_tilt(landmarks)

    metrics = {
        "head_tilt_deg": round(head_tilt, 1),
        "ear_y_ratio":   round(ear_ratio, 3),
        "eye_tilt_deg":  round(eye_tilt, 1),
    }

    slouch_score = 0
    if head_tilt > thresholds["head_tilt_warn"]:
        slouch_score += 1
    if head_tilt > thresholds["head_tilt_bad"]:
        slouch_score += 1
    if ear_ratio > thresholds["ear_ratio_warn"]:
        slouch_score += 1
    if ear_ratio > thresholds["ear_ratio_bad"]:
        slouch_score += 1
    if eye_tilt > thresholds["eye_tilt_warn"]:
        slouch_score += 1

    if slouch_score >= 3:
        state = "SLOUCH"
    elif slouch_score >= 1:
        state = "WARNING"
    else:
        state = "GOOD"

    return state, metrics


def draw_landmarks_on_frame(frame, landmarks, face_rect, state):
    colors = {
        "GOOD":    (56, 217, 169),
        "WARNING": (0, 200, 255),
        "SLOUCH":  (80, 80, 247),
    }
    color = colors.get(state, (200, 200, 200))

    x1 = face_rect.left()
    y1 = face_rect.top()
    x2 = face_rect.right()
    y2 = face_rect.bottom()
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    key_points = [0, 8, 16, 30, 36, 45]
    for idx in key_points:
        pt = tuple(landmarks[idx].astype(int))
        cv2.circle(frame, pt, 4, color, -1)

    nose = tuple(landmarks[30].astype(int))
    chin = tuple(landmarks[8].astype(int))
    cv2.line(frame, nose, chin, color, 2)

    leye = tuple(landmarks[36].astype(int))
    reye = tuple(landmarks[45].astype(int))
    cv2.line(frame, leye, reye, color, 1)

    return frame


DEFAULT_THRESHOLDS = {
    "head_tilt_warn":  18.0,
    "head_tilt_bad":   30.0,
    "ear_ratio_warn":  0.42,
    "ear_ratio_bad":   0.50,
    "eye_tilt_warn":   8.0,
}
