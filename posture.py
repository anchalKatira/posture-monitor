import dlib
import cv2
import numpy as np
import os

# ── dlib setup ──
detector = dlib.get_frontal_face_detector()

# Path to shape predictor model (will be downloaded if not present)
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = None


def load_predictor():
    """Load dlib's 68-point landmark predictor."""
    global predictor
    if not os.path.exists(PREDICTOR_PATH):
        raise FileNotFoundError(
            f"Missing: {PREDICTOR_PATH}\n"
            "Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
            "Extract and place in the project folder."
        )
    predictor = dlib.shape_predictor(PREDICTOR_PATH)


def get_landmarks(gray_frame):
    """
    Detect face and return 68 facial landmarks.
    Returns (landmarks_array, face_rect) or (None, None) if no face found.
    landmarks_array shape: (68, 2) — (x, y) per point
    """
    faces = detector(gray_frame, 0)
    if len(faces) == 0:
        return None, None

    face = faces[0]  # use first detected face
    shape = predictor(gray_frame, face)
    landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
    return landmarks, face


def calculate_angle(p1, p2):
    """
    Calculate the angle (in degrees) of the line from p1 to p2
    relative to the horizontal axis.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle


def calculate_head_tilt(landmarks):
    """
    Estimate vertical head tilt using:
    - Nose tip (point 30)
    - Chin (point 8)
    A well-aligned head will have a near-vertical nose-to-chin line.
    Returns the angle deviation from vertical (0 = perfect upright).
    """
    nose_tip = landmarks[30]
    chin     = landmarks[8]
    angle = calculate_angle(nose_tip, chin)
    # Vertical is -90 degrees; deviation from that
    deviation = abs(angle - (-90))
    if deviation > 180:
        deviation = 360 - deviation
    return deviation


def calculate_ear_shoulder_ratio(landmarks, frame_height):
    """
    Estimate forward head posture by comparing:
    - Ear position (avg of left ear point 0 and right ear point 16)
    - Eye level (avg of left eye corner 36 and right eye corner 45)

    When the head leans forward/down, ears drop relative to the frame height.
    We use the y-position of ears relative to frame height as a proxy.

    Returns a normalized ear_y ratio (higher = head lower / more forward).
    """
    left_ear  = landmarks[0]
    right_ear = landmarks[16]
    ear_y     = (left_ear[1] + right_ear[1]) / 2
    return ear_y / frame_height


def calculate_eye_level_tilt(landmarks):
    """
    Detect lateral head tilt by measuring the slope of the eye line.
    - Left eye outer corner: point 36
    - Right eye outer corner: point 45
    Returns the absolute angle of the eye line from horizontal.
    """
    left_eye  = landmarks[36]
    right_eye = landmarks[45]
    angle = abs(calculate_angle(left_eye, right_eye))
    if angle > 90:
        angle = 180 - angle
    return angle


def get_posture_state(landmarks, frame_height, thresholds):
    """
    Combine multiple metrics to determine posture state.

    Returns:
        state (str): "GOOD", "WARNING", or "SLOUCH"
        metrics (dict): individual metric values for display
    """
    head_tilt     = calculate_head_tilt(landmarks)
    ear_ratio     = calculate_ear_shoulder_ratio(landmarks, frame_height)
    eye_tilt      = calculate_eye_level_tilt(landmarks)

    metrics = {
        "head_tilt_deg": round(head_tilt, 1),
        "ear_y_ratio":   round(ear_ratio, 3),
        "eye_tilt_deg":  round(eye_tilt, 1),
    }

    # Determine state
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
    """
    Annotate the frame with:
    - Bounding box around face (color-coded by state)
    - Key landmark points (eyes, nose, chin, ears)
    - State label
    """
    colors = {
        "GOOD":    (56, 217, 169),   # teal-green
        "WARNING": (0, 200, 255),    # orange-yellow
        "SLOUCH":  (80, 80, 247),    # red
    }
    color = colors.get(state, (200, 200, 200))

    # Face bounding box
    x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Key landmark dots
    key_points = [0, 8, 16, 30, 36, 45]   # ears, chin, nose, eye corners
    for idx in key_points:
        pt = tuple(landmarks[idx].astype(int))
        cv2.circle(frame, pt, 4, color, -1)

    # Draw nose-to-chin line
    nose = tuple(landmarks[30].astype(int))
    chin = tuple(landmarks[8].astype(int))
    cv2.line(frame, nose, chin, color, 2)

    # Draw eye line
    leye = tuple(landmarks[36].astype(int))
    reye = tuple(landmarks[45].astype(int))
    cv2.line(frame, leye, reye, color, 1)

    return frame


# ── Default thresholds (tunable from GUI) ──
DEFAULT_THRESHOLDS = {
    "head_tilt_warn":  18.0,   # degrees deviation from vertical
    "head_tilt_bad":   30.0,
    "ear_ratio_warn":  0.42,   # ear y / frame height
    "ear_ratio_bad":   0.50,
    "eye_tilt_warn":   8.0,    # degrees from horizontal
}
