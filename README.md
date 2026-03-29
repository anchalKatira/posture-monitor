#  PostureGuard — Real-Time Posture Monitor

A Computer Vision project that uses your webcam and facial landmark detection to monitor your sitting posture in real time — alerting you with both sound and on-screen warnings when you start slouching.

---

#  Problem Statement

Extended sitting at desks — especially while studying or coding — leads to poor posture habits that most people don't notice in the moment. Chronic slouching causes neck pain, back strain, and long-term spinal issues. PostureGuard solves this by passively monitoring your posture through your laptop webcam and nudging you to correct it before the habit sets in.

---

#  Features

- **68-point facial landmark detection** using dlib
- **Multi-metric posture analysis** — head tilt, ear position, eye-line slope
- **Visual alert** — on-screen red banner when slouching is detected
- **Sound alert** — cross-platform beep on slouch detection
- **Live metrics panel** — real-time angle and ratio values
- **Session statistics** — good posture %, total alerts, session duration
- **Adjustable sensitivity** — tune head tilt limit and alert cooldown via sliders
- **Tkinter GUI** — clean dark-themed interface, no browser needed

---

# CV Concepts Used

| Concept | Application |
|---|---|
| Face Detection (HOG) | dlib's frontal face detector finds face region per frame |
| Facial Landmark Detection | 68-point shape predictor maps key face geometry |
| Angle Calculation | arctan2 of nose-chin vector gives head tilt deviation |
| Keypoint Geometry | Ear y-position normalized by frame height measures forward head lean |
| Eye Line Slope | Angle of left-right eye vector measures lateral tilt |
| Real-Time Video | Frame-by-frame webcam processing at ~30 FPS |
| Image Annotation | Bounding boxes, landmark dots, lines, and overlay banners |
| Alpha Blending | Semi-transparent alert overlay using cv2.addWeighted |

---

#  Project Structure

```
posture-monitor/
├── main.py                              # Tkinter GUI — entry point
├── posture.py                           # Landmark analysis, angle math, posture state
├── alert.py                             # Sound + visual alert manager with cooldown
├── requirements.txt                     # Python dependencies
├── shape_predictor_68_face_landmarks.dat  # dlib model (download separately)
└── README.md
```

---

# Setup & Installation

#Step 1 — Download the dlib landmark model

Download `shape_predictor_68_face_landmarks.dat.bz2` from:
```
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```
Extract it and place `shape_predictor_68_face_landmarks.dat` in the project folder.

# Step 2 — Install dependencies

**Windows:**
```bash
pip install cmake
pip install dlib
pip install -r requirements.txt
```

**Linux:**
```bash
sudo apt-get install cmake libopenblas-dev liblapack-dev
pip install -r requirements.txt
```

**macOS:**
```bash
brew install cmake
pip install -r requirements.txt
```

# Step 3 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/posture-monitor.git
cd posture-monitor
```

# Step 4 — Run
```bash
python main.py
```

---

#  How to Use

1. **Launch** the app — a dark-themed window will open
2. Click **▶ Start Monitoring** to activate the webcam
3. **Sit normally** in front of the camera (face clearly visible)
4. The system immediately begins tracking:
   -  **GOOD** — posture is fine
   -  **WARNING** — slightly off, keep an eye on it
   -  **SLOUCH** — alert fires (visual banner + beep)
5. Use the **sensitivity sliders** to adjust head tilt tolerance and alert frequency
6. Click **🔄 Reset Stats** to start a fresh session

---

#  How Posture Is Detected

PostureGuard uses three geometric signals from facial landmarks:

# 1. Head Tilt Angle
The angle of the **nose-to-chin line** from vertical. When you lean forward, this line tilts away from vertical. Deviation > threshold triggers an alert.

# 2. Ear Y-Ratio
The vertical position of your **ears** (landmark points 0 and 16) normalized by frame height. As your head droops forward, ears drop lower in the frame — this ratio increases.

# 3. Eye Line Tilt
The angle of the **left-to-right eye line** from horizontal. A tilted eye line indicates lateral head lean or slouch.

All three signals are combined into a **slouch score** — a sustained score above threshold triggers the alert.

---

#  Adjusting Sensitivity

- **Head Tilt Limit** — lower value = more sensitive (alerts earlier)
- **Alert Cooldown** — how many seconds between repeated alerts (prevents spam)
- **Slouch Frames Threshold** — hardcoded at 20 frames; edit in `main.py` if needed

---

# Tips for Best Results

- Sit **1–2 feet** from the camera with your face centered
- Use in **good lighting** — dlib's HOG detector needs clear contrast
- The `.dat` model file must be in the **same folder** as `main.py`
- Works best when **only one person** is in frame

---

#  Technologies Used

- **Python 3.x**
- **OpenCV** — video capture, frame annotation, alpha blending
- **dlib** — HOG face detection + 68-point facial landmark predictor
- **NumPy** — landmark array operations, angle math (arctan2)
- **Pillow (PIL)** — BGR→RGB conversion for Tkinter rendering
- **Tkinter** — cross-platform desktop GUI
- **winsound / subprocess** — cross-platform audio alert

---

#  License

MIT License — free to use, modify, and distribute.
