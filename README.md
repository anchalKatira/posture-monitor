## PostureGuard — Real-Time Posture Monitor

A Computer Vision project that uses your webcam to monitor sitting posture in real time — alerting you with both a visual on-screen warning and a sound beep when you start slouching.

---

##  Problem Statement

Extended sitting at desks — especially while studying or coding — leads to poor posture habits that most people don't notice in the moment. Chronic slouching causes neck pain, back strain, and long-term spinal issues. PostureGuard solves this by passively monitoring your posture through your webcam and alerting you to correct it before the habit sets in.

---

##  Features

-  **Real-time face detection** using OpenCV Haar Cascade
-  **Posture analysis** via face position tracking — Y position, size ratio, lateral offset
-  **Visual alert** — on-screen red banner when slouching is detected
-  **Sound alert** — beep on slouch detection (Windows)
-  **Live metrics panel** — real-time face position values
-  **Session statistics** — good posture %, total alerts, session duration
-  **Auto-calibration** — sits normally for 3 seconds to set baseline
-  **Tkinter GUI** — clean dark-themed interface

---

##  CV Concepts Used

| Concept | Application |
|---|---|
| Face Detection (Haar Cascade) | OpenCV's frontal face detector locates face bounding box per frame |
| Histogram Equalization | cv2.equalizeHist boosts contrast for detection in varied lighting |
| Bounding Box Geometry | Face Y position, width ratio, and center offset track posture change |
| Baseline Comparison | Good posture captured at startup; deviations trigger alerts |
| Real-Time Video Processing | Frames captured at ~30 FPS from webcam |
| Image Annotation | Bounding boxes, state labels, alert banners drawn per frame |
| Alpha Blending | Semi-transparent alert overlay using cv2.addWeighted |
| Frame Mirroring | cv2.flip for natural mirror-view experience |

---

##  Project Structure

```
posture-monitor/
├── main.py          # Complete app — GUI + detection + alerts
├── posture.py       # Face detection and posture metric logic
├── requirements.txt # Python dependencies
└── README.md
```

---

##  Setup & Installation

### Prerequisites
- Python 3.8 to 3.10 recommended
- Windows OS (for winsound beep alert)
- Webcam or phone camera via DroidCam

### Step 1 — Clone the repository
```bash
git clone https://github.com/anchalKatira/posture-monitor.git
cd posture-monitor
```

### Step 2 — Create virtual environment
```bash
py -3.10 -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run
```bash
python main.py
```

---

##  How to Use

1. Launch the app — dark GUI window opens
2. Click **Start Monitoring** — webcam activates
3. **Sit normally** and look at the camera — system auto-calibrates in 3 seconds
4. Once **"Baseline set!"** appears in status bar, monitoring is active
5. **Sit up straight** → shows GOOD in green ✅
6. **Lean forward** → shows WARNING in yellow ⚠️
7. **Keep slouching** → red banner + beep alert 🔔

Click **Reset Baseline** anytime to recalibrate for a new sitting position.

---

##  How Posture Is Detected

PostureGuard uses three geometric signals from the face bounding box:

**1. Face Y Position**
When you slouch forward, your head moves downward in the frame. The face Y coordinate increases relative to the calibrated baseline — this is the primary slouch signal.

**2. Face Size Ratio**
Face width divided by frame width. When you lean closer to the camera (forward slouch), the face appears larger — this ratio increases beyond the baseline.

**3. Lateral Offset**
Distance of face center from frame center. Significant sideways drift indicates lateral tilt or asymmetric posture.

All three signals contribute to a **slouch score**. Score >= 2 triggers SLOUCH state, sustained for 10 frames fires the alert.

---

##  Tips for Best Results

- Sit **1.5–2 feet** from the camera with face centered
- Camera should be at **eye level** — not looking up or down
- Use in **good front lighting** — avoid strong backlight
- Look **straight at the camera** during calibration

---

##  Technologies Used

- **Python 3.x**
- **OpenCV** — face detection, frame annotation, alpha blending, histogram equalization
- **Pillow (PIL)** — BGR to RGB conversion for Tkinter rendering
- **NumPy** — array operations
- **Tkinter** — cross-platform desktop GUI
- **winsound** — Windows audio alert

---

##  License

MIT License — free to use, modify, and distribute.
