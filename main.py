import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import numpy as np
import winsound

BG_DARK  = "#0d1117"
BG_CARD  = "#161b22"
GOOD_C   = "#38d9a9"
WARN_C   = "#f59f00"
BAD_C    = "#f76f6f"
ACCENT   = "#4f8ef7"
TEXT_PRI = "#e6edf3"
TEXT_SEC = "#7d8590"
BORDER   = "#30363d"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def analyze_posture(fx, fy, fw, fh, frame_w, frame_h, baseline):
    cx = fx + fw // 2
    cy = fy + fh // 2
    size_ratio = fw / frame_w
    metrics = {
        "face_y_pct":    round(cy / frame_h * 100, 1),
        "face_size_pct": round(size_ratio * 100, 1),
        "center_offset": round(abs(cx - frame_w // 2) / frame_w * 100, 1),
    }
    if baseline is None:
        return "GOOD", metrics, {"y": cy, "size": size_ratio}

    y_drop     = cy - baseline["y"]
    size_change = size_ratio - baseline["size"]
    x_offset   = abs(cx - frame_w // 2) / frame_w * 100

    score = 0
    if y_drop > frame_h * 0.04:   score += 2
    elif y_drop > frame_h * 0.02: score += 1
    if size_change > 0.6:         score += 1
    if x_offset > 15:              score += 1

    state = "SLOUCH" if score >= 2 else "WARNING" if score >= 1 else "GOOD"
    return state, metrics, baseline


class PostureApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PostureGuard — Real-Time Posture Monitor")
        self.geometry("1100x700")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        self.cap = None
        self.running = False
        self.baseline = None
        self.alert_streak = 0
        self.alert_count = 0
        self.total_frames = 0
        self.good_frames = 0
        self.session_start = None
        self.last_alert = 0
        self.warmup = 0

        self.status_var  = tk.StringVar(value="Ready. Click Start Monitoring.")
        self.posture_var = tk.StringVar(value="—")
        self.y_var       = tk.StringVar(value="—")
        self.size_var    = tk.StringVar(value="—")
        self.offset_var  = tk.StringVar(value="—")
        self.good_var    = tk.StringVar(value="—")
        self.alert_var   = tk.StringVar(value="0")
        self.session_var = tk.StringVar(value="00:00")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick()

    def _build_ui(self):
        hdr = tk.Frame(self, bg=BG_DARK)
        hdr.pack(fill="x", padx=20, pady=(16, 0))
        tk.Label(hdr, text="PostureGuard", font=("Segoe UI", 20, "bold"), bg=BG_DARK, fg=TEXT_PRI).pack(side="left")
        tk.Label(hdr, text="Real-Time Posture Monitoring via Webcam", font=("Segoe UI", 10), bg=BG_DARK, fg=TEXT_SEC).pack(side="left", padx=12)

        main = tk.Frame(self, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=20, pady=12)

        left = tk.Frame(main, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
        left.pack(side="left", fill="both", expand=True)
        self.video_label = tk.Label(left, bg=BG_CARD, text="Camera feed will appear here", fg=TEXT_SEC, font=("Segoe UI", 11))
        self.video_label.pack(fill="both", expand=True, padx=8, pady=8)

        right = tk.Frame(main, bg=BG_CARD, width=290, highlightthickness=1, highlightbackground=BORDER)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)
        self._build_panel(right)

        sb = tk.Frame(self, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
        sb.pack(fill="x", padx=20, pady=(0, 10))
        tk.Label(sb, textvariable=self.status_var, font=("Consolas", 9), bg=BG_CARD, fg=ACCENT, anchor="w").pack(side="left", padx=10, pady=5)

    def _build_panel(self, p):
        pad = {"padx": 14, "pady": 4}
        tk.Label(p, text="POSTURE STATE", font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg=TEXT_SEC).pack(anchor="w", padx=14, pady=(8,2))
        self.badge = tk.Label(p, textvariable=self.posture_var, font=("Segoe UI", 16, "bold"), bg=BG_CARD, fg=GOOD_C)
        self.badge.pack(**pad)
        ttk.Separator(p, orient="horizontal").pack(fill="x", padx=14, pady=6)

        tk.Label(p, text="CAMERA", font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg=TEXT_SEC).pack(anchor="w", padx=14, pady=(8,2))
        self.btn_start = tk.Button(p, text="Start Monitoring", bg=ACCENT, fg=TEXT_PRI, font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2", bd=0, pady=7, command=self._start)
        self.btn_start.pack(fill="x", **pad)
        self.btn_stop = tk.Button(p, text="Stop", bg="#444", fg=TEXT_PRI, font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2", bd=0, pady=7, command=self._stop, state="disabled")
        self.btn_stop.pack(fill="x", **pad)
        tk.Button(p, text="Reset Baseline", bg="#2a2d3e", fg=TEXT_PRI, font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2", bd=0, pady=7, command=self._reset_baseline).pack(fill="x", **pad)
        ttk.Separator(p, orient="horizontal").pack(fill="x", padx=14, pady=6)

        tk.Label(p, text="LIVE METRICS", font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg=TEXT_SEC).pack(anchor="w", padx=14, pady=(8,2))
        mf = tk.Frame(p, bg=BG_CARD)
        mf.pack(fill="x", padx=14, pady=4)
        for lbl, var in [("Face Y pos", self.y_var), ("Face size", self.size_var), ("Side offset", self.offset_var)]:
            row = tk.Frame(mf, bg=BG_CARD)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=lbl, font=("Consolas", 9), bg=BG_CARD, fg=TEXT_SEC, width=12, anchor="w").pack(side="left")
            tk.Label(row, textvariable=var, font=("Consolas", 10), bg=BG_CARD, fg=TEXT_PRI).pack(side="left")
        ttk.Separator(p, orient="horizontal").pack(fill="x", padx=14, pady=6)

        tk.Label(p, text="SESSION STATS", font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg=TEXT_SEC).pack(anchor="w", padx=14, pady=(8,2))
        for lbl, var in [("Duration", self.session_var), ("Good posture", self.good_var), ("Alerts fired", self.alert_var)]:
            row = tk.Frame(p, bg=BG_CARD)
            row.pack(fill="x", padx=14, pady=2)
            tk.Label(row, text=lbl, font=("Consolas", 9), bg=BG_CARD, fg=TEXT_SEC, width=14, anchor="w").pack(side="left")
            tk.Label(row, textvariable=var, font=("Consolas", 10), bg=BG_CARD, fg=TEXT_PRI).pack(side="left")
        ttk.Separator(p, orient="horizontal").pack(fill="x", padx=14, pady=6)

        tk.Label(p, text="Sit normally -> auto-calibrates\nin 3 sec. Then slouch to test!",
                 font=("Segoe UI", 9), bg=BG_CARD, fg=TEXT_SEC, justify="left").pack(padx=14, pady=4, anchor="w")

    def _start(self):
        self.cap = None
        for idx in [1, 0, 2]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    break
            cap.release()
        if not self.cap:
            messagebox.showerror("Camera Error", "No camera found. Make sure DroidCam is open and connected.")
            return
        self.running = True
        self.warmup = 0
        self.baseline = None
        self.session_start = time.time()
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status_var.set("Camera active. Sit normally — calibrating in 3 seconds...")
        threading.Thread(target=self._loop, daemon=True).start()

    def _stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.config(image="", text="Camera stopped.", fg=TEXT_SEC)
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.posture_var.set("—")
        self.status_var.set("Stopped.")

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            display = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

            state = "NO_FACE"
            metrics = {}

            if len(faces) > 0:
                self.warmup += 1
                fx, fy, fw, fh = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]

                if self.warmup == 30:
                    self.baseline = {"y": fy + fh//2, "size": fw/w}
                    self.after(0, lambda: self.status_var.set("Baseline set! Monitoring active. Slouch forward to test the alert."))

                state, metrics, self.baseline = analyze_posture(fx, fy, fw, fh, w, h, self.baseline)

                col = {"GOOD": (56,217,169), "WARNING": (0,200,255), "SLOUCH": (80,80,247)}.get(state, (200,200,200))
                cv2.rectangle(display, (fx, fy), (fx+fw, fy+fh), col, 2)
                cv2.rectangle(display, (fx, fy-30), (fx+fw, fy), col, -1)
                cv2.putText(display, state, (fx+6, fy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cx2, cy2 = fx+fw//2, fy+fh//2
                cv2.circle(display, (cx2, cy2), 4, col, -1)

                self.total_frames += 1
                if state == "GOOD":
                    self.good_frames += 1
                    self.alert_streak = 0
                elif state == "SLOUCH":
                    self.alert_streak += 1
                    if self.alert_streak >= 25 and time.time() - self.last_alert > 8:
                        self.last_alert = time.time()
                        self.alert_count += 1
                        self.alert_streak = 0
                        threading.Thread(target=lambda: winsound.Beep(880, 600), daemon=True).start()

                if state == "SLOUCH" and self.alert_streak > 15:
                    ov = display.copy()
                    cv2.rectangle(ov, (0, 0), (w, 55), (0, 0, 180), -1)
                    cv2.addWeighted(ov, 0.7, display, 0.3, 0, display)
                    cv2.putText(display, "SLOUCH DETECTED - Sit Up Straight!", (12, 38), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255), 2)
            else:
                cv2.putText(display, "No face — look straight at camera", (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120,120,120), 1)

            dot = {"GOOD":(56,217,169),"WARNING":(0,200,255),"SLOUCH":(80,80,247),"NO_FACE":(80,80,80)}.get(state,(80,80,80))
            cv2.circle(display, (w-20, 20), 10, dot, -1)

            pct = self.good_frames / max(self.total_frames, 1) * 100
            self.after(0, self._update_ui, state, metrics, pct)

            rgb  = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img  = Image.fromarray(rgb)
            lw   = self.video_label.winfo_width() or 720
            lh   = self.video_label.winfo_height() or 500
            img  = img.resize((lw, lh), Image.LANCZOS)
            imgt = ImageTk.PhotoImage(image=img)
            self.after(0, self._set_frame, imgt)
            time.sleep(0.033)

    def _set_frame(self, imgt):
        self.video_label.imgtk = imgt
        self.video_label.config(image=imgt, text="")

    def _update_ui(self, state, metrics, pct):
        cols  = {"GOOD": GOOD_C, "WARNING": WARN_C, "SLOUCH": BAD_C, "NO_FACE": TEXT_SEC}
        labls = {"GOOD": "GOOD", "WARNING": "WARNING", "SLOUCH": "SLOUCH", "NO_FACE": "No Face"}
        self.posture_var.set(labls.get(state, state))
        self.badge.config(fg=cols.get(state, TEXT_SEC))
        if metrics:
            self.y_var.set(f"{metrics.get('face_y_pct','—')}%")
            self.size_var.set(f"{metrics.get('face_size_pct','—')}%")
            self.offset_var.set(f"{metrics.get('center_offset','—')}%")
        self.good_var.set(f"{pct:.1f}%")
        self.alert_var.set(str(self.alert_count))
        msgs = {"GOOD": "Great posture! Keep it up.", "WARNING": "Posture slightly off.", "SLOUCH": "Slouching! Sit straight.", "NO_FACE": "Look straight at the camera."}
        self.status_var.set(msgs.get(state, ""))

    def _reset_baseline(self):
        self.baseline = None
        self.warmup = 0
        self.status_var.set("Baseline reset. Sit normally to re-calibrate...")

    def _tick(self):
        if self.session_start and self.running:
            e = int(time.time() - self.session_start)
            self.session_var.set(f"{e//60:02d}:{e%60:02d}")
        self.after(1000, self._tick)

    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = PostureApp()
    app.mainloop()
