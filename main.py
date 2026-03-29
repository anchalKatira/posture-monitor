import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import numpy as np

from posture import (
    load_predictor, get_landmarks, get_posture_state,
    draw_landmarks_on_frame, DEFAULT_THRESHOLDS
)
from alert import AlertManager

# ── Theme ──
BG_DARK   = "#0d1117"
BG_CARD   = "#161b22"
BG_PANEL  = "#0d1117"
GOOD_C    = "#38d9a9"
WARN_C    = "#f59f00"
BAD_C     = "#f76f6f"
ACCENT    = "#4f8ef7"
TEXT_PRI  = "#e6edf3"
TEXT_SEC  = "#7d8590"
BORDER    = "#30363d"

FONT_HEAD = ("Segoe UI", 18, "bold")
FONT_SUB  = ("Segoe UI", 10)
FONT_MONO = ("Consolas", 10)
FONT_SM   = ("Consolas", 9)


class PostureApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PostureGuard — Real-Time Posture Monitor")
        self.geometry("1150x720")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        # Load dlib predictor
        try:
            load_predictor()
        except FileNotFoundError as e:
            messagebox.showerror("Missing Model File", str(e))
            self.destroy()
            return

        # State
        self.cap        = None
        self.running    = False
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        self.alert_mgr  = AlertManager(
            sound_enabled=True,
            visual_enabled=True,
            cooldown_seconds=8,
            slouch_frames_threshold=20
        )

        # Stats
        self.total_frames  = 0
        self.good_frames   = 0
        self.warn_frames   = 0
        self.bad_frames    = 0
        self.session_start = None

        # Tkinter vars
        self.status_var    = tk.StringVar(value="Ready. Click 'Start' to begin monitoring.")
        self.posture_var   = tk.StringVar(value="—")
        self.score_var     = tk.StringVar(value="—")
        self.tilt_var      = tk.StringVar(value="—")
        self.ear_var       = tk.StringVar(value="—")
        self.eye_var       = tk.StringVar(value="—")
        self.session_var   = tk.StringVar(value="00:00")
        self.good_pct_var  = tk.StringVar(value="—")
        self.alerts_var    = tk.StringVar(value="0")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick_clock()

    # ─────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG_DARK)
        hdr.pack(fill="x", padx=20, pady=(16, 0))
        tk.Label(hdr, text="🧍 PostureGuard", font=FONT_HEAD,
                 bg=BG_DARK, fg=TEXT_PRI).pack(side="left")
        tk.Label(hdr, text="Real-Time Posture Monitoring via Webcam",
                 font=FONT_SUB, bg=BG_DARK, fg=TEXT_SEC).pack(side="left", padx=12)

        # Main layout
        main = tk.Frame(self, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=20, pady=12)

        # Left — video
        left = tk.Frame(main, bg=BG_CARD,
                        highlightthickness=1, highlightbackground=BORDER)
        left.pack(side="left", fill="both", expand=True)

        self.video_label = tk.Label(
            left, bg=BG_CARD,
            text="📷  Camera feed will appear here",
            fg=TEXT_SEC, font=FONT_SUB
        )
        self.video_label.pack(fill="both", expand=True, padx=8, pady=8)

        # Right — panel
        right = tk.Frame(main, bg=BG_CARD, width=290,
                         highlightthickness=1, highlightbackground=BORDER)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)
        self._build_panel(right)

        # Status bar
        sb = tk.Frame(self, bg=BG_CARD, height=30,
                      highlightthickness=1, highlightbackground=BORDER)
        sb.pack(fill="x", padx=20, pady=(0, 10))
        tk.Label(sb, textvariable=self.status_var, font=FONT_SM,
                 bg=BG_CARD, fg=ACCENT, anchor="w").pack(side="left", padx=10, pady=5)

    def _build_panel(self, parent):
        p = {"padx": 14, "pady": 4}

        # ── Posture state ──
        self._sec(parent, "POSTURE STATE")
        self.posture_badge = tk.Label(
            parent, textvariable=self.posture_var,
            font=("Segoe UI", 16, "bold"), bg=BG_CARD, fg=GOOD_C
        )
        self.posture_badge.pack(**p)

        self._sep(parent)

        # ── Camera controls ──
        self._sec(parent, "CAMERA")
        self.btn_start = self._btn(parent, "▶  Start Monitoring", ACCENT, self._start)
        self.btn_start.pack(fill="x", **p)
        self.btn_stop = self._btn(parent, "⏹  Stop", "#444", self._stop, state="disabled")
        self.btn_stop.pack(fill="x", **p)
        self._btn(parent, "🔄  Reset Stats", "#2a2d3e", self._reset_stats).pack(fill="x", **p)

        self._sep(parent)

        # ── Live metrics ──
        self._sec(parent, "LIVE METRICS")
        metrics_frame = tk.Frame(parent, bg=BG_CARD)
        metrics_frame.pack(fill="x", padx=14, pady=4)

        rows = [
            ("Head Tilt",  self.tilt_var),
            ("Ear Y Ratio", self.ear_var),
            ("Eye Tilt",   self.eye_var),
        ]
        for label, var in rows:
            row = tk.Frame(metrics_frame, bg=BG_CARD)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, font=FONT_SM, bg=BG_CARD,
                     fg=TEXT_SEC, width=12, anchor="w").pack(side="left")
            tk.Label(row, textvariable=var, font=FONT_MONO,
                     bg=BG_CARD, fg=TEXT_PRI).pack(side="left")

        self._sep(parent)

        # ── Session stats ──
        self._sec(parent, "SESSION STATS")
        stats = [
            ("⏱  Duration",     self.session_var),
            ("✅  Good posture", self.good_pct_var),
            ("🔔  Alerts fired", self.alerts_var),
        ]
        for label, var in stats:
            row = tk.Frame(parent, bg=BG_CARD)
            row.pack(fill="x", padx=14, pady=2)
            tk.Label(row, text=label, font=FONT_SM, bg=BG_CARD,
                     fg=TEXT_SEC, width=16, anchor="w").pack(side="left")
            tk.Label(row, textvariable=var, font=FONT_MONO,
                     bg=BG_CARD, fg=TEXT_PRI).pack(side="left")

        self._sep(parent)

        # ── Threshold tuning ──
        self._sec(parent, "SENSITIVITY")
        sens_frame = tk.Frame(parent, bg=BG_CARD)
        sens_frame.pack(fill="x", padx=14, pady=4)

        tk.Label(sens_frame, text="Head Tilt Limit (°)",
                 font=FONT_SM, bg=BG_CARD, fg=TEXT_SEC).pack(anchor="w")
        self.tilt_slider = tk.Scale(
            sens_frame, from_=10, to=50, orient="horizontal",
            bg=BG_CARD, fg=TEXT_PRI, troughcolor=BORDER,
            highlightthickness=0, bd=0,
            command=self._update_thresholds
        )
        self.tilt_slider.set(int(self.thresholds["head_tilt_bad"]))
        self.tilt_slider.pack(fill="x")

        tk.Label(sens_frame, text="Alert Cooldown (sec)",
                 font=FONT_SM, bg=BG_CARD, fg=TEXT_SEC).pack(anchor="w", pady=(6, 0))
        self.cooldown_slider = tk.Scale(
            sens_frame, from_=3, to=30, orient="horizontal",
            bg=BG_CARD, fg=TEXT_PRI, troughcolor=BORDER,
            highlightthickness=0, bd=0,
            command=self._update_cooldown
        )
        self.cooldown_slider.set(8)
        self.cooldown_slider.pack(fill="x")

    def _sec(self, parent, text):
        tk.Label(parent, text=text, font=("Segoe UI", 8, "bold"),
                 bg=BG_CARD, fg=TEXT_SEC).pack(anchor="w", padx=14, pady=(8, 2))

    def _sep(self, parent):
        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=14, pady=6)

    def _btn(self, parent, text, color, cmd, state="normal"):
        return tk.Button(
            parent, text=text, bg=color, fg=TEXT_PRI,
            font=("Segoe UI", 10, "bold"), relief="flat",
            cursor="hand2", activebackground=color,
            bd=0, pady=7, command=cmd, state=state
        )

    # ─────────────────────────────────────────────
    # CAMERA CONTROL
    # ─────────────────────────────────────────────
    def _start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam.")
            return
        self.running = True
        self.session_start = time.time()
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.alert_mgr.reset()
        self.status_var.set("Monitoring active. Sit comfortably and face the camera.")
        threading.Thread(target=self._video_loop, daemon=True).start()

    def _stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.config(image="", text="📷  Monitoring stopped.", fg=TEXT_SEC)
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.posture_var.set("—")
        self.status_var.set("Monitoring stopped.")

    # ─────────────────────────────────────────────
    # VIDEO LOOP
    # ─────────────────────────────────────────────
    def _video_loop(self):
        alert_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks, face_rect = get_landmarks(gray)

            state   = "NO_FACE"
            metrics = {}

            if landmarks is not None:
                state, metrics = get_posture_state(
                    landmarks, frame.shape[0], self.thresholds
                )
                frame = draw_landmarks_on_frame(frame, landmarks, face_rect, state)

                # Update stats
                self.total_frames += 1
                if state == "GOOD":
                    self.good_frames += 1
                elif state == "WARNING":
                    self.warn_frames += 1
                else:
                    self.bad_frames += 1

                # Alert manager
                banner, fired = self.alert_mgr.update(state)
                if fired:
                    alert_count += 1

                # Draw banner overlay on frame
                if banner:
                    text, color, _ = banner
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), color, -1)
                    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
                    cv2.putText(frame, text, (16, 42),
                                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

            else:
                # No face — draw subtle message
                cv2.putText(frame, "No face detected", (16, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)

            # Draw state indicator in corner
            state_colors = {
                "GOOD":    (56, 217, 169),
                "WARNING": (0, 200, 255),
                "SLOUCH":  (80, 80, 247),
                "NO_FACE": (120, 120, 120),
            }
            sc = state_colors.get(state, (200, 200, 200))
            cv2.circle(frame, (frame.shape[1] - 24, 24), 12, sc, -1)

            # UI updates via main thread
            pct = (self.good_frames / max(self.total_frames, 1)) * 100
            self.after(0, self._update_ui, state, metrics, pct, alert_count)

            # Convert frame for Tkinter
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img  = Image.fromarray(rgb)
            w    = self.video_label.winfo_width()  or 720
            h    = self.video_label.winfo_height() or 500
            img  = img.resize((w, h), Image.LANCZOS)
            imgt = ImageTk.PhotoImage(image=img)
            self.after(0, self._set_frame, imgt)

            time.sleep(0.033)  # ~30 FPS

    def _set_frame(self, imgt):
        self.video_label.imgtk = imgt
        self.video_label.config(image=imgt, text="")

    def _update_ui(self, state, metrics, good_pct, alert_count):
        colors = {
            "GOOD":    GOOD_C,
            "WARNING": WARN_C,
            "SLOUCH":  BAD_C,
            "NO_FACE": TEXT_SEC,
        }
        labels = {
            "GOOD":    "✅  GOOD",
            "WARNING": "⚠️  WARNING",
            "SLOUCH":  "❌  SLOUCH",
            "NO_FACE": "👤  No Face",
        }
        self.posture_var.set(labels.get(state, state))
        self.posture_badge.config(fg=colors.get(state, TEXT_SEC))

        if metrics:
            self.tilt_var.set(f"{metrics.get('head_tilt_deg', '—')}°")
            self.ear_var.set(f"{metrics.get('ear_y_ratio', '—')}")
            self.eye_var.set(f"{metrics.get('eye_tilt_deg', '—')}°")

        self.good_pct_var.set(f"{good_pct:.1f}%")
        self.alerts_var.set(str(alert_count))

        if state == "SLOUCH":
            self.status_var.set("⚠ Poor posture detected! Straighten your back and lift your head.")
        elif state == "WARNING":
            self.status_var.set("⚡ Posture slightly off — adjust your sitting position.")
        elif state == "GOOD":
            self.status_var.set("✅ Great posture! Keep it up.")
        elif state == "NO_FACE":
            self.status_var.set("👤 Face not visible — position yourself in front of the camera.")

    # ─────────────────────────────────────────────
    # CLOCK & STATS
    # ─────────────────────────────────────────────
    def _tick_clock(self):
        if self.session_start and self.running:
            elapsed = int(time.time() - self.session_start)
            m, s = divmod(elapsed, 60)
            self.session_var.set(f"{m:02d}:{s:02d}")
        self.after(1000, self._tick_clock)

    def _reset_stats(self):
        self.total_frames = self.good_frames = self.warn_frames = self.bad_frames = 0
        self.session_start = time.time() if self.running else None
        self.alert_mgr.reset()
        self.good_pct_var.set("—")
        self.alerts_var.set("0")
        self.session_var.set("00:00")

    # ─────────────────────────────────────────────
    # SETTINGS
    # ─────────────────────────────────────────────
    def _update_thresholds(self, val):
        v = float(val)
        self.thresholds["head_tilt_bad"]  = v
        self.thresholds["head_tilt_warn"] = max(10, v - 10)

    def _update_cooldown(self, val):
        self.alert_mgr.cooldown = int(val)

    # ─────────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────────
    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = PostureApp()
    app.mainloop()
