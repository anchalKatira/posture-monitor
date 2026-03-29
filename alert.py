import time
import sys
import threading

# ── Cross-platform beep ──
def _beep_sound(frequency=880, duration=400):
    """Play a beep sound cross-platform."""
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(frequency, duration)
        elif sys.platform == "darwin":
            import subprocess
            subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"],
                           capture_output=True)
        else:
            # Linux — try multiple fallbacks
            import subprocess
            try:
                subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                               capture_output=True, timeout=1)
            except Exception:
                try:
                    subprocess.run(["aplay", "-q", "/usr/share/sounds/alsa/Front_Left.wav"],
                                   capture_output=True, timeout=1)
                except Exception:
                    print("\a", end="", flush=True)  # terminal bell fallback
    except Exception:
        print("\a", end="", flush=True)


class AlertManager:
    """
    Manages alert state, cooldowns, and streak counting.
    Prevents spam — only fires alerts after sustained slouching.
    """

    def __init__(self,
                 sound_enabled=True,
                 visual_enabled=True,
                 cooldown_seconds=8,
                 slouch_frames_threshold=20):
        """
        Args:
            sound_enabled: play beep on alert
            visual_enabled: show on-screen banner
            cooldown_seconds: minimum gap between consecutive alerts
            slouch_frames_threshold: how many consecutive slouch frames before alerting
        """
        self.sound_enabled   = sound_enabled
        self.visual_enabled  = visual_enabled
        self.cooldown        = cooldown_seconds
        self.frames_threshold = slouch_frames_threshold

        self._last_alert_time = 0
        self._slouch_streak   = 0    # consecutive frames with bad posture
        self._warning_streak  = 0
        self._good_streak     = 0

        self.current_banner   = None  # (text, color, expiry_time)
        self._lock            = threading.Lock()

    def update(self, state):
        """
        Call once per frame with the current posture state.
        Returns banner info if an alert should be shown, else None.
        """
        with self._lock:
            if state == "SLOUCH":
                self._slouch_streak  += 1
                self._warning_streak += 1
                self._good_streak     = 0
            elif state == "WARNING":
                self._warning_streak += 1
                self._slouch_streak   = max(0, self._slouch_streak - 1)
                self._good_streak     = 0
            else:  # GOOD
                self._good_streak    += 1
                self._slouch_streak   = max(0, self._slouch_streak - 2)
                self._warning_streak  = max(0, self._warning_streak - 2)

            now = time.time()
            fired = False

            # Fire alert if slouch sustained and cooldown passed
            if (self._slouch_streak >= self.frames_threshold and
                    now - self._last_alert_time >= self.cooldown):

                self._last_alert_time = now
                self._slouch_streak   = 0
                fired = True

                if self.sound_enabled:
                    threading.Thread(target=_beep_sound, args=(880, 500), daemon=True).start()

                self.current_banner = (
                    "⚠  SLOUCH DETECTED — Sit Up Straight!",
                    (0, 0, 200),       # red in BGR
                    now + 3.0          # show for 3 seconds
                )

            # Clear expired banner
            if self.current_banner and time.time() > self.current_banner[2]:
                self.current_banner = None

            return self.current_banner, fired

    def get_streaks(self):
        with self._lock:
            return {
                "slouch": self._slouch_streak,
                "warning": self._warning_streak,
                "good": self._good_streak,
            }

    def reset(self):
        with self._lock:
            self._slouch_streak   = 0
            self._warning_streak  = 0
            self._good_streak     = 0
            self._last_alert_time = 0
            self.current_banner   = None
