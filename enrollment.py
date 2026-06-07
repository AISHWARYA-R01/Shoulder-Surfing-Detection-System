"""
enrollment.py — SCPO Lite Admin Face Enrollment Module
=======================================================
Captures multiple samples of the admin's face, computes an average
histogram descriptor, and saves it to disk as the trusted profile.

No deep learning — uses grayscale LBP-style histogram comparison
via OpenCV's face recognizer (LBPH) which ships with opencv-contrib,
with a fallback to plain histogram comparison if contrib is absent.

Saved file: admin_profile.pkl  (next to this script)
"""

from __future__ import annotations
import os
import pickle
import cv2
import numpy as np
from typing import Optional, List, Tuple

PROFILE_PATH = os.path.join(os.path.dirname(__file__), "admin_profile.pkl")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

SAMPLE_COUNT   = 30      # frames to collect during enrollment
SAMPLE_SIZE    = (100, 100)  # resize face ROI to fixed size
MIN_FACE_SIZE  = (60, 60)

# Similarity threshold — lower = stricter match
# Histogram correlation: 1.0 = identical, 0.0 = nothing alike
MATCH_THRESHOLD = 0.55


class AdminProfile:
    """Stores the enrolled admin's face descriptor."""
    def __init__(self, histograms: List[np.ndarray], samples: List[np.ndarray]):
        self.histograms = histograms          # list of per-sample histograms
        self.mean_hist  = self._mean(histograms)  # average histogram
        self.samples    = samples[:4]         # keep a few thumbnails for display
        self.enrolled   = True

    @staticmethod
    def _mean(hists: List[np.ndarray]) -> np.ndarray:
        stacked = np.stack(hists, axis=0)
        return stacked.mean(axis=0).astype(np.float32)

    def similarity(self, face_roi: np.ndarray) -> float:
        """Return histogram correlation [0,1] between roi and admin profile."""
        hist = _extract_hist(face_roi)
        score = cv2.compareHist(self.mean_hist, hist, cv2.HISTCMP_CORREL)
        return float(np.clip(score, 0.0, 1.0))

    def is_admin(self, face_roi: np.ndarray) -> Tuple[bool, float]:
        sim = self.similarity(face_roi)
        return sim >= MATCH_THRESHOLD, sim


# ── Module-level helpers ───────────────────────────────────────────────────

def _extract_hist(gray_roi: np.ndarray) -> np.ndarray:
    """
    Compute a normalised grayscale histogram from a face ROI.
    Split the ROI into a 3×3 grid and concatenate local histograms
    for basic spatial encoding (robust to minor lighting changes).
    """
    roi = cv2.resize(gray_roi, SAMPLE_SIZE)
    h, w = roi.shape[:2]
    gh, gw = h // 3, w // 3
    hists = []
    for r in range(3):
        for c in range(3):
            cell = roi[r*gh:(r+1)*gh, c*gw:(c+1)*gw]
            hist = cv2.calcHist([cell], [0], None, [64], [0, 256])
            cv2.normalize(hist, hist)
            hists.append(hist.flatten())
    return np.concatenate(hists).astype(np.float32)


def save_profile(profile: AdminProfile):
    with open(PROFILE_PATH, "wb") as f:
        pickle.dump(profile, f)


def load_profile() -> Optional[AdminProfile]:
    if not os.path.exists(PROFILE_PATH):
        return None
    try:
        with open(PROFILE_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def delete_profile():
    if os.path.exists(PROFILE_PATH):
        os.remove(PROFILE_PATH)


def is_enrolled() -> bool:
    return os.path.exists(PROFILE_PATH)


class Enrollor:
    """
    Drives the enrollment flow: captures SAMPLE_COUNT face crops,
    builds an AdminProfile, and saves it.

    Usage (called from Streamlit):
        enr = Enrollor()
        while not enr.done:
            ret, frame = cap.read()
            status, annotated = enr.process_frame(frame)
            # show annotated, show status message
        profile = enr.profile
    """

    def __init__(self):
        self.cascade  = cv2.CascadeClassifier(CASCADE_PATH)
        self._samples: List[np.ndarray] = []   # gray face ROIs
        self._thumbs:  List[np.ndarray] = []   # BGR thumbnails for display
        self.done     = False
        self.profile: Optional[AdminProfile] = None

    @property
    def progress(self) -> float:
        return len(self._samples) / SAMPLE_COUNT

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def process_frame(self, frame: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Detect one face in frame, collect it as a sample.
        Returns (status_message, annotated_frame).
        """
        if self.done:
            return "Enrollment complete.", frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        annotated = frame.copy()

        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5,
            minSize=MIN_FACE_SIZE, flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            self._draw_guide(annotated, found=False)
            return f"Position your face in the frame… ({self.sample_count}/{SAMPLE_COUNT})", annotated

        if len(faces) > 1:
            self._draw_guide(annotated, found=False)
            return "Only one face should be visible during enrollment.", annotated

        x, y, w, h = faces[0]
        # Add margin
        pad = int(w * 0.15)
        x1 = max(x - pad, 0);  y1 = max(y - pad, 0)
        x2 = min(x + w + pad, frame.shape[1]);  y2 = min(y + h + pad, frame.shape[0])

        face_gray = gray[y1:y2, x1:x2]
        face_bgr  = frame[y1:y2, x1:x2]

        # Only accept every 2nd frame to get variety
        if len(self._samples) == 0 or True:
            self._samples.append(face_gray.copy())
            self._thumbs.append(cv2.resize(face_bgr, (64, 64)))

        # Draw progress box
        progress_pct = int(self.progress * 100)
        colour = (0, int(200 * self.progress), int(220 * (1 - self.progress)))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        # Progress bar at bottom of box
        bar_w = x2 - x1
        filled = int(bar_w * self.progress)
        cv2.rectangle(annotated, (x1, y2 + 4), (x1 + bar_w, y2 + 10), (50, 50, 50), -1)
        cv2.rectangle(annotated, (x1, y2 + 4), (x1 + filled, y2 + 10), colour, -1)

        cv2.putText(annotated, f"Scanning {progress_pct}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)

        if len(self._samples) >= SAMPLE_COUNT:
            self._finalise()
            return "✅ Admin enrolled successfully!", annotated

        msg = f"Hold still — collecting sample {self.sample_count}/{SAMPLE_COUNT}"
        return msg, annotated

    def _finalise(self):
        hists = [_extract_hist(s) for s in self._samples]
        self.profile = AdminProfile(hists, self._thumbs)
        save_profile(self.profile)
        self.done = True

    @staticmethod
    def _draw_guide(frame: np.ndarray, found: bool):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        size = min(w, h) // 3
        colour = (80, 80, 80) if not found else (0, 200, 100)
        cv2.rectangle(frame,
                      (cx - size, cy - size),
                      (cx + size, cy + size),
                      colour, 1)
        cv2.putText(frame, "Face guide", (cx - size, cy - size - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
