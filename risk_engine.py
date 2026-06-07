"""
risk_engine.py — SCPO Lite Visual Exposure Risk Engine (VER-E)
==============================================================
Risk is now based on STRANGER faces only, not total face count.

  Admin alone         → SAFE  (score ≈ 0)
  No one enrolled yet → WATCH (unknown baseline)
  Stranger present    → WATCH / DANGER depending on behavior
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
from detector import DetectionResult

W_STRANGER    = 0.30
W_PROXIMITY   = 0.30
W_DWELL       = 0.20
W_MOTION      = 0.10
W_ORIENTATION = 0.10

MAX_STRANGERS    = 3
DWELL_SATURATION = 90
MOTION_THRESHOLD = 25.0
EMA_ALPHA        = 0.35


@dataclass
class RiskAssessment:
    raw_score:         float
    smooth_score:      float
    level:             str
    level_code:        int
    components:        dict
    face_count:        int
    stranger_count:    int
    admin_present:     bool
    dominant_proximity: float
    is_intrusion:      bool

    @property
    def color(self) -> str:
        return {"SAFE":"#00e676","WATCH":"#ffab00","DANGER":"#ff1744"}[self.level]

    @property
    def emoji(self) -> str:
        return {"SAFE":"🟢","WATCH":"🟡","DANGER":"🔴"}[self.level]

    @property
    def status_line(self) -> str:
        if self.admin_present and self.stranger_count == 0:
            return "Admin verified — environment secure"
        elif not self.admin_present and self.face_count == 0:
            return "No faces detected"
        elif not self.admin_present and self.face_count > 0:
            return f"{self.face_count} unrecognised face(s) — admin not detected"
        else:
            return f"Admin present + {self.stranger_count} stranger(s) detected"


class RiskEngine:
    def __init__(self):
        self._smooth_score:    float = 0.0
        self._danger_frames:   int   = 0
        self._sustain_thresh:  int   = 15
        self._history:         List[float] = []

    def evaluate(self, result: DetectionResult) -> RiskAssessment:
        components = self._compute_components(result)
        raw        = self._composite(components) * 100.0

        self._smooth_score = EMA_ALPHA * raw + (1 - EMA_ALPHA) * self._smooth_score
        smooth = round(min(self._smooth_score, 100.0), 1)
        raw    = round(min(raw, 100.0), 1)

        level, code = self._classify(smooth)

        if code == 2:
            self._danger_frames += 1
        else:
            self._danger_frames = max(0, self._danger_frames - 2)

        is_intrusion = self._danger_frames >= self._sustain_thresh

        self._history.append(smooth)
        if len(self._history) > 300:
            self._history.pop(0)

        return RiskAssessment(
            raw_score=raw,
            smooth_score=smooth,
            level=level,
            level_code=code,
            components={k: round(v*100,1) for k,v in components.items()},
            face_count=result.face_count,
            stranger_count=result.stranger_count,
            admin_present=result.admin_present,
            dominant_proximity=round(result.max_proximity, 3),
            is_intrusion=is_intrusion,
        )

    def reset(self):
        self._smooth_score  = 0.0
        self._danger_frames = 0

    @property
    def score_history(self) -> List[float]:
        return list(self._history)

    def _compute_components(self, result: DetectionResult) -> dict:
        strangers = [f for f in result.faces if not f.is_admin]
        n = len(strangers)

        # 1. Stranger count (admin alone = 0 risk)
        stranger_score = min(n / MAX_STRANGERS, 1.0)

        # 2. Proximity of closest stranger
        proximity_score = max((f.proximity_score for f in strangers), default=0.0)

        # 3. Dwell time of most persistent stranger
        dwell_score = 0.0
        if strangers:
            max_dwell = max(f.dwell_frames for f in strangers)
            dwell_score = min(max_dwell / DWELL_SATURATION, 1.0)

        # 4. Motion (inverse — slow = higher sustained risk)
        motion_score = 0.0
        if strangers:
            avg_motion = np.mean([f.motion_delta for f in strangers])
            motion_score = max(0.0, 1.0 - avg_motion / MOTION_THRESHOLD)

        # 5. Orientation of strangers
        orientation_score = 0.0
        if strangers:
            orientation_score = float(np.mean([f.orientation_score for f in strangers]))

        return {
            "strangers":   stranger_score,
            "proximity":   proximity_score,
            "dwell":       dwell_score,
            "motion":      motion_score,
            "orientation": orientation_score,
        }

    @staticmethod
    def _composite(c: dict) -> float:
        return (
            W_STRANGER    * c["strangers"]   +
            W_PROXIMITY   * c["proximity"]   +
            W_DWELL       * c["dwell"]       +
            W_MOTION      * c["motion"]      +
            W_ORIENTATION * c["orientation"]
        )

    @staticmethod
    def _classify(score: float):
        if score <= 30:  return "SAFE",   0
        if score <= 70:  return "WATCH",  1
        return                  "DANGER", 2
