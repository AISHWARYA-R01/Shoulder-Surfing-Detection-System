"""
alerts.py — SCPO Lite Privacy Shield & Alert Module  [FIXED]
=============================================================
FIX 1: _determine_shield() — PRIVACY shield now activates at any DANGER
        level (level_code == 2), not only when score >= 85.  Previously
        the shield would only reach PRIVACY for extremely high scores,
        meaning most DANGER situations only showed a blur.

FIX 2: on_stranger_entry callback — fires the first time a stranger is
        detected (stranger_count transitions 0 → >0), giving app.py a
        clean hook to trigger screenshot saves independently of intrusion
        (which requires 15 sustained DANGER frames).

Uses RiskAssessment.stranger_count (not observer_count).
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable

import cv2
import numpy as np
from risk_engine import RiskAssessment


class ProtectionMode(str, Enum):
    FOCUS  = "Focus"
    NORMAL = "Normal"
    SILENT = "Silent"


class ShieldType(str, Enum):
    NONE    = "none"
    WARNING = "warning"
    BLUR    = "blur"
    PRIVACY = "privacy"


@dataclass
class AlertState:
    active:      bool       = False
    shield:      ShieldType = ShieldType.NONE
    message:     str        = ""
    color_hex:   str        = "#00e676"
    sound_alert: bool       = False
    popup_text:  str        = ""


class AlertManager:
    def __init__(
        self,
        mode: ProtectionMode = ProtectionMode.NORMAL,
        on_intrusion: Optional[Callable] = None,
        on_stranger_entry: Optional[Callable] = None,   # FIX: new callback
    ):
        self.mode               = mode
        self.on_intrusion       = on_intrusion
        self.on_stranger_entry  = on_stranger_entry      # FIX
        self._last_shield:      ShieldType = ShieldType.NONE
        self._prev_stranger_count: int = 0               # FIX: track transitions

    def process(
        self, assessment: RiskAssessment, frame: np.ndarray
    ) -> tuple[AlertState, np.ndarray]:
        shield = self._determine_shield(assessment)
        state  = self._build_state(assessment, shield)
        out    = self._apply_shield(frame, shield, assessment)

        # ── FIX: fire on_stranger_entry when stranger first appears ───────
        if (assessment.stranger_count > 0
                and self._prev_stranger_count == 0
                and self.on_stranger_entry):
            self.on_stranger_entry(assessment, frame)

        if assessment.is_intrusion and self.on_intrusion:
            self.on_intrusion(assessment, frame)

        self._prev_stranger_count = assessment.stranger_count
        self._last_shield = shield
        return state, out

    def set_mode(self, mode: ProtectionMode):
        self.mode = mode

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _determine_shield(self, a: RiskAssessment) -> ShieldType:
        """
        FIX: Shield escalation table

        SILENT mode  → no visual shield ever
        FOCUS  mode  → blur at WATCH, privacy at DANGER
        NORMAL mode  → warning at WATCH, blur at low DANGER, privacy at DANGER

        Old bug: NORMAL DANGER → privacy only if score >= 85.
        Fix:     NORMAL DANGER → always privacy (score >= 71 is already DANGER).
        """
        if self.mode == ProtectionMode.SILENT:
            return ShieldType.NONE

        code = a.level_code

        if self.mode == ProtectionMode.FOCUS:
            if code == 0: return ShieldType.NONE
            if code == 1: return ShieldType.BLUR
            return ShieldType.PRIVACY          # any DANGER → PRIVACY

        # NORMAL mode
        if code == 0: return ShieldType.NONE
        if code == 1: return ShieldType.WARNING
        # FIX: DANGER always → PRIVACY (removed the score >= 85 gate)
        return ShieldType.PRIVACY

    def _build_state(self, a: RiskAssessment, shield: ShieldType) -> AlertState:
        n = a.stranger_count
        messages = {
            ShieldType.NONE:    a.status_line,
            ShieldType.WARNING: f"⚠ Observer detected — {n} unknown face(s) in view",
            ShieldType.BLUR:    "🔒 High risk — screen partially shielded",
            ShieldType.PRIVACY: "🚨 CRITICAL — screen blocked for your protection",
        }
        return AlertState(
            active=shield != ShieldType.NONE,
            shield=shield,
            message=messages[shield],
            color_hex=a.color,
            sound_alert=(shield == ShieldType.PRIVACY),
            popup_text=messages[shield],
        )

    def _apply_shield(
        self, frame: np.ndarray, shield: ShieldType, a: RiskAssessment
    ) -> np.ndarray:
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        out = frame.copy()

        if shield == ShieldType.NONE:
            return out

        if shield == ShieldType.WARNING:
            return self._overlay_banner(out, a)

        if shield == ShieldType.BLUR:
            blurred = cv2.GaussianBlur(out, (45, 45), 0)
            return self._overlay_banner(blurred, a, title="⚠  PRIVACY SHIELD ACTIVE")

        if shield == ShieldType.PRIVACY:
            h, w = out.shape[:2]
            # Dark overlay — keep faces visible but screen unreadable
            cv2.addWeighted(np.zeros_like(out), 0.85, out, 0.15, 0, out)
            cx = w // 2
            n  = a.stranger_count
            cv2.putText(out, "PRIVACY SHIELD",
                        (cx - 160, h // 2 - 40), cv2.FONT_HERSHEY_DUPLEX,
                        1.4, (0, 60, 255), 2, cv2.LINE_AA)
            cv2.putText(out, "Screen content hidden",
                        (cx - 130, h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (180, 180, 255), 1, cv2.LINE_AA)
            cv2.putText(
                out,
                f"Risk: {a.smooth_score:.0f}/100  |  {n} observer(s)",
                (cx - 180, h // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (120, 120, 200), 1, cv2.LINE_AA,
            )
            return out

        return out

    @staticmethod
    def _overlay_banner(
        frame: np.ndarray, a: RiskAssessment, title: str = ""
    ) -> np.ndarray:
        h, w  = frame.shape[:2]
        bar_h = 38
        colours = {
            "SAFE":   (0, 200, 80),
            "WATCH":  (0, 160, 255),
            "DANGER": (0, 30, 255),
        }
        colour  = colours.get(a.level, (128, 128, 128))

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), colour, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        n    = a.stranger_count
        text = title or (
            f"SCPO Lite  |  {a.level}  |  Risk {a.smooth_score:.0f}/100"
            f"  |  {n} observer(s)"
        )
        cv2.putText(frame, text, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

        badge = f"{a.smooth_score:.0f}"
        cv2.putText(frame, badge, (w - 55, h - 12),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, colour, 2, cv2.LINE_AA)
        return frame
