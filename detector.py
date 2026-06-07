"""
detector.py — SCPO Lite Face Detection & Tracking Module  [FIXED]
=================================================================
FIX 1: Tracking now uses IoU-matched track records (dict keyed by track_id)
        so admin identity is NEVER lost when a stranger enters the frame.
        Previously, index-based matching shifted recognition results across
        faces when new faces appeared mid-frame.

FIX 2: Recognition runs on a per-track cache — once a track_id is confirmed
        as ADMIN it stays ADMIN for the life of that track, preventing
        flicker when the face is partially occluded.

FIX 3: Detailed per-frame logging added (face count, similarity score,
        matched identity, distance score) for easier debugging.

Key logic (unchanged):
  • Single face detected + matches admin  → stranger_count = 0
  • Single face detected + no match       → stranger_count = 1
  • Two faces: admin + stranger           → stranger_count = 1
  • Two strangers (no admin)              → stranger_count = 2
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

from enrollment import AdminProfile, load_profile, _extract_hist, MATCH_THRESHOLD

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [DETECTOR] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("detector")

CASCADE_PATH     = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"

SCALE_FACTOR  = 1.15
MIN_NEIGHBORS = 4
MIN_FACE_SIZE = (40, 40)
FRAME_SKIP    = 2

# How many frames a track must be UNSEEN before we drop its identity cache
TRACK_TTL = 10


@dataclass
class FaceRecord:
    bbox:              Tuple[int, int, int, int]
    center:            Tuple[int, int]
    area:              int
    proximity_score:   float
    orientation_score: float
    motion_delta:      float
    dwell_frames:      int
    track_id:          int
    is_admin:          bool  = False
    admin_similarity:  float = 0.0


@dataclass
class DetectionResult:
    faces:           List[FaceRecord]       = field(default_factory=list)
    frame_annotated: Optional[np.ndarray]   = None
    raw_frame:       Optional[np.ndarray]   = None
    face_count:      int                    = 0
    stranger_count:  int                    = 0
    admin_present:   bool                   = False
    max_proximity:   float                  = 0.0
    total_motion:    float                  = 0.0


# ── Track record ──────────────────────────────────────────────────────────────
@dataclass
class _TrackState:
    """Persistent state per track_id across frames."""
    track_id:    int
    dwell:       int   = 1
    is_admin:    bool  = False
    similarity:  float = 0.0
    last_seen:   int   = 0          # frame_counter when last matched
    confirmed:   bool  = False      # True once recognition ran & returned admin


class FaceDetector:
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        self.eye_cascade  = cv2.CascadeClassifier(EYE_CASCADE_PATH)
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade.")

        self.frame_w = frame_width
        self.frame_h = frame_height
        self.admin_profile: Optional[AdminProfile] = load_profile()

        # ── FIX: per-track state dict instead of parallel arrays ──────────
        self._tracks: Dict[int, _TrackState] = {}   # track_id → _TrackState
        self._prev_boxes: List[Tuple]  = []         # boxes from last processed frame
        self._prev_ids:   List[int]    = []         # track_ids matching _prev_boxes
        self._next_id:    int          = 0
        self._frame_counter: int       = 0
        self._last_result: DetectionResult = DetectionResult()

    def reload_profile(self):
        """Call this after re-enrollment to pick up the new profile."""
        self.admin_profile = load_profile()
        # Invalidate all cached admin confirmations so they re-check
        for t in self._tracks.values():
            t.confirmed = False
            t.is_admin  = False
        log.info("Admin profile reloaded — track cache cleared.")

    # ── Main entry ────────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        self._frame_counter += 1

        # On skipped frames re-annotate with cached positions
        if self._frame_counter % FRAME_SKIP != 0:
            annotated = frame.copy()
            for rec in self._last_result.faces:
                annotated = self._draw_face(annotated, rec)
            annotated = self._draw_safe_zone(annotated)
            self._last_result.frame_annotated = annotated
            self._last_result.raw_frame       = frame.copy()
            return self._last_result

        raw  = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        boxes = self.face_cascade.detectMultiScale(
            gray, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE, flags=cv2.CASCADE_SCALE_IMAGE,
        )
        current_boxes = list(boxes) if len(boxes) > 0 else []

        # ── FIX: IoU-based tracking returns stable track_ids per box ──────
        assigned_ids, assigned_dwells = self._update_tracking(current_boxes)

        # Expire stale tracks (not seen for TRACK_TTL frames)
        active_ids = set(assigned_ids)
        stale = [tid for tid, t in self._tracks.items()
                 if self._frame_counter - t.last_seen > TRACK_TTL]
        for tid in stale:
            del self._tracks[tid]

        face_records: List[FaceRecord] = []
        annotated     = frame.copy()
        max_proximity = 0.0
        total_motion  = 0.0
        admin_present = False

        log.debug("── Frame %d  detected %d face(s) ──",
                  self._frame_counter, len(current_boxes))

        for i, (x, y, w, h) in enumerate(current_boxes):
            tid   = assigned_ids[i]
            track = self._tracks[tid]

            motion      = self._compute_motion_for_track(tid, (x, y, w, h))
            proximity   = self._estimate_proximity(w, h)
            orientation = self._estimate_orientation(gray, x, y, w, h)

            # ── FIX: recognition only if not already confirmed as admin ───
            # Once confirmed admin, keep that status until track expires.
            # This prevents stranger appearance from "stealing" the admin label.
            if self.admin_profile is not None and not track.confirmed:
                pad = int(w * 0.15)
                x1 = max(x - pad, 0);      y1 = max(y - pad, 0)
                x2 = min(x + w + pad, frame.shape[1])
                y2 = min(y + h + pad, frame.shape[0])
                face_gray = gray[y1:y2, x1:x2]

                is_adm, sim = self.admin_profile.is_admin(face_gray)
                track.is_admin   = is_adm
                track.similarity = sim

                # ── FIX: mark confirmed only on positive match ─────────────
                # For non-admin faces we keep re-checking every frame so
                # that a mis-classified face can self-correct, but we
                # never flip a confirmed admin to stranger.
                if is_adm:
                    track.confirmed = True

                log.debug(
                    "  Track %d  bbox=(%d,%d,%d,%d)  sim=%.3f  threshold=%.2f"
                    "  → %s",
                    tid, x, y, w, h, sim, MATCH_THRESHOLD,
                    "ADMIN ✓" if is_adm else "STRANGER",
                )
            elif self.admin_profile is None:
                track.is_admin   = False
                track.similarity = 0.0
                log.debug("  Track %d — no admin profile loaded.", tid)
            else:
                log.debug(
                    "  Track %d — confirmed ADMIN (cached sim=%.3f)",
                    tid, track.similarity,
                )

            if track.is_admin:
                admin_present = True

            rec = FaceRecord(
                bbox=(x, y, w, h),
                center=(x + w // 2, y + h // 2),
                area=w * h,
                proximity_score=proximity,
                orientation_score=orientation,
                motion_delta=motion,
                dwell_frames=track.dwell,
                track_id=tid,
                is_admin=track.is_admin,
                admin_similarity=track.similarity,
            )
            face_records.append(rec)
            max_proximity = max(max_proximity, proximity)
            total_motion += motion
            annotated = self._draw_face(annotated, rec)

        annotated = self._draw_safe_zone(annotated)

        stranger_count = sum(1 for f in face_records if not f.is_admin)

        # ── Summary log line ──────────────────────────────────────────────
        log.info(
            "Frame %d | faces=%d | admin=%s | strangers=%d | max_prox=%.2f",
            self._frame_counter, len(face_records),
            admin_present, stranger_count, max_proximity,
        )

        result = DetectionResult(
            faces=face_records,
            frame_annotated=annotated,
            raw_frame=raw,
            face_count=len(face_records),
            stranger_count=stranger_count,
            admin_present=admin_present,
            max_proximity=max_proximity,
            total_motion=total_motion,
        )
        self._last_result = result
        return result

    # ── Tracking ──────────────────────────────────────────────────────────────

    def _update_tracking(
        self, current_boxes: List[Tuple]
    ) -> Tuple[List[int], List[int]]:
        """
        FIX: Returns (track_ids, dwell_counts) aligned with current_boxes.

        Uses IoU matching against previous frame's boxes.  Each current box
        is matched to the best previous box (IoU > 0.30).  Matched → reuse
        existing track_id and increment dwell.  Unmatched → new track_id.
        """
        if not current_boxes:
            self._prev_boxes = []
            self._prev_ids   = []
            return [], []

        assigned_ids:   List[int] = []
        assigned_dwells: List[int] = []

        used_prev: set = set()

        for box in current_boxes:
            best_iou, best_j = 0.0, -1
            for j, prev_box in enumerate(self._prev_boxes):
                if j in used_prev:
                    continue
                iou = self._iou(box, prev_box)
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou > 0.30 and best_j >= 0:
                tid = self._prev_ids[best_j]
                used_prev.add(best_j)
            else:
                tid = self._next_id
                self._next_id += 1

            # Create track state if new
            if tid not in self._tracks:
                self._tracks[tid] = _TrackState(track_id=tid)

            self._tracks[tid].dwell     += 1
            self._tracks[tid].last_seen  = self._frame_counter

            assigned_ids.append(tid)
            assigned_dwells.append(self._tracks[tid].dwell)

        # Update prev state for next frame
        self._prev_boxes = list(current_boxes)
        self._prev_ids   = list(assigned_ids)

        return assigned_ids, assigned_dwells

    def _compute_motion_for_track(self, tid: int, box: Tuple) -> float:
        """FIX: motion computed per-track, not by array index."""
        x, y, w, h = box
        cx, cy = x + w // 2, y + h // 2
        if tid not in self._tracks:
            return 0.0
        # Look for this track_id in prev list
        for j, prev_id in enumerate(self._prev_ids):
            if prev_id == tid and j < len(self._prev_boxes):
                px, py, pw, ph = self._prev_boxes[j]
                return float(np.hypot(cx - (px + pw // 2), cy - (py + ph // 2)))
        return 0.0

    # ── Proximity / Orientation ───────────────────────────────────────────────

    def _estimate_proximity(self, w, h) -> float:
        return min((h / self.frame_h) / 0.40, 1.0)

    def _estimate_orientation(self, gray, x, y, w, h) -> float:
        roi  = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(
            roi, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
        n = len(eyes) if len(eyes) > 0 else 0
        return {0: 0.2, 1: 0.6}.get(n, 1.0)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_face(self, frame, rec: FaceRecord) -> np.ndarray:
        x, y, w, h = rec.bbox
        if rec.is_admin:
            colour = (0, 230, 100)
            label  = f"ADMIN  sim:{rec.admin_similarity:.2f}"
        else:
            p      = rec.proximity_score
            colour = (30, int(255 * (1 - p)), int(255 * p))
            label  = f"STRANGER  ID{rec.track_id}  P:{p:.2f}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        corner = 12
        for cx2, cy2, dx, dy in [
            (x, y, 1, 1), (x+w, y, -1, 1),
            (x, y+h, 1, -1), (x+w, y+h, -1, -1),
        ]:
            cv2.line(frame, (cx2, cy2), (cx2 + dx * corner, cy2), colour, 3)
            cv2.line(frame, (cx2, cy2), (cx2, cy2 + dy * corner), colour, 3)
        cv2.putText(frame, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
        return frame

    def _draw_safe_zone(self, frame) -> np.ndarray:
        zx  = int(self.frame_w * 0.20);  zy  = int(self.frame_h * 0.15)
        zx2 = int(self.frame_w * 0.80);  zy2 = int(self.frame_h * 0.85)
        overlay = frame.copy()
        cv2.rectangle(overlay, (zx, zy), (zx2, zy2), (0, 220, 120), 1)
        cv2.putText(overlay, "SAFE ZONE", (zx + 4, zy + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 120), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        return frame

    @staticmethod
    def _iou(a, b) -> float:
        ax, ay, aw, ah = a;  bx, by, bw, bh = b
        ix  = max(ax, bx);   iy  = max(ay, by)
        ix2 = min(ax+aw, bx+bw);  iy2 = min(ay+ah, by+bh)
        if ix2 <= ix or iy2 <= iy:
            return 0.0
        inter = (ix2 - ix) * (iy2 - iy)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0
