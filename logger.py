"""
logger.py — SCPO Lite Database & Event Logging Module
=====================================================
Persists privacy events to a local SQLite database.
Provides query helpers for the Streamlit analytics dashboard.

Field name used throughout: stranger_count  (non-admin faces detected)
"""

from __future__ import annotations
import sqlite3, base64, os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import cv2

DB_PATH = os.path.join(os.path.dirname(__file__), "scpo_lite.db")


class PrivacyLogger:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    # ── Init ───────────────────────────────────────────────────────────────
    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT    NOT NULL,
                    level           TEXT    NOT NULL,
                    level_code      INTEGER NOT NULL,
                    risk_score      REAL    NOT NULL,
                    face_count      INTEGER NOT NULL,
                    stranger_count  INTEGER NOT NULL DEFAULT 0,
                    admin_present   INTEGER NOT NULL DEFAULT 0,
                    proximity       REAL    NOT NULL DEFAULT 0,
                    is_intrusion    INTEGER NOT NULL DEFAULT 0,
                    protection_mode TEXT    NOT NULL DEFAULT 'Normal',
                    session_id      TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS screenshots (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id    INTEGER NOT NULL REFERENCES events(id),
                    timestamp   TEXT    NOT NULL,
                    image_b64   TEXT    NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_ts ON events (timestamp)"
            )
            # Migrate old DBs that used observer_count column name
            try:
                conn.execute("ALTER TABLE events ADD COLUMN stranger_count INTEGER NOT NULL DEFAULT 0")
            except Exception:
                pass  # column already exists — fine
            try:
                conn.execute("ALTER TABLE events ADD COLUMN admin_present INTEGER NOT NULL DEFAULT 0")
            except Exception:
                pass

    # ── Write ──────────────────────────────────────────────────────────────
    def log_event(
        self,
        level: str,
        level_code: int,
        risk_score: float,
        face_count: int,
        stranger_count: int = 0,
        admin_present: bool = False,
        proximity: float = 0.0,
        is_intrusion: bool = False,
        protection_mode: str = "Normal",
        session_id: Optional[str] = None,
        frame: Optional[np.ndarray] = None,
    ) -> int:
        ts = datetime.now().isoformat(timespec="seconds")
        with self._conn() as conn:
            cur = conn.execute("""
                INSERT INTO events
                    (timestamp, level, level_code, risk_score, face_count,
                     stranger_count, admin_present, proximity,
                     is_intrusion, protection_mode, session_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (ts, level, level_code, risk_score, face_count,
                  stranger_count, int(admin_present), proximity,
                  int(is_intrusion), protection_mode, session_id))
            event_id = cur.lastrowid

            if frame is not None:
                b64 = self._encode_frame(frame)
                conn.execute(
                    "INSERT INTO screenshots (event_id,timestamp,image_b64) VALUES (?,?,?)",
                    (event_id, ts, b64))
        return event_id

    # ── Read ───────────────────────────────────────────────────────────────
    def get_recent_events(self, limit: int = 50) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query("""
                SELECT id, timestamp, level, risk_score, face_count,
                       stranger_count, admin_present, is_intrusion, protection_mode
                FROM   events ORDER BY id DESC LIMIT ?
            """, conn, params=(limit,))

    def get_risk_history(self, minutes: int = 60) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query("""
                SELECT timestamp, risk_score, level, stranger_count
                FROM   events
                WHERE  timestamp >= datetime('now', ?)
                ORDER  BY timestamp ASC
            """, conn, params=(f"-{minutes} minutes",))

    def get_summary_stats(self) -> dict:
        with self._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)                                        AS total_events,
                    SUM(is_intrusion)                               AS total_intrusions,
                    ROUND(AVG(risk_score), 1)                       AS avg_risk,
                    MAX(risk_score)                                 AS peak_risk,
                    SUM(CASE WHEN level='DANGER' THEN 1 ELSE 0 END) AS danger_count,
                    SUM(CASE WHEN level='WATCH'  THEN 1 ELSE 0 END) AS watch_count,
                    SUM(CASE WHEN level='SAFE'   THEN 1 ELSE 0 END) AS safe_count
                FROM events
            """).fetchone()
        keys = ["total_events","total_intrusions","avg_risk","peak_risk",
                "danger_count","watch_count","safe_count"]
        return dict(zip(keys, row)) if row else {}

    def get_intrusions(self) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query("""
                SELECT e.id, e.timestamp, e.risk_score, e.face_count,
                       e.stranger_count, e.proximity, e.protection_mode,
                       s.image_b64
                FROM   events e
                LEFT   JOIN screenshots s ON s.event_id = e.id
                WHERE  e.is_intrusion = 1
                ORDER  BY e.id DESC LIMIT 20
            """, conn)

    def get_level_distribution(self) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT level, COUNT(*) AS count FROM events GROUP BY level", conn)

    def clear_all(self):
        with self._conn() as conn:
            conn.execute("DELETE FROM screenshots")
            conn.execute("DELETE FROM events")

    # ── Helpers ────────────────────────────────────────────────────────────
    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @staticmethod
    def decode_frame(b64: str) -> np.ndarray:
        buf = base64.b64decode(b64)
        arr = np.frombuffer(buf, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
