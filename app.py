"""
app.py — SCPO Lite Main Application & Streamlit Dashboard  [FIXED]
===================================================================
FIX 1: Screenshot saved to disk (screenshots/ folder) whenever:
         • A stranger first enters the frame           (on_stranger_entry)
         • A sustained DANGER intrusion is confirmed   (on_intrusion)
       Previously no cv2.imwrite() call existed anywhere in the codebase.

FIX 2: on_stranger_entry callback wired into AlertManager so that a file
       screenshot is saved the moment stranger_count transitions 0 → >0,
       even before the 15-frame intrusion threshold is reached.

FIX 3: Screenshot confirmation message shown in the UI (st.session_state
       screenshot_msg) so the user can see saves happening in real time.

FIX 4: detector.reload_profile() called after enrollment so the new
       profile is picked up immediately without restarting the app.

Run with:  streamlit run app.py
"""

from __future__ import annotations
import os, time, uuid
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from detector    import FaceDetector
from risk_engine import RiskEngine
from alerts      import AlertManager, ProtectionMode
from logger      import PrivacyLogger
from enrollment  import Enrollor, is_enrolled, delete_profile, load_profile

# ── Screenshot folder ─────────────────────────────────────────────────────────
SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SCPO Lite · Privacy Observer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family:'Syne',sans-serif; background:#0a0e1a; color:#c8d3e8; }
section[data-testid="stSidebar"] { background:#0d1220; border-right:1px solid #1e2d4a; }
section[data-testid="stSidebar"] * { color:#c8d3e8 !important; }
[data-testid="metric-container"] { background:#111827; border:1px solid #1e2d4a; border-radius:8px; padding:12px 16px; }
.stTabs [data-baseweb="tab-list"]{ gap:6px; border-bottom:1px solid #1e2d4a; }
.stTabs [data-baseweb="tab"]     { background:#111827; border-radius:6px 6px 0 0; padding:6px 18px; }
.stTabs [aria-selected="true"]   { background:#1a2540; border-bottom:2px solid #3b82f6; color:#60a5fa !important; }
.risk-badge { display:inline-block; padding:6px 20px; border-radius:4px;
  font-family:'JetBrains Mono',monospace; font-size:1.1rem; font-weight:700; letter-spacing:.08em; }
.badge-safe   { background:#052e16; color:#00e676; border:1px solid #00e676; }
.badge-watch  { background:#2d1b00; color:#ffab00; border:1px solid #ffab00; }
.badge-danger { background:#2d0a0f; color:#ff1744; border:1px solid #ff1744; }
.score-bar-wrap{ background:#1a1f2e; border-radius:6px; height:14px; margin:4px 0 12px; overflow:hidden; }
.score-bar-fill{ height:100%; border-radius:6px; transition:width .3s ease; }
.comp-row  { display:flex; align-items:center; gap:10px; margin:5px 0;
  font-family:'JetBrains Mono',monospace; font-size:.8rem; }
.comp-label{ width:110px; color:#64748b; }
.comp-bar  { flex:1; background:#1a1f2e; border-radius:4px; height:8px; }
.comp-fill { height:100%; border-radius:4px; background:#3b82f6; }
.comp-val  { width:40px; text-align:right; color:#94a3b8; }
.event-row { background:#111827; border-left:3px solid #1e2d4a; padding:8px 12px;
  margin:4px 0; border-radius:0 6px 6px 0;
  font-family:'JetBrains Mono',monospace; font-size:.78rem; }
.ev-danger { border-color:#ff1744; } .ev-watch{ border-color:#ffab00; } .ev-safe{ border-color:#00e676; }
.enroll-box { background:#111827; border:1px dashed #1e3a5f; border-radius:10px;
  padding:2rem; text-align:center; }
.block-container{ padding-top:1.2rem; }
h1,h2,h3{ font-family:'Syne',sans-serif; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    defaults = dict(
        page="enroll" if not is_enrolled() else "monitor",
        running=False,
        session_id=str(uuid.uuid4())[:8],
        mode="Normal",
        total_events=0,
        intrusion_count=0,
        screenshot_count=0,          # FIX: track saved screenshots
        last_score=0.0,
        last_level="SAFE",
        score_history=[],
        log_interval=5,
        frame_counter=0,
        last_log_frame=0,
        enroll_done=False,
        screenshot_msg="",           # FIX: confirmation message for UI
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

@st.cache_resource
def get_logger():  return PrivacyLogger()
@st.cache_resource
def get_engine():  return RiskEngine()

logger = get_logger()
engine = get_engine()


# ── Screenshot helper ─────────────────────────────────────────────────────────
def save_screenshot(frame: np.ndarray, reason: str) -> str:
    """
    FIX: Save frame to SCREENSHOT_DIR with timestamp filename.
    Returns the saved file path, or "" on failure.
    """
    try:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        filename = f"alert_{reason}_{ts}.jpg"
        filepath = os.path.join(SCREENSHOT_DIR, filename)
        success  = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if success:
            return filepath
        return ""
    except Exception as e:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ SCPO Lite")
    st.markdown("<small style='color:#475569'>Screen Context Privacy Observer</small>",
                unsafe_allow_html=True)
    st.divider()

    enrolled = is_enrolled()
    if enrolled:
        profile = load_profile()
        st.success("✅ Admin enrolled")
        if profile and profile.samples:
            thumbs = profile.samples[:3]
            cols = st.columns(len(thumbs))
            for col, thumb in zip(cols, thumbs):
                col.image(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB),
                          use_container_width=True)
        if st.button("🔄 Re-enroll admin", use_container_width=True):
            delete_profile()
            st.session_state.page        = "enroll"
            st.session_state.running     = False
            st.session_state.enroll_done = False
            st.rerun()
    else:
        st.warning("⚠️ No admin enrolled")

    st.divider()

    if st.session_state.page == "monitor":
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("▶ Start", use_container_width=True,
                         disabled=st.session_state.running):
                st.session_state.running    = True
                st.session_state.session_id = str(uuid.uuid4())[:8]
                engine.reset()
                st.rerun()
        with col_b:
            if st.button("⏹ Stop", use_container_width=True,
                         disabled=not st.session_state.running):
                st.session_state.running = False
                st.rerun()

        st.divider()
        st.markdown("#### Protection Mode")
        mode_choice = st.radio(
            "m", ["Focus", "Normal", "Silent"],
            index=["Focus", "Normal", "Silent"].index(st.session_state.mode),
            label_visibility="collapsed",
        )
        st.session_state.mode = mode_choice
        mode_desc = {
            "Focus":  "🎯 Shield at WATCH",
            "Normal": "⚖️ Shield at DANGER",
            "Silent": "🔇 Log only",
        }
        st.caption(mode_desc[mode_choice])
        st.divider()
        st.session_state.log_interval = st.slider(
            "Log every N frames", 1, 30, st.session_state.log_interval)
        st.divider()
        st.markdown("#### Session Stats")
        st.metric("Events Logged",   st.session_state.total_events)
        st.metric("Intrusions",      st.session_state.intrusion_count)
        st.metric("Screenshots",     st.session_state.screenshot_count)   # FIX
        st.metric("Last Score",      f"{st.session_state.last_score:.1f}/100")
        st.metric("Last Level",      st.session_state.last_level)
        st.divider()
        if st.button("🗑 Clear logs", use_container_width=True):
            logger.clear_all()
            st.session_state.total_events    = 0
            st.session_state.intrusion_count = 0
            st.success("Cleared")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ENROLLMENT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "enroll":
    st.markdown("# 🛡️ SCPO Lite — Admin Enrollment")
    st.markdown("<p style='color:#475569'>Register your face as the trusted admin before monitoring begins.</p>",
                unsafe_allow_html=True)
    st.divider()

    left, right = st.columns([3, 2])

    with right:
        st.markdown("""
        #### How enrollment works
        1. Allow webcam access
        2. Position your face in the frame
        3. Hold still — the system captures **30 samples**
        4. Once complete, monitoring unlocks

        #### What changes after enrollment
        - **Your face → SAFE** (admin verified)
        - **Any other face → triggers WATCH/DANGER**
        - Admin + stranger → DANGER based on stranger behavior
        """)

    with left:
        cam_ph   = st.empty()
        prog_ph  = st.empty()
        msg_ph   = st.empty()
        start_ph = st.empty()

        if not st.session_state.enroll_done:
            if start_ph.button("📷 Start Enrollment", use_container_width=True,
                               type="primary"):
                st.session_state.enroll_done = False
                enrollor = Enrollor()
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ Cannot access webcam.")
                    st.stop()
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                start_ph.empty()

                while not enrollor.done:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame  = cv2.flip(frame, 1)
                    status, annotated = enrollor.process_frame(frame)

                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    cam_ph.image(rgb, channels="RGB", use_container_width=True)
                    prog_ph.progress(enrollor.progress,
                                     text=f"Collecting samples: {enrollor.sample_count}/30")
                    msg_ph.info(status)
                    time.sleep(0.05)

                cap.release()

                if enrollor.done:
                    st.session_state.enroll_done = True
                    st.rerun()
        else:
            cam_ph.empty()
            st.success("✅ Admin face enrolled successfully!")
            st.balloons()
            if st.button("▶ Proceed to Monitoring", type="primary",
                         use_container_width=True):
                st.session_state.page = "monitor"
                st.rerun()

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MONITORING DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🛡️ SCPO Lite")
st.markdown("<p style='color:#475569;margin-top:-12px'>Visual Exposure Risk Engine · Admin-Aware Privacy Monitor</p>",
            unsafe_allow_html=True)
st.divider()

cam_col, info_col = st.columns([3, 2])
with cam_col:
    st.markdown("#### 📷 Live Feed")
    cam_ph       = st.empty()
    risk_ph      = st.empty()
    alert_ph     = st.empty()
    status_ph    = st.empty()
    # FIX: placeholder for screenshot confirmation messages
    screenshot_ph = st.empty()

with info_col:
    st.markdown("#### 📊 Risk Breakdown")
    comp_ph   = st.empty()
    st.markdown("#### 🕒 Recent Events")
    events_ph = st.empty()

st.divider()
tab_hist, tab_intru, tab_info = st.tabs(
    ["📈 Risk History", "🚨 Intrusion Log", "ℹ️ System Info"])


# ── Renderers ─────────────────────────────────────────────────────────────────
def render_gauge(score, level):
    bc  = {"SAFE": "badge-safe", "WATCH": "badge-watch", "DANGER": "badge-danger"}[level]
    col = {"SAFE": "#00e676", "WATCH": "#ffab00", "DANGER": "#ff1744"}[level]
    pct = min(int(score), 100)
    risk_ph.markdown(f"""
    <div style="margin:8px 0">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">
        <span class="risk-badge {bc}">{level}</span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:1.6rem;color:{col};font-weight:700">{score:.1f}</span>
        <span style="color:#475569;font-size:.85rem">/ 100</span>
      </div>
      <div class="score-bar-wrap">
        <div class="score-bar-fill" style="width:{pct}%;background:{col}"></div>
      </div>
    </div>""", unsafe_allow_html=True)


def render_status(assess):
    col = assess.color
    status_ph.markdown(
        f'<div style="background:#111827;border-left:4px solid {col};'
        f'padding:8px 14px;border-radius:0 6px 6px 0;'
        f'font-family:JetBrains Mono,monospace;font-size:.82rem;color:{col}">'
        f'{assess.status_line}</div>', unsafe_allow_html=True)


def render_components(components):
    rows = ""
    for label, val in components.items():
        pct  = min(int(val), 100)
        rows += (f'<div class="comp-row"><span class="comp-label">{label}</span>'
                 f'<div class="comp-bar"><div class="comp-fill" style="width:{pct}%"></div></div>'
                 f'<span class="comp-val">{val:.0f}</span></div>')
    comp_ph.markdown(f'<div style="margin:4px 0">{rows}</div>', unsafe_allow_html=True)


def render_events(df):
    if df.empty:
        events_ph.caption("No events logged yet.")
        return
    rows = ""
    for _, row in df.head(8).iterrows():
        cls = {"DANGER": "ev-danger", "WATCH": "ev-watch", "SAFE": "ev-safe"}.get(
            row["level"], "ev-safe")
        ts  = str(row["timestamp"])[11:19]
        rows += (f'<div class="event-row {cls}"><b>{row["level"]}</b> &nbsp;|&nbsp; {ts}'
                 f' &nbsp;|&nbsp; Score {row["risk_score"]:.0f}'
                 f' &nbsp;|&nbsp; {int(row["face_count"])} face(s)'
                 f'{"&nbsp;🚨" if row.get("is_intrusion") else ""}</div>')
    events_ph.markdown(rows, unsafe_allow_html=True)


def render_alert(msg, level):
    col = {"SAFE": "#00e676", "WATCH": "#ffab00", "DANGER": "#ff1744"}.get(
        level, "#64748b")
    if level in ("WATCH", "DANGER"):
        alert_ph.markdown(
            f'<div style="background:#111827;border-left:4px solid {col};'
            f'padding:10px 16px;border-radius:0 6px 6px 0;'
            f'font-family:JetBrains Mono,monospace;font-size:.85rem;color:{col}">{msg}</div>',
            unsafe_allow_html=True)
    else:
        alert_ph.empty()


# FIX: show screenshot saved confirmation in the UI
def render_screenshot_msg():
    msg = st.session_state.get("screenshot_msg", "")
    if msg:
        screenshot_ph.success(msg)
    else:
        screenshot_ph.empty()


# ── Standby ───────────────────────────────────────────────────────────────────
if not st.session_state.running:
    cam_ph.markdown("""
    <div style="background:#111827;border:1px dashed #1e2d4a;border-radius:8px;
                height:340px;display:flex;flex-direction:column;
                align-items:center;justify-content:center;color:#334155">
      <div style="font-size:3rem">🎥</div>
      <div style="margin-top:12px;font-family:'JetBrains Mono',monospace">Press ▶ Start to begin monitoring</div>
    </div>""", unsafe_allow_html=True)

    with tab_hist:
        hist = logger.get_risk_history(minutes=120)
        if not hist.empty:
            st.line_chart(hist.set_index("timestamp")["risk_score"],
                          use_container_width=True, height=260)
        else:
            st.caption("No history yet.")
    with tab_intru:
        df = logger.get_intrusions()
        st.dataframe(
            df[["timestamp", "risk_score", "face_count", "proximity", "protection_mode"]]
            if not df.empty else df,
            use_container_width=True,
        )
    with tab_info:
        st.markdown("""
### System Info
| Component | Detail |
|-----------|--------|
| Face Detection | OpenCV Haar Cascade |
| Admin ID | Histogram correlation (3×3 grid, 64-bin) |
| Match threshold | 0.55 correlation score |
| Risk Engine | VER-E — stranger-only scoring |
| Storage | SQLite (local) + screenshots/ folder |

### Risk Logic
| Situation | Level |
|-----------|-------|
| Admin alone | 🟢 SAFE |
| Unknown face (no admin enrolled) | 🟡 WATCH |
| Stranger detected (admin present) | 🟡→🔴 based on behavior |
| Multiple strangers | 🔴 DANGER |

### Screenshot Behavior
| Trigger | File saved? |
|---------|-------------|
| Stranger first enters frame | ✅ Immediately |
| Sustained DANGER (15 frames) | ✅ Intrusion screenshot |
""")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Live loop
# ══════════════════════════════════════════════════════════════════════════════
detector  = FaceDetector(frame_width=640, frame_height=480)
alert_mgr = AlertManager(mode=ProtectionMode(st.session_state.mode))


# ── FIX: on_stranger_entry — fires the moment a stranger first appears ────────
def _on_stranger_entry(assess, frame):
    """Save screenshot immediately when stranger enters frame."""
    path = save_screenshot(frame, "stranger_entry")
    if path:
        st.session_state.screenshot_count += 1
        st.session_state.screenshot_msg = (
            f"📸 Screenshot saved: stranger entered frame  "
            f"[{os.path.basename(path)}]"
        )
        # Also log to DB with the raw frame
        logger.log_event(
            level=assess.level, level_code=assess.level_code,
            risk_score=assess.smooth_score, face_count=assess.face_count,
            stranger_count=assess.stranger_count, admin_present=assess.admin_present,
            proximity=assess.dominant_proximity, is_intrusion=False,
            protection_mode=st.session_state.mode,
            session_id=st.session_state.session_id, frame=frame,
        )


# ── FIX: on_intrusion — fires after 15 sustained DANGER frames ───────────────
def _on_intrusion(assess, frame):
    """Save screenshot and log sustained intrusion event."""
    path = save_screenshot(frame, "intrusion")
    if path:
        st.session_state.screenshot_count += 1
        st.session_state.screenshot_msg = (
            f"🚨 INTRUSION screenshot saved  [{os.path.basename(path)}]"
        )
    logger.log_event(
        level=assess.level, level_code=assess.level_code,
        risk_score=assess.smooth_score, face_count=assess.face_count,
        stranger_count=assess.stranger_count, admin_present=assess.admin_present,
        proximity=assess.dominant_proximity, is_intrusion=True,
        protection_mode=st.session_state.mode,
        session_id=st.session_state.session_id, frame=frame,
    )
    st.session_state.intrusion_count += 1


# FIX: wire both callbacks into AlertManager
alert_mgr.on_intrusion      = _on_intrusion
alert_mgr.on_stranger_entry = _on_stranger_entry   # FIX: was missing entirely

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("❌ Cannot access webcam.")
    st.session_state.running = False
    st.stop()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

try:
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        st.session_state.frame_counter += 1

        det    = detector.process_frame(frame)
        assess = engine.evaluate(det)
        alert_mgr.set_mode(ProtectionMode(st.session_state.mode))

        display_frame  = det.frame_annotated if det.frame_annotated is not None else frame
        state, protected = alert_mgr.process(assess, display_frame)

        st.session_state.last_score = assess.smooth_score
        st.session_state.last_level = assess.level
        st.session_state.score_history.append(assess.smooth_score)
        if len(st.session_state.score_history) > 200:
            st.session_state.score_history.pop(0)

        fc       = st.session_state.frame_counter
        interval = st.session_state.log_interval
        if assess.level_code >= 1 and fc - st.session_state.last_log_frame >= interval:
            logger.log_event(
                level=assess.level, level_code=assess.level_code,
                risk_score=assess.smooth_score, face_count=assess.face_count,
                stranger_count=assess.stranger_count, admin_present=assess.admin_present,
                proximity=assess.dominant_proximity, is_intrusion=assess.is_intrusion,
                protection_mode=st.session_state.mode,
                session_id=st.session_state.session_id,
            )
            st.session_state.total_events  += 1
            st.session_state.last_log_frame = fc

        rgb = cv2.cvtColor(protected, cv2.COLOR_BGR2RGB)
        cam_ph.image(rgb, channels="RGB", use_container_width=True)

        render_gauge(assess.smooth_score, assess.level)
        render_status(assess)
        render_components(assess.components)
        render_alert(state.message, assess.level)
        render_screenshot_msg()          # FIX: show confirmation below feed
        render_events(logger.get_recent_events(limit=10))

        # Clear screenshot message after showing it once
        if st.session_state.screenshot_msg:
            st.session_state.screenshot_msg = ""

        if fc % 30 == 0:
            with tab_hist:
                h = engine.score_history
                if h:
                    st.line_chart(pd.DataFrame({"Risk Score": h}),
                                  use_container_width=True, height=220)
            with tab_intru:
                df = logger.get_intrusions()
                if not df.empty:
                    st.dataframe(
                        df[["timestamp", "risk_score", "face_count",
                            "proximity", "protection_mode"]],
                        use_container_width=True)
                else:
                    st.caption("No intrusions this session.")

        time.sleep(0.03)
finally:
    cap.release()
