"""
app.py — Streamlit GUI for Hand Gesture Recognition System
Hand Gesture Recognition System for Sign Language
Author: Anjali Negi & Divyansh Agrawal | BCA Final Year Project

Run:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import time
import threading
from collections import deque
from datetime import datetime
from test import predict_gesture

# ── Optional TTS ──────────────────────────────────────────────────────────────
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
#  Custom CSS — dark-mode "neural terminal" aesthetic
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── Root tokens ──────────────────────────────────────────────────────────── */
:root {
    --bg-deep:    #080d12;
    --bg-card:    #0f1923;
    --bg-card2:   #141e2b;
    --accent:     #00e5a0;
    --accent2:    #0af;
    --accent3:    #f72585;
    --text-main:  #e8f4f0;
    --text-muted: #6b8a7a;
    --border:     rgba(0,229,160,.18);
    --glow:       0 0 24px rgba(0,229,160,.35);
    --glow2:      0 0 40px rgba(0,170,255,.25);
    --radius:     14px;
    --font-ui:    'Syne', sans-serif;
    --font-mono:  'Space Mono', monospace;
}

/* ── Global reset ──────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    background-color: var(--bg-deep) !important;
    color: var(--text-main) !important;
    font-family: var(--font-ui) !important;
}

/* ── Hide Streamlit chrome ─────────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Hero banner ───────────────────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #040a10 0%, #0a1628 50%, #040d18 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2.2rem 2.8rem 1.8rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--glow2);
}
.hero-banner::before {
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 60% 50% at 15% 50%, rgba(0,229,160,.07) 0%, transparent 70%),
        radial-gradient(ellipse 40% 40% at 85% 30%, rgba(0,170,255,.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,229,160,.12);
    border: 1px solid rgba(0,229,160,.35);
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: .7rem;
    letter-spacing: .12em;
    padding: .25rem .75rem;
    border-radius: 20px;
    margin-bottom: .9rem;
    text-transform: uppercase;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1.15;
    margin: 0 0 .5rem;
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 60%, #fff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-family: var(--font-mono);
    font-size: .82rem;
    color: var(--text-muted);
    letter-spacing: .04em;
}

/* ── Cards ─────────────────────────────────────────────────────────────────── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: var(--font-mono);
    font-size: .72rem;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: .9rem;
    display: flex; align-items: center; gap: .5rem;
}
.card-title::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
}

/* ── Prediction display ────────────────────────────────────────────────────── */
.pred-wrap {
    background: linear-gradient(135deg, var(--bg-card) 0%, #0a1f18 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem 1.6rem 1.6rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: var(--glow);
}
.pred-wrap::after {
    content: '';
    position: absolute; bottom: 0; left: 10%; right: 10%; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
}
.pred-label {
    font-size: 3.2rem;
    font-weight: 800;
    color: var(--accent);
    text-shadow: var(--glow);
    letter-spacing: -.01em;
    line-height: 1.1;
    margin: .3rem 0;
}
.pred-label.no-hand { color: var(--text-muted); font-size: 1.4rem; }
.pred-tag {
    font-family: var(--font-mono);
    font-size: .72rem;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--text-muted);
}

/* ── Confidence bar ────────────────────────────────────────────────────────── */
.conf-wrap { margin-top: 1rem; }
.conf-label {
    font-family: var(--font-mono);
    font-size: .72rem;
    color: var(--text-muted);
    letter-spacing: .1em;
    text-transform: uppercase;
    margin-bottom: .35rem;
    display: flex; justify-content: space-between;
}
.conf-track {
    background: rgba(255,255,255,.06);
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 99px;
    transition: width .4s ease;
}
.conf-high   { background: linear-gradient(90deg, #00e5a0, #0af); box-shadow: 0 0 10px rgba(0,229,160,.5); }
.conf-medium { background: linear-gradient(90deg, #ffd700, #f4a020); }
.conf-low    { background: linear-gradient(90deg, #f72585, #f4542a); }

/* ── History log ───────────────────────────────────────────────────────────── */
.hist-item {
    display: flex;
    align-items: center;
    gap: .9rem;
    padding: .55rem .8rem;
    border-radius: 8px;
    margin-bottom: .4rem;
    background: rgba(255,255,255,.03);
    border-left: 3px solid var(--accent);
    transition: background .2s;
}
.hist-item:hover { background: rgba(0,229,160,.07); }
.hist-gesture { font-weight: 700; font-size: .95rem; color: var(--text-main); flex: 1; }
.hist-conf {
    font-family: var(--font-mono);
    font-size: .72rem;
    color: var(--accent);
    background: rgba(0,229,160,.1);
    padding: .15rem .5rem;
    border-radius: 99px;
}
.hist-time { font-family: var(--font-mono); font-size: .65rem; color: var(--text-muted); }

/* ── Status badge ──────────────────────────────────────────────────────────── */
.status-dot {
    display: inline-block;
    width: 9px; height: 9px;
    border-radius: 50%;
    margin-right: .45rem;
    animation: pulse 1.6s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: .55; transform: scale(1.25); }
}
.status-live  { background: #00e5a0; box-shadow: 0 0 8px #00e5a0; }
.status-idle  { background: #6b8a7a; animation: none; }
.status-error { background: #f72585; box-shadow: 0 0 8px #f72585; }

/* ── Info tiles ────────────────────────────────────────────────────────────── */
.tile-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .65rem; margin-top: .3rem; }
.tile {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 10px;
    padding: .75rem 1rem;
    text-align: center;
}
.tile-val { font-size: 1.6rem; font-weight: 800; color: var(--accent2); line-height: 1; }
.tile-key { font-family: var(--font-mono); font-size: .65rem; color: var(--text-muted); margin-top: .25rem; letter-spacing: .08em; }

/* ── Sidebar ───────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-card2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] label { color: var(--text-main) !important; }

/* ── Buttons ───────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #00c882 100%) !important;
    color: #080d12 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--font-ui) !important;
    font-weight: 700 !important;
    font-size: .92rem !important;
    letter-spacing: .04em !important;
    padding: .65rem 1.8rem !important;
    box-shadow: 0 4px 20px rgba(0,229,160,.35) !important;
    transition: all .2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(0,229,160,.5) !important;
}
.stop-btn > button {
    background: linear-gradient(135deg, #f72585 0%, #c0175f 100%) !important;
    box-shadow: 0 4px 20px rgba(247,37,133,.35) !important;
}

/* ── Divider ───────────────────────────────────────────────────────────────── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1.2rem 0;
}

/* ── Error box ─────────────────────────────────────────────────────────────── */
.err-box {
    background: rgba(247,37,133,.08);
    border: 1px solid rgba(247,37,133,.3);
    border-radius: 10px;
    padding: .75rem 1rem;
    font-family: var(--font-mono);
    font-size: .78rem;
    color: #f72585;
}

/* ── Video frame ───────────────────────────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--glow2) !important;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Session-state initialisation
# ═════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = dict(
        camera_running  = False,
        history         = deque(maxlen=5),
        total_detections= 0,
        session_start   = time.time(),
        voice_enabled   = False,
        last_spoken     = "",
        last_speak_time = 0.0,
        frame_count     = 0,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ═════════════════════════════════════════════════════════════════════════════
#  TTS helper
# ═════════════════════════════════════════════════════════════════════════════
_tts_lock   = threading.Lock()
_tts_engine = None


def _speak(text: str):
    """Fire-and-forget TTS in a background thread."""
    if not TTS_AVAILABLE or not st.session_state.voice_enabled:
        return
    # Debounce: don't repeat same word within 4 s
    now = time.time()
    if (text == st.session_state.last_spoken and
            now - st.session_state.last_speak_time < 4.0):
        return
    st.session_state.last_spoken    = text
    st.session_state.last_speak_time = now

    def _run():
        global _tts_engine
        with _tts_lock:
            try:
                if _tts_engine is None:
                    _tts_engine = pyttsx3.init()
                    _tts_engine.setProperty("rate", 165)
                _tts_engine.say(text)
                _tts_engine.runAndWait()
            except Exception:
                pass
    threading.Thread(target=_run, daemon=True).start()


# ═════════════════════════════════════════════════════════════════════════════
#  Confidence helpers
# ═════════════════════════════════════════════════════════════════════════════
def _conf_class(conf: float) -> str:
    if conf >= 0.75: return "conf-high"
    if conf >= 0.45: return "conf-medium"
    return "conf-low"

def _conf_emoji(conf: float) -> str:
    if conf >= 0.75: return "🟢"
    if conf >= 0.45: return "🟡"
    return "🔴"


# ═════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 .5rem;">
        <div style="font-size:2.4rem;">🤟</div>
        <div style="font-family:'Space Mono',monospace; font-size:.72rem;
                    letter-spacing:.15em; color:#00e5a0; text-transform:uppercase;
                    margin-top:.4rem;">
            SignSense AI
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Camera controls ──
    st.markdown('<div class="card-title">📷  Camera Control</div>', unsafe_allow_html=True)

    if not st.session_state.camera_running:
        if st.button("▶  Start Camera", key="btn_start"):
            st.session_state.camera_running = True
            st.session_state.session_start  = time.time()
            st.rerun()
    else:
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        if st.button("⏹  Stop Camera", key="btn_stop"):
            st.session_state.camera_running = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Status
    if st.session_state.camera_running:
        st.markdown('<span class="status-dot status-live"></span><span style="font-size:.82rem;">Live</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-dot status-idle"></span><span style="font-size:.82rem; color:#6b8a7a;">Idle</span>',
                    unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Settings ──
    st.markdown('<div class="card-title">⚙️  Settings</div>', unsafe_allow_html=True)

    conf_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, 0.50, 0.05,
        help="Predictions below this value are suppressed."
    )

    st.markdown("**Voice output**")
    if TTS_AVAILABLE:
        st.session_state.voice_enabled = st.toggle(
            "Speak predictions aloud", value=st.session_state.voice_enabled
        )
    else:
        st.caption("⚠️ pyttsx3 not installed — voice disabled.")

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Gesture legend ──
    st.markdown('<div class="card-title">🗂  Gesture Labels</div>', unsafe_allow_html=True)
    labels = ["Hello", "I Love You", "No", "Okay", "Please", "Thank You", "Yes"]
    emojis = ["👋", "🤟", "🙅", "👌", "🙏", "🤲", "✌️"]
    for i, (label, emoji) in enumerate(zip(labels, emojis)):
        st.markdown(
            f'<div style="display:flex;gap:.6rem;align-items:center;'
            f'padding:.3rem .5rem; border-radius:6px; margin-bottom:.2rem; '
            f'background:rgba(255,255,255,.03);">'
            f'<span>{emoji}</span>'
            f'<span style="font-size:.85rem;">{label}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Clear history ──
    if st.button("🗑  Clear History", key="btn_clear"):
        st.session_state.history.clear()
        st.session_state.total_detections = 0
        st.rerun()

    st.markdown(
        '<div style="font-family:\'Space Mono\',monospace; font-size:.6rem; '
        'color:#3a5a4a; text-align:center; margin-top:1.5rem;">'
        'BCA Final Year Project<br>IINTM · 2023–26</div>',
        unsafe_allow_html=True
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Hero header
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🤟 AI · Computer Vision · Sign Language</div>
    <div class="hero-title">Hand Gesture Recognition</div>
    <div class="hero-sub">Real-time sign language interpretation using MediaPipe + Keras · OpenCV pipeline</div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Main layout — two columns
# ═════════════════════════════════════════════════════════════════════════════
col_video, col_panel = st.columns([3, 2], gap="large")

# ── Column 1 : Video feed ──────────────────────────────────────────────────
with col_video:
    st.markdown('<div class="card-title">📹  Live Camera Feed</div>', unsafe_allow_html=True)
    video_placeholder = st.empty()
    err_placeholder   = st.empty()

    # Static placeholder when idle
    if not st.session_state.camera_running:
        video_placeholder.markdown("""
        <div style="
            background:rgba(255,255,255,.03);
            border:1px dashed rgba(0,229,160,.25);
            border-radius:12px;
            height:360px;
            display:flex; flex-direction:column;
            align-items:center; justify-content:center;
            gap:.8rem; color:#3a5a4a;">
            <div style="font-size:3rem;">📷</div>
            <div style="font-family:'Space Mono',monospace; font-size:.78rem; letter-spacing:.1em;">
                CAMERA OFFLINE
            </div>
            <div style="font-size:.8rem;">Press ▶ Start Camera in the sidebar</div>
        </div>
        """, unsafe_allow_html=True)

# ── Column 2 : Prediction panel ───────────────────────────────────────────
with col_panel:
    pred_placeholder  = st.empty()
    conf_placeholder  = st.empty()
    stats_placeholder = st.empty()
    hist_placeholder  = st.empty()

    def _render_idle_panel():
        pred_placeholder.markdown("""
        <div class="pred-wrap">
            <div class="pred-tag">GESTURE PREDICTION</div>
            <div class="pred-label no-hand">Awaiting feed…</div>
        </div>
        """, unsafe_allow_html=True)

    _render_idle_panel()


# ═════════════════════════════════════════════════════════════════════════════
#  Camera loop
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.camera_running:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        err_placeholder.markdown(
            '<div class="err-box">⚠ Cannot open webcam. '
            'Check that no other app is using the camera.</div>',
            unsafe_allow_html=True
        )
        st.session_state.camera_running = False
        st.rerun()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while st.session_state.camera_running:
            ok, frame = cap.read()
            if not ok:
                err_placeholder.markdown(
                    '<div class="err-box">⚠ Frame read failed — check camera connection.</div>',
                    unsafe_allow_html=True
                )
                break

            st.session_state.frame_count += 1
            res = predict_gesture(frame)

            # ── Show annotated frame ──────────────────────────────────────
            rgb = cv2.cvtColor(res["annotated"], cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, channels="RGB", width=640)

            if res["error"]:
                err_placeholder.markdown(
                    f'<div class="err-box">⚠ {res["error"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                err_placeholder.empty()

            label      = res["label"]
            confidence = res["confidence"]
            hand_found = res["hand_found"]

            # ── Apply threshold ───────────────────────────────────────────
            if hand_found and confidence < conf_threshold:
                label      = "Low Confidence"
                hand_found = False

            # ── Prediction card ───────────────────────────────────────────
            if hand_found:
                pred_html = f"""
                <div class="pred-wrap">
                    <div class="pred-tag">GESTURE PREDICTION</div>
                    <div class="pred-label">{label}</div>
                    <div style="font-family:'Space Mono',monospace; font-size:.7rem;
                                color:#3a8a6a; margin-top:.5rem;">
                        {_conf_emoji(confidence)}&nbsp; Detected with confidence
                    </div>
                </div>"""
            else:
                no_msg = label if label != "No Hand Detected" else "No Hand Detected"
                pred_html = f"""
                <div class="pred-wrap">
                    <div class="pred-tag">GESTURE PREDICTION</div>
                    <div class="pred-label no-hand">{no_msg}</div>
                </div>"""
            pred_placeholder.markdown(pred_html, unsafe_allow_html=True)

            # ── Confidence bar ─────────────────────────────────────────────
            if hand_found:
                pct     = int(confidence * 100)
                c_class = _conf_class(confidence)
                conf_placeholder.markdown(f"""
                <div class="card conf-wrap" style="margin-top:.6rem;">
                    <div class="conf-label">
                        <span>Confidence Score</span>
                        <span style="color:var(--text-main);">{pct}%</span>
                    </div>
                    <div class="conf-track">
                        <div class="conf-fill {c_class}" style="width:{pct}%"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                conf_placeholder.empty()

            # ── Update history ─────────────────────────────────────────────
            if hand_found and confidence >= conf_threshold:
                st.session_state.total_detections += 1
                history_list = list(st.session_state.history)
                # Only append if different from last OR enough time passed
                if not history_list or history_list[-1]["label"] != label:
                    st.session_state.history.append({
                        "label":      label,
                        "confidence": confidence,
                        "time":       datetime.now().strftime("%H:%M:%S"),
                    })
                    _speak(label)

            # ── Stats tiles ────────────────────────────────────────────────
            elapsed = int(time.time() - st.session_state.session_start)
            mm, ss  = divmod(elapsed, 60)
            stats_placeholder.markdown(f"""
            <div class="tile-grid" style="margin-top:.7rem;">
                <div class="tile">
                    <div class="tile-val">{st.session_state.total_detections}</div>
                    <div class="tile-key">Total Detections</div>
                </div>
                <div class="tile">
                    <div class="tile-val">{mm:02d}:{ss:02d}</div>
                    <div class="tile-key">Session Time</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # ── History log ────────────────────────────────────────────────
            history_list = list(st.session_state.history)
            if history_list:
                items_html = "".join([
                    f'<div class="hist-item">'
                    f'  <span class="hist-gesture">{e["label"]}</span>'
                    f'  <span class="hist-conf">{int(e["confidence"]*100)}%</span>'
                    f'  <span class="hist-time">{e["time"]}</span>'
                    f'</div>'
                    for e in reversed(history_list)
                ])
                hist_placeholder.markdown(f"""
                <div class="card" style="margin-top:.7rem;">
                    <div class="card-title">🕘  Recent History</div>
                    {items_html}
                </div>""", unsafe_allow_html=True)

            # ── Frame pacing ───────────────────────────────────────────────
            time.sleep(0.03)  # ~30 fps ceiling

    finally:
        cap.release()

    st.session_state.camera_running = False


# ═════════════════════════════════════════════════════════════════════════════
#  Footer
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; font-family:'Space Mono',monospace;
            font-size:.68rem; color:#3a5a4a; padding: .4rem 0 1.2rem; letter-spacing:.05em;">
    Hand Gesture Recognition System for Sign Language &nbsp;·&nbsp;
    BCA Final Year Project &nbsp;·&nbsp;
    Anjali Negi &amp; Divyansh Agrawal &nbsp;·&nbsp;
    IINTM 2023–26
</div>
""", unsafe_allow_html=True)
