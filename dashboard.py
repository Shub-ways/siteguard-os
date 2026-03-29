import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
import sqlite3
import time
import threading
import smtplib
from email.message import EmailMessage
from datetime import datetime
from PIL import Image
import pandas as pd
from dotenv import load_dotenv

# ──────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="SiteGuard OS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
#  GLOBAL CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=IBM+Plex+Mono:wght@300;400;600&family=Barlow+Condensed:wght@300;400;600&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0a0c0f;
    --surface:   #111318;
    --border:    #1e2330;
    --accent:    #f59e0b;        /* amber */
    --accent2:   #ef4444;        /* red alert */
    --ok:        #10b981;        /* green */
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font-display: 'Orbitron', monospace;
    --font-mono:    'IBM Plex Mono', monospace;
    --font-body:    'Barlow Condensed', sans-serif;
}

/* ── Full app background ── */
.stApp {
    background: var(--bg);
    background-image:
        radial-gradient(ellipse 80% 40% at 50% -10%, rgba(245,158,11,0.07) 0%, transparent 70%),
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(30,35,48,0.5) 39px, rgba(30,35,48,0.5) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(30,35,48,0.5) 39px, rgba(30,35,48,0.5) 40px);
    color: var(--text);
    font-family: var(--font-body);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem; max-width: 100%; }

/* ── Master header ── */
.site-header {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    padding: 1.2rem 1.8rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 4px;
    margin-bottom: 1.5rem;
}
.site-header .logo {
    font-family: var(--font-display);
    font-size: 1.6rem;
    font-weight: 900;
    color: var(--accent);
    letter-spacing: 0.1em;
    line-height: 1;
}
.site-header .sub {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-top: 2px;
}
.site-header .badge {
    margin-left: auto;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--muted);
    text-align: right;
    letter-spacing: 0.1em;
}
.site-header .badge span {
    display: block;
    color: var(--ok);
    font-size: 0.75rem;
}

/* ── Section cards ── */
.section-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    height: 100%;
}
.section-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    background: var(--accent);
    border-radius: 50%;
}
.section-title {
    font-family: var(--font-display);
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.8rem;
    border-radius: 2px;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    font-weight: 600;
    text-transform: uppercase;
    border: 1px solid;
}
.pill-idle   { color: var(--muted); border-color: var(--muted); background: rgba(100,116,139,0.08); }
.pill-active { color: var(--ok);    border-color: var(--ok);    background: rgba(16,185,129,0.08); }
.pill-alert  { color: var(--accent2); border-color: var(--accent2); background: rgba(239,68,68,0.10);
               animation: pulse-border 1s infinite; }

@keyframes pulse-border {
    0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
    50%      { box-shadow: 0 0 0 4px rgba(239,68,68,0); }
}

/* ── Metric tiles ── */
.metric-row {
    display: flex;
    gap: 0.8rem;
    margin-bottom: 1rem;
}
.metric-tile {
    flex: 1;
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0.75rem 1rem;
}
.metric-tile .val {
    font-family: var(--font-display);
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.metric-tile .lbl {
    font-family: var(--font-mono);
    font-size: 0.58rem;
    color: var(--muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Video feed frame ── */
.feed-wrapper {
    background: #000;
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}
.feed-overlay-label {
    position: absolute;
    top: 10px; left: 10px;
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--accent);
    letter-spacing: 0.2em;
    background: rgba(0,0,0,0.7);
    padding: 3px 8px;
    border-radius: 2px;
    z-index: 5;
}
.stImage { border: 1px solid var(--border) !important; border-radius: 4px; overflow: hidden; }

/* ── Toggle / checkbox override ── */
.stCheckbox label {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.12em !important;
    color: var(--text) !important;
}
.stCheckbox [data-testid="stCheckbox"] > div { gap: 10px; }

/* ── Dataframe table ── */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 4px; }
.stDataFrame table { font-family: var(--font-mono) !important; font-size: 0.73rem !important; }
.stDataFrame thead th {
    background: rgba(245,158,11,0.08) !important;
    color: var(--accent) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
}
.stDataFrame tbody tr:hover td { background: rgba(255,255,255,0.03) !important; }

/* ── Alert banner ── */
.alert-banner {
    background: rgba(239,68,68,0.12);
    border: 1px solid var(--accent2);
    border-left: 4px solid var(--accent2);
    border-radius: 3px;
    padding: 0.7rem 1rem;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--accent2);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    animation: flicker 1.5s infinite;
}
@keyframes flicker {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.7; }
}

/* ── Log entry row ── */
.log-entry {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.45rem 0.6rem;
    border-bottom: 1px solid rgba(30,35,48,0.8);
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text);
}
.log-entry:first-child { border-top: 1px solid rgba(30,35,48,0.8); }
.log-entry .ts { color: var(--muted); min-width: 80px; }
.log-entry .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--ok); flex-shrink: 0; }

/* ── Divider ── */
.hl { height: 1px; background: var(--border); margin: 1rem 0; }

/* ── Sidebar override ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  HELPER FUNCTIONS (unchanged logic)
# ──────────────────────────────────────────────
def get_ear(eye_points):
    p = np.array(eye_points)
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    return (A + B) / (2.0 * C)

@st.cache_resource
def load_known_faces(workers_dir="registered_workers"):
    known_encodings, known_names = [], []
    if not os.path.exists(workers_dir):
        os.makedirs(workers_dir)
        return known_encodings, known_names
    for filename in os.listdir(workers_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            try:
                pil_image = Image.open(os.path.join(workers_dir, filename)).convert("RGB")
                encodings = face_recognition.face_encodings(np.array(pil_image))
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(os.path.splitext(filename)[0].replace("_", " "))
            except Exception:
                pass
    return known_encodings, known_names

def send_alert_email(image_path, time_str, sender, password, receiver):
    try:
        msg = EmailMessage()
        msg['Subject'] = f"URGENT: Security Breach at {time_str}"
        msg['From'] = sender
        msg['To'] = receiver
        msg.set_content("Unknown individual detected in restricted zone. See attached image.")
        with open(image_path, 'rb') as f:
            msg.add_attachment(f.read(), maintype='image', subtype='jpeg', filename=os.path.basename(image_path))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
    except Exception as e:
        print(f"Email error: {e}")

def init_db():
    conn = sqlite3.connect('security_data.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_name TEXT NOT NULL,
            entry_date TEXT NOT NULL,
            entry_time TEXT NOT NULL
        )
    ''')
    conn.commit()
    return conn


# ──────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────
st.markdown(f"""
<div class="site-header">
    <div>
        <div class="logo">🛡 SITEGUARD OS</div>
        <div class="sub">Construction Site Intelligence Platform · v3.1</div>
    </div>
    <div class="badge">
        SYSTEM DATE<br><span>{datetime.now().strftime('%d %b %Y  %H:%M')}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  TOP METRICS
# ──────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
    <div class="metric-tile">
        <div class="val" id="m-cam">1</div>
        <div class="lbl">Active Cameras</div>
    </div>""", unsafe_allow_html=True)

with m2:
    enrolled_count = 0
    if os.path.exists("registered_workers"):
        enrolled_count = len([f for f in os.listdir("registered_workers")
                               if f.endswith((".jpg",".jpeg",".png"))])
    st.markdown(f"""
    <div class="metric-tile">
        <div class="val">{enrolled_count}</div>
        <div class="lbl">Enrolled Workers</div>
    </div>""", unsafe_allow_html=True)

with m3:
    today_count = 0
    try:
        conn_tmp = sqlite3.connect('security_data.db', check_same_thread=False)
        today_count = conn_tmp.execute(
            "SELECT COUNT(*) FROM attendance WHERE entry_date=?",
            (datetime.now().strftime("%Y-%m-%d"),)
        ).fetchone()[0]
        conn_tmp.close()
    except Exception:
        pass
    st.markdown(f"""
    <div class="metric-tile">
        <div class="val">{today_count}</div>
        <div class="lbl">Check-ins Today</div>
    </div>""", unsafe_allow_html=True)

with m4:
    log_count = len(os.listdir("logs")) if os.path.exists("logs") else 0
    st.markdown(f"""
    <div class="metric-tile">
        <div class="val">{log_count}</div>
        <div class="lbl">Intruder Snapshots</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  MAIN LAYOUT: Feed (left) · Logs + Status (right)
# ──────────────────────────────────────────────
col_feed, col_right = st.columns([3, 2], gap="medium")

with col_feed:
    st.markdown('<div class="section-label">Live Camera Feed</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">ZONE SURVEILLANCE MONITOR</div>', unsafe_allow_html=True)
    
    # ARM toggle
    run_system = st.checkbox("⬤  ARM SECURITY SYSTEM", value=False)
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    
    status_placeholder = st.empty()
    FRAME_WINDOW = st.empty()
    alert_placeholder = st.empty()

with col_right:
    st.markdown('<div class="section-label">Attendance Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">REAL-TIME ENTRY LOG</div>', unsafe_allow_html=True)
    log_placeholder = st.empty()

    st.markdown("<div class='hl'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">System Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">OPERATIONAL STATUS</div>', unsafe_allow_html=True)
    diag_placeholder = st.empty()


# ──────────────────────────────────────────────
#  IDLE STATE
# ──────────────────────────────────────────────
if not run_system:
    status_placeholder.markdown("""
    <div class="status-pill pill-idle">◉ &nbsp; SYSTEM OFFLINE</div>
    """, unsafe_allow_html=True)

    FRAME_WINDOW.markdown("""
    <div style="background:#000;border:1px solid #1e2330;border-radius:4px;
                height:340px;display:flex;flex-direction:column;
                align-items:center;justify-content:center;gap:1rem;">
        <div style="font-size:3rem;opacity:0.15">📷</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                    color:#64748b;letter-spacing:0.25em;text-transform:uppercase;">
            NO SIGNAL · AWAITING ARM COMMAND
        </div>
    </div>""", unsafe_allow_html=True)

    diag_placeholder.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#64748b;
                line-height:2;letter-spacing:0.08em;">
        <div>▸ CAMERA MODULE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#ef4444">OFFLINE</span></div>
        <div>▸ FACE ENGINE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#ef4444">STANDBY</span></div>
        <div>▸ GEOFENCE ZONE &nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#ef4444">INACTIVE</span></div>
        <div>▸ DATABASE CONN &nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#10b981">READY</span></div>
        <div>▸ EMAIL ALERTS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#f59e0b">PENDING</span></div>
    </div>""", unsafe_allow_html=True)

    log_placeholder.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                color:#64748b;text-align:center;padding:2rem;
                border:1px solid #1e2330;border-radius:4px;">
        ARM system to begin logging
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  ARMED STATE
# ──────────────────────────────────────────────
if run_system:
    status_placeholder.markdown("""
    <div class="status-pill pill-active">◉ &nbsp; SYSTEM ARMED · MONITORING</div>
    """, unsafe_allow_html=True)

    diag_placeholder.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#64748b;
                line-height:2;letter-spacing:0.08em;">
        <div>▸ CAMERA MODULE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#10b981">ACTIVE</span></div>
        <div>▸ FACE ENGINE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#10b981">RUNNING</span></div>
        <div>▸ GEOFENCE ZONE &nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#10b981">ARMED</span></div>
        <div>▸ DATABASE CONN &nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#10b981">CONNECTED</span></div>
        <div>▸ EMAIL ALERTS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#10b981">ENABLED</span></div>
    </div>""", unsafe_allow_html=True)

    load_dotenv()
    SENDER   = os.getenv("SENDER_EMAIL")
    PASSWORD = os.getenv("SENDER_PASSWORD")
    RECEIVER = os.getenv("RECEIVER_EMAIL")

    db_conn = init_db()
    known_face_encodings, known_face_names = load_known_faces()
    video = cv2.VideoCapture(0)
    restricted_zone = np.array([[150, 100], [500, 100], [550, 400], [100, 400]], np.int32)

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logged_workers, blink_counters, liveness_verified = {}, {}, {}
    last_log_time, last_email_time = 0, 0
    log_cooldown, email_cooldown = 10, 60
    EAR_THRESHOLD, EAR_CONSEC_FRAMES = 0.26, 1
    area_was_clear = True
    process_this_frame = True

    while run_system:
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        clean_frame = frame.copy()

        # Draw restricted zone overlay
        overlay = frame.copy()
        cv2.fillPoly(overlay, [restricted_zone], (255, 165, 0))
        cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)
        cv2.polylines(frame, [restricted_zone], isClosed=True, color=(245, 158, 11), thickness=2)
        cv2.putText(frame, "[ RESTRICTED ZONE ]", (155, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 158, 11), 1, cv2.LINE_AA)

        # Corner brackets (HUD style)
        for (x, y, dx, dy) in [(10,10,1,1),(630,10,-1,1),(10,470,1,-1),(630,470,-1,-1)]:
            cv2.line(frame, (x, y), (x + dx*20, y), (245,158,11), 2)
            cv2.line(frame, (x, y), (x, y + dy*20), (245,158,11), 2)

        # Timestamp HUD
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 116, 139), 1, cv2.LINE_AA)
        cv2.putText(frame, "SITEGUARD CAM-01", (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245,158,11), 1, cv2.LINE_AA)

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        if process_this_frame:
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations     = face_recognition.face_locations(rgb_small)
            face_encodings_list = face_recognition.face_encodings(rgb_small, face_locations)
            face_landmarks_list = face_recognition.face_landmarks(rgb_small, face_locations)

        process_this_frame = not process_this_frame
        current_frame_names = []
        intruder_in_frame   = False

        for (top, right, bottom, left), face_encoding, face_landmarks in zip(
                face_locations, face_encodings_list, face_landmarks_list):
            top *= 4; right *= 4; bottom *= 4; left *= 4

            cx = left + (right - left) // 2
            cy = top  + (bottom - top) // 2
            is_inside = cv2.pointPolygonTest(restricted_zone, (cx, cy), False) >= 0

            if not is_inside:
                continue

            name, box_color = "UNKNOWN", (239, 68, 68)
            is_authorized = False

            if known_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.5)
                if True in matches:
                    worker_name = known_face_names[matches.index(True)]
                    current_frame_names.append(worker_name)

                    ear = (get_ear(face_landmarks['left_eye']) + get_ear(face_landmarks['right_eye'])) / 2.0
                    if ear < EAR_THRESHOLD:
                        blink_counters[worker_name] = blink_counters.get(worker_name, 0) + 1
                    elif blink_counters.get(worker_name, 0) >= EAR_CONSEC_FRAMES:
                        liveness_verified[worker_name] = True
                        blink_counters[worker_name] = 0

                    if liveness_verified.get(worker_name, False):
                        name, box_color = worker_name, (16, 185, 129)
                        is_authorized = True
                        current_time = time.time()
                        if worker_name not in logged_workers or (current_time - logged_workers[worker_name]) > 60:
                            now = datetime.now()
                            db_conn.cursor().execute(
                                'INSERT INTO attendance (worker_name, entry_date, entry_time) VALUES (?,?,?)',
                                (worker_name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"))
                            )
                            db_conn.commit()
                            logged_workers[worker_name] = current_time
                    else:
                        name, box_color = f"VERIFYING: {worker_name}", (253, 224, 71)
                        is_authorized = True

            if not is_authorized:
                intruder_in_frame = True

            # Draw detection box (clean HUD style)
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            label_bg_h = 22
            cv2.rectangle(frame, (left, top - label_bg_h), (right, top), box_color, -1)
            cv2.putText(frame, name.upper(), (left + 4, top - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        # Cleanup liveness state
        for n in [k for k in liveness_verified if k not in current_frame_names]:
            del liveness_verified[n]
            blink_counters.pop(n, None)

        current_time = time.time()
        if intruder_in_frame:
            # Red scanline overlay
            for y in range(0, 480, 6):
                cv2.line(frame, (0, y), (640, y), (239, 68, 68), 1)
            frame = cv2.addWeighted(frame, 0.85, np.full_like(frame, (30, 0, 0)), 0.15, 0)
            cv2.putText(frame, "SECURITY BREACH DETECTED", (60, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (239, 68, 68), 2, cv2.LINE_AA)

            try:
                import winsound
                winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_ASYNC)
            except Exception:
                pass

            if area_was_clear or (current_time - last_email_time) > email_cooldown:
                filepath = os.path.join(log_dir, f"intruder_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
                cv2.imwrite(filepath, cv2.cvtColor(clean_frame, cv2.COLOR_RGB2BGR))
                if SENDER and PASSWORD and RECEIVER:
                    threading.Thread(target=send_alert_email,
                                     args=(filepath, datetime.now().strftime('%H:%M:%S'), SENDER, PASSWORD, RECEIVER)).start()
                last_email_time = current_time
                last_log_time   = current_time
                area_was_clear  = False

            elif (current_time - last_log_time) > log_cooldown:
                filepath = os.path.join(log_dir, f"intruder_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
                cv2.imwrite(filepath, cv2.cvtColor(clean_frame, cv2.COLOR_RGB2BGR))
                last_log_time = current_time

            alert_placeholder.markdown("""
            <div class="alert-banner">⚠ UNAUTHORIZED ACCESS DETECTED IN RESTRICTED ZONE · ALERT DISPATCHED</div>
            """, unsafe_allow_html=True)
        else:
            area_was_clear = True
            alert_placeholder.empty()

        # Render frame
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width='stretch')

        # Attendance table
        try:
            df = pd.read_sql_query(
                "SELECT worker_name AS 'WORKER', entry_date AS 'DATE', entry_time AS 'TIME' "
                "FROM attendance ORDER BY id DESC LIMIT 12",
                db_conn
            )
            log_placeholder.dataframe(df, width='stretch', hide_index=True)
        except Exception:
            pass

    video.release()
    db_conn.close()