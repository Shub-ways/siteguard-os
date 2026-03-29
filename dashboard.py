import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
import sqlite3
import time
import threading
import smtplib
import json
from email.message import EmailMessage
from datetime import datetime
from PIL import Image
import pandas as pd
from dotenv import load_dotenv

# ──────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="SiteGuard OS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
#  ZONE COLOUR PALETTE
# ──────────────────────────────────────────────
ZONE_PALETTE = [
    {"name": "Amber",   "bgr": (30,  144, 255), "hex": "#f59e0b"},
    {"name": "Crimson", "bgr": (60,   20, 220), "hex": "#ef4444"},
    {"name": "Cyan",    "bgr": (180, 200,  30), "hex": "#06b6d4"},
    {"name": "Violet",  "bgr": (200,  80, 160), "hex": "#8b5cf6"},
    {"name": "Lime",    "bgr": (40,  200,  80), "hex": "#84cc16"},
    {"name": "Pink",    "bgr": (180,  80, 220), "hex": "#ec4899"},
]

# ──────────────────────────────────────────────
#  GLOBAL CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=IBM+Plex+Mono:wght@300;400;600&family=Barlow+Condensed:wght@300;400;600&display=swap');

:root {
    --bg:        #0a0c0f;
    --surface:   #111318;
    --border:    #1e2330;
    --accent:    #f59e0b;
    --accent2:   #ef4444;
    --ok:        #10b981;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font-display: 'Orbitron', monospace;
    --font-mono:    'IBM Plex Mono', monospace;
    --font-body:    'Barlow Condensed', sans-serif;
}

.stApp {
    background: var(--bg);
    background-image:
        radial-gradient(ellipse 80% 40% at 50% -10%, rgba(245,158,11,0.07) 0%, transparent 70%),
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(30,35,48,0.5) 39px, rgba(30,35,48,0.5) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(30,35,48,0.5) 39px, rgba(30,35,48,0.5) 40px);
    color: var(--text);
    font-family: var(--font-body);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem; max-width: 100%; }

.site-header {
    display: flex; align-items: center; gap: 1.2rem;
    padding: 1.2rem 1.8rem;
    background: var(--surface); border: 1px solid var(--border);
    border-left: 4px solid var(--accent); border-radius: 4px; margin-bottom: 1.5rem;
}
.site-header .logo { font-family: var(--font-display); font-size: 1.6rem; font-weight: 900; color: var(--accent); letter-spacing: 0.1em; line-height: 1; }
.site-header .sub  { font-family: var(--font-mono); font-size: 0.7rem; color: var(--muted); letter-spacing: 0.25em; text-transform: uppercase; margin-top: 2px; }
.site-header .badge { margin-left: auto; font-family: var(--font-mono); font-size: 0.65rem; color: var(--muted); text-align: right; letter-spacing: 0.1em; }
.site-header .badge span { display: block; color: var(--ok); font-size: 0.75rem; }

.section-label {
    font-family: var(--font-mono); font-size: 0.6rem; letter-spacing: 0.3em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 0.6rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-label::before { content:''; display:inline-block; width:6px; height:6px; background:var(--accent); border-radius:50%; }
.section-title { font-family: var(--font-display); font-size: 0.95rem; font-weight: 700; color: var(--text); letter-spacing: 0.05em; margin-bottom: 1rem; }

.status-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.3rem 0.8rem; border-radius: 2px;
    font-family: var(--font-mono); font-size: 0.7rem;
    letter-spacing: 0.15em; font-weight: 600; text-transform: uppercase; border: 1px solid;
}
.pill-idle   { color: var(--muted);   border-color: var(--muted);   background: rgba(100,116,139,0.08); }
.pill-active { color: var(--ok);      border-color: var(--ok);      background: rgba(16,185,129,0.08);  }
.pill-warn   { color: #f59e0b;        border-color: #f59e0b;        background: rgba(245,158,11,0.08);  }

@keyframes flicker { 0%,100%{opacity:1} 50%{opacity:0.7} }

.metric-tile { flex:1; background:rgba(255,255,255,0.02); border:1px solid var(--border); border-radius:3px; padding:0.75rem 1rem; }
.metric-tile .val { font-family:var(--font-display); font-size:1.6rem; font-weight:700; color:var(--accent); line-height:1; }
.metric-tile .lbl { font-family:var(--font-mono); font-size:0.58rem; color:var(--muted); letter-spacing:0.2em; text-transform:uppercase; margin-top:4px; }

.stImage { border:1px solid var(--border) !important; border-radius:4px; overflow:hidden; }
.stCheckbox label { font-family:var(--font-mono) !important; font-size:0.78rem !important; letter-spacing:0.12em !important; color:var(--text) !important; }

.stDataFrame { border:1px solid var(--border) !important; border-radius:4px; }
.stDataFrame table { font-family:var(--font-mono) !important; font-size:0.73rem !important; }
.stDataFrame thead th { background:rgba(245,158,11,0.08) !important; color:var(--accent) !important; font-size:0.65rem !important; letter-spacing:0.2em !important; text-transform:uppercase !important; }
.stDataFrame tbody tr:hover td { background:rgba(255,255,255,0.03) !important; }

.alert-banner {
    background:rgba(239,68,68,0.12); border:1px solid var(--accent2);
    border-left:4px solid var(--accent2); border-radius:3px; padding:0.7rem 1rem;
    font-family:var(--font-mono); font-size:0.72rem; color:var(--accent2);
    letter-spacing:0.1em; text-transform:uppercase; animation:flicker 1.5s infinite;
}

.editor-hint {
    background:rgba(245,158,11,0.06); border:1px solid rgba(245,158,11,0.2);
    border-radius:3px; padding:0.6rem 0.9rem;
    font-family:var(--font-mono); font-size:0.64rem; color:var(--muted);
    line-height:1.8; margin-bottom:0.8rem;
}

.zone-row {
    display:flex; align-items:center; gap:0.6rem;
    padding:0.5rem 0.7rem; margin-bottom:0.4rem;
    border-radius:3px; border:1px solid;
}
.zone-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.zone-row-name { font-family:var(--font-mono); font-size:0.72rem; font-weight:600; }
.zone-row-pts  { font-family:var(--font-mono); font-size:0.6rem;  color:var(--muted); margin-left:auto; }

.hl { height:1px; background:var(--border); margin:1rem 0; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  CORE HELPERS
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

def send_alert_email(image_path, time_str, zone_name, sender, password, receiver):
    try:
        msg = EmailMessage()
        msg['Subject'] = f"URGENT: Breach in [{zone_name}] at {time_str}"
        msg['From']    = sender
        msg['To']      = receiver
        msg.set_content(f"Unknown individual detected in zone '{zone_name}'. See attached image.")
        with open(image_path, 'rb') as f:
            msg.add_attachment(f.read(), maintype='image', subtype='jpeg',
                               filename=os.path.basename(image_path))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
    except Exception as e:
        print(f"Email error: {e}")

def init_db():
    conn = sqlite3.connect('security_data.db', check_same_thread=False)
    conn.cursor().execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_name TEXT NOT NULL,
            zone_name   TEXT NOT NULL DEFAULT 'Unknown',
            entry_date  TEXT NOT NULL,
            entry_time  TEXT NOT NULL
        )
    ''')
    conn.commit()
    return conn

def save_zones(zones):
    with open("zones.json", "w") as f:
        json.dump(zones, f)

def load_zones_from_disk():
    if os.path.exists("zones.json"):
        with open("zones.json") as f:
            return json.load(f)
    return []

def draw_all_zones(frame, zones):
    """Render all zones on frame with fill, border, label, and vertex dots."""
    for zone in zones:
        pts     = np.array(zone["points"], np.int32)
        hex_col = zone.get("color_hex", "#f59e0b")
        h = hex_col.lstrip("#")
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        bgr = (b, g, r)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], bgr)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=bgr, thickness=2)

        # Label at centroid
        M = cv2.moments(pts)
        cx_z = int(M["m10"]/M["m00"]) if M["m00"] else pts[0][0]
        cy_z = int(M["m01"]/M["m00"]) if M["m00"] else pts[0][1]
        label = f"[ {zone['name'].upper()} ]"
        (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(frame, (cx_z-tw//2-4, cy_z-th_-6), (cx_z+tw//2+4, cy_z+4), bgr, -1)
        cv2.putText(frame, label, (cx_z-tw//2, cy_z),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,0), 1, cv2.LINE_AA)

        # Vertex dots
        for pt in zone["points"]:
            cv2.circle(frame, tuple(pt), 4, bgr,    -1)
            cv2.circle(frame, tuple(pt), 5, (0,0,0), 1)
    return frame

def point_in_any_zone(cx, cy, zones):
    for zone in zones:
        pts = np.array(zone["points"], np.int32)
        if cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0:
            return True, zone["name"]
    return False, None


# ──────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────
if "zones"          not in st.session_state: st.session_state.zones          = load_zones_from_disk()
if "current_pts"    not in st.session_state: st.session_state.current_pts    = []
if "snap_rgb"       not in st.session_state: st.session_state.snap_rgb       = None
if "last_click"     not in st.session_state: st.session_state.last_click     = None


# ──────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────
st.markdown(f"""
<div class="site-header">
    <div>
        <div class="logo">🛡 SITEGUARD OS</div>
        <div class="sub">Multi-Zone Security Intelligence Platform · v4.0</div>
    </div>
    <div class="badge">SYSTEM DATE<br><span>{datetime.now().strftime('%d %b %Y  %H:%M')}</span></div>
</div>
""", unsafe_allow_html=True)

# ── Metrics ──
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="metric-tile"><div class="val">1</div><div class="lbl">Active Cameras</div></div>', unsafe_allow_html=True)
with m2:
    n_workers = len([f for f in os.listdir("registered_workers") if f.endswith((".jpg",".jpeg",".png"))]) if os.path.exists("registered_workers") else 0
    st.markdown(f'<div class="metric-tile"><div class="val">{n_workers}</div><div class="lbl">Enrolled Workers</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-tile"><div class="val">{len(st.session_state.zones)}</div><div class="lbl">Defined Zones</div></div>', unsafe_allow_html=True)
with m4:
    n_logs = len(os.listdir("logs")) if os.path.exists("logs") else 0
    st.markdown(f'<div class="metric-tile"><div class="val">{n_logs}</div><div class="lbl">Intruder Snapshots</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  3-COLUMN LAYOUT
# ──────────────────────────────────────────────
col_editor, col_feed, col_right = st.columns([1.5, 2.5, 1.6], gap="medium")


# ══════════════════════════════════════════════════════
#  LEFT  ·  ZONE EDITOR
# ══════════════════════════════════════════════════════
with col_editor:
    st.markdown('<div class="section-label">Zone Configuration</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">ZONE EDITOR</div>',       unsafe_allow_html=True)

    st.markdown("""
    <div class="editor-hint">
        <b style="color:#f59e0b">How to define a zone:</b><br>
        ① Click <b>Capture Frame</b> to grab camera snapshot<br>
        ② Enter a name &amp; pick a colour<br>
        ③ Click points directly on the image (≥ 3)<br>
        ④ Click <b>Save Zone</b> — repeat for more zones
    </div>
    """, unsafe_allow_html=True)

    # ── Step 1: Capture ──
    if st.button("📷  Capture Frame from Camera", use_container_width=True):
        cap = cv2.VideoCapture(0)
        ret, snap = cap.read()
        cap.release()
        if ret:
            snap_rgb = cv2.cvtColor(cv2.resize(snap, (640, 480)), cv2.COLOR_BGR2RGB)
            st.session_state.snap_rgb    = snap_rgb
            st.session_state.current_pts = []
            st.session_state.last_click  = None
        else:
            st.error("Camera not accessible.")

    # ── Step 2: Name + colour ──
    zone_name = st.text_input("Zone name", placeholder="e.g. Crane Area, Gate, Vault")
    color_choice = st.selectbox("Zone colour", [p["name"] for p in ZONE_PALETTE])
    palette_entry = next(p for p in ZONE_PALETTE if p["name"] == color_choice)

    # ── Step 3: Click points on snapshot ──
    if st.session_state.snap_rgb is not None:
        # Build preview with existing zones + current in-progress points
        preview = draw_all_zones(st.session_state.snap_rgb.copy(), st.session_state.zones)
        pts = st.session_state.current_pts

        # Draw in-progress polygon
        h_col = palette_entry["hex"].lstrip("#")
        pr, pg, pb = int(h_col[0:2],16), int(h_col[2:4],16), int(h_col[4:6],16)
        p_bgr = (pb, pg, pr)  # note: preview is RGB so use as-is below
        p_rgb = (pr, pg, pb)

        for i, pt in enumerate(pts):
            cv2.circle(preview, tuple(pt), 6, p_rgb, -1)
            cv2.circle(preview, tuple(pt), 7, (0,0,0), 1)
            cv2.putText(preview, str(i+1), (pt[0]+9, pt[1]-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)
        if len(pts) >= 2:
            for i in range(len(pts)-1):
                cv2.line(preview, tuple(pts[i]), tuple(pts[i+1]), p_rgb, 1)
        if len(pts) >= 3:
            cv2.line(preview, tuple(pts[-1]), tuple(pts[0]), p_rgb, 1)
            # Faint fill preview
            poly_overlay = preview.copy()
            cv2.fillPoly(poly_overlay, [np.array(pts, np.int32)], p_rgb)
            cv2.addWeighted(poly_overlay, 0.15, preview, 0.85, 0, preview)

        # ── Clickable image via streamlit-image-coordinates ──
        try:
            from streamlit_image_coordinates import streamlit_image_coordinates
            coords = streamlit_image_coordinates(Image.fromarray(preview), key="zone_img")
            if coords:
                new_pt = [int(coords["x"]), int(coords["y"])]
                last   = st.session_state.last_click
                # Only register if different from last click (avoid double-fire)
                if last is None or new_pt != last:
                    st.session_state.current_pts.append(new_pt)
                    st.session_state.last_click = new_pt
                    st.rerun()
        except ImportError:
            # Fallback: show image, manual coordinate entry
            st.image(preview, use_column_width=True)
            st.caption("💡 Install `streamlit-image-coordinates` for click-to-place.")
            c_x = st.number_input("Point X (0–640)", 0, 640, 320, key="px")
            c_y = st.number_input("Point Y (0–480)", 0, 480, 240, key="py")
            if st.button("➕ Add Point", use_container_width=True):
                st.session_state.current_pts.append([c_x, c_y])
                st.rerun()

        # Point counter
        n_pts = len(pts)
        pt_color = "#10b981" if n_pts >= 3 else "#f59e0b"
        st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
            color:{pt_color};margin:0.3rem 0 0.6rem;">
            {'✓' if n_pts>=3 else '○'} {n_pts} point{'s' if n_pts!=1 else ''}
            {'— ready to save' if n_pts>=3 else f' (need {3-n_pts} more)'}
        </div>""", unsafe_allow_html=True)

        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("↩ Undo", use_container_width=True):
                if st.session_state.current_pts:
                    st.session_state.current_pts.pop()
                    st.session_state.last_click = None
                    st.rerun()
        with btn2:
            if st.button("✕ Clear", use_container_width=True):
                st.session_state.current_pts = []
                st.session_state.last_click  = None
                st.rerun()

        # ── Save Zone ──
        if st.button("💾  Save Zone", type="primary", use_container_width=True):
            if len(st.session_state.current_pts) < 3:
                st.error("Need at least 3 points.")
            elif not zone_name.strip():
                st.error("Enter a zone name.")
            else:
                st.session_state.zones.append({
                    "name":      zone_name.strip(),
                    "points":    st.session_state.current_pts.copy(),
                    "color_hex": palette_entry["hex"],
                })
                save_zones(st.session_state.zones)
                st.session_state.current_pts = []
                st.session_state.last_click  = None
                st.session_state.snap_rgb    = None
                st.success(f"✓ Zone '{zone_name.strip()}' saved!")
                st.rerun()
    else:
        # Placeholder before capture
        st.markdown("""
        <div style="background:#050708;border:1px dashed #1e2330;border-radius:4px;
                    height:200px;display:flex;flex-direction:column;
                    align-items:center;justify-content:center;gap:0.6rem;margin-bottom:0.8rem;">
            <div style="font-size:2.5rem;opacity:0.1">🗺</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                        color:#64748b;letter-spacing:0.2em;">CAPTURE FRAME TO START</div>
        </div>""", unsafe_allow_html=True)

    # ── Saved zones list ──
    st.markdown("<div class='hl'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Saved Zones</div>', unsafe_allow_html=True)

    if not st.session_state.zones:
        st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:#64748b;padding:0.4rem 0;">No zones defined yet.</div>', unsafe_allow_html=True)
    else:
        for idx, zone in enumerate(st.session_state.zones):
            hx = zone.get("color_hex","#f59e0b")
            n  = len(zone["points"])
            st.markdown(f"""
            <div class="zone-row" style="border-color:{hx}44;background:{hx}0d;">
                <div class="zone-dot" style="background:{hx};"></div>
                <div class="zone-row-name" style="color:{hx};">{zone['name']}</div>
                <div class="zone-row-pts">{n} pts</div>
            </div>""", unsafe_allow_html=True)
            if st.button(f"🗑 Delete", key=f"del_{idx}", use_container_width=True):
                st.session_state.zones.pop(idx)
                save_zones(st.session_state.zones)
                st.rerun()

        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        if st.button("🗑  Clear ALL Zones", use_container_width=True):
            st.session_state.zones = []
            save_zones([])
            st.rerun()


# ══════════════════════════════════════════════════════
#  CENTRE  ·  LIVE FEED
# ══════════════════════════════════════════════════════
with col_feed:
    st.markdown('<div class="section-label">Live Camera Feed</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">ZONE SURVEILLANCE MONITOR</div>', unsafe_allow_html=True)

    run_system = st.checkbox("⬤  ARM SECURITY SYSTEM", value=False)
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    status_ph = st.empty()
    feed_ph   = st.empty()
    alert_ph  = st.empty()


# ══════════════════════════════════════════════════════
#  RIGHT  ·  LOG + STATUS
# ══════════════════════════════════════════════════════
with col_right:
    st.markdown('<div class="section-label">Attendance Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">REAL-TIME ENTRY LOG</div>', unsafe_allow_html=True)
    log_ph = st.empty()

    st.markdown("<div class='hl'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">System Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">OPERATIONAL STATUS</div>', unsafe_allow_html=True)
    diag_ph = st.empty()


# ──────────────────────────────────────────────
#  IDLE PLACEHOLDERS
# ──────────────────────────────────────────────
if not run_system:
    status_ph.markdown('<div class="status-pill pill-idle">◉ &nbsp; SYSTEM OFFLINE</div>', unsafe_allow_html=True)
    feed_ph.markdown("""
    <div style="background:#000;border:1px solid #1e2330;border-radius:4px;height:340px;
                display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1rem;">
        <div style="font-size:3rem;opacity:0.12">📷</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#64748b;letter-spacing:0.25em;">
            NO SIGNAL · AWAITING ARM COMMAND
        </div>
    </div>""", unsafe_allow_html=True)

    z_count = len(st.session_state.zones)
    z_col   = "#10b981" if z_count else "#ef4444"
    z_txt   = f"{z_count} ZONE{'S' if z_count!=1 else ''} ARMED" if z_count else "NO ZONES DEFINED"
    diag_ph.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#64748b;line-height:2;letter-spacing:0.08em;">
        <div>▸ CAMERA MODULE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#ef4444">OFFLINE</span></div>
        <div>▸ FACE ENGINE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#ef4444">STANDBY</span></div>
        <div>▸ GEOFENCE ZONES &nbsp;&nbsp;&nbsp;&nbsp; <span style="color:{z_col}">{z_txt}</span></div>
        <div>▸ DATABASE CONN &nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#10b981">READY</span></div>
        <div>▸ EMAIL ALERTS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#f59e0b">PENDING</span></div>
    </div>""", unsafe_allow_html=True)
    log_ph.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#64748b;
                text-align:center;padding:2rem;border:1px solid #1e2330;border-radius:4px;">
        ARM system to begin logging
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  ARMED STATE
# ──────────────────────────────────────────────
if run_system:
    active_zones = st.session_state.zones

    # Guard: no zones defined
    if not active_zones:
        status_ph.markdown(
            '<div class="status-pill pill-warn">⚠ &nbsp; ARM FAILED — NO ZONES DEFINED</div>',
            unsafe_allow_html=True
        )
        feed_ph.markdown("""
        <div style="background:#0a0c0f;border:1px solid #f59e0b55;border-radius:4px;height:300px;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.8rem;">
            <div style="font-size:2.5rem;opacity:0.25">🗺</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.66rem;color:#f59e0b;
                        letter-spacing:0.2em;text-align:center;">
                DEFINE AT LEAST ONE ZONE<br>USING THE EDITOR ON THE LEFT
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        # All zones loaded — start camera loop
        status_ph.markdown(
            f'<div class="status-pill pill-active">◉ &nbsp; ARMED · {len(active_zones)} ZONE{"S" if len(active_zones)>1 else ""} ACTIVE</div>',
            unsafe_allow_html=True
        )
        diag_ph.markdown(f"""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#64748b;line-height:2;letter-spacing:0.08em;">
            <div>▸ CAMERA MODULE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#10b981">ACTIVE</span></div>
            <div>▸ FACE ENGINE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#10b981">RUNNING</span></div>
            <div>▸ GEOFENCE ZONES &nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#10b981">{len(active_zones)} ACTIVE</span></div>
            <div>▸ DATABASE CONN &nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#10b981">CONNECTED</span></div>
            <div>▸ EMAIL ALERTS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#10b981">ENABLED</span></div>
        </div>""", unsafe_allow_html=True)

        load_dotenv()
        SENDER   = os.getenv("SENDER_EMAIL")
        PASSWORD = os.getenv("SENDER_PASSWORD")
        RECEIVER = os.getenv("RECEIVER_EMAIL")

        db_conn  = init_db()
        known_face_encodings, known_face_names = load_known_faces()
        video    = cv2.VideoCapture(0)
        log_dir  = "logs"
        os.makedirs(log_dir, exist_ok=True)

        logged_workers    = {}
        blink_counters    = {}
        liveness_verified = {}
        last_log_time     = 0
        last_email_time   = 0
        log_cooldown      = 10
        email_cooldown    = 60
        EAR_THRESHOLD     = 0.26
        EAR_CONSEC_FRAMES = 1
        area_was_clear    = True
        process_this_frame = True

        while run_system:
            ret, frame = video.read()
            if not ret:
                break

            frame       = cv2.resize(frame, (640, 480))
            clean_frame = frame.copy()

            # Draw all zones
            frame = draw_all_zones(frame, active_zones)

            # HUD corners + timestamp
            for (x, y, dx, dy) in [(10,10,1,1),(630,10,-1,1),(10,470,1,-1),(630,470,-1,-1)]:
                cv2.line(frame,(x,y),(x+dx*20,y),(245,158,11),2)
                cv2.line(frame,(x,y),(x,y+dy*20),(245,158,11),2)
            cv2.putText(frame, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                        (10,470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,116,139), 1, cv2.LINE_AA)
            cv2.putText(frame, "SITEGUARD CAM-01",
                        (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245,158,11), 1, cv2.LINE_AA)

            small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            if process_this_frame:
                rgb_small            = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations       = face_recognition.face_locations(rgb_small)
                face_encodings_list  = face_recognition.face_encodings(rgb_small, face_locations)
                face_landmarks_list  = face_recognition.face_landmarks(rgb_small, face_locations)
            process_this_frame = not process_this_frame

            current_frame_names = []
            intruder_in_frame   = False
            breach_zones        = set()

            for (top, right, bottom, left), face_enc, face_lm in zip(
                    face_locations, face_encodings_list, face_landmarks_list):
                top*=4; right*=4; bottom*=4; left*=4
                cx = left + (right-left)//2
                cy = top  + (bottom-top)//2

                in_zone, zone_name = point_in_any_zone(cx, cy, active_zones)
                if not in_zone:
                    continue

                name, box_color = "UNKNOWN", (239, 68, 68)
                is_authorized   = False

                if known_face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_enc, 0.5)
                    if True in matches:
                        worker = known_face_names[matches.index(True)]
                        current_frame_names.append(worker)

                        ear = (get_ear(face_lm['left_eye']) + get_ear(face_lm['right_eye'])) / 2.0
                        if ear < EAR_THRESHOLD:
                            blink_counters[worker] = blink_counters.get(worker, 0) + 1
                        elif blink_counters.get(worker, 0) >= EAR_CONSEC_FRAMES:
                            liveness_verified[worker] = True
                            blink_counters[worker] = 0

                        if liveness_verified.get(worker, False):
                            name, box_color = worker, (16, 185, 129)
                            is_authorized   = True
                            cur_t = time.time()
                            log_key = f"{worker}_{zone_name}"
                            if log_key not in logged_workers or cur_t - logged_workers[log_key] > 60:
                                now = datetime.now()
                                db_conn.cursor().execute(
                                    'INSERT INTO attendance (worker_name,zone_name,entry_date,entry_time) VALUES(?,?,?,?)',
                                    (worker, zone_name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"))
                                )
                                db_conn.commit()
                                logged_workers[log_key] = cur_t
                        else:
                            name, box_color = "VERIFYING", (253, 224, 71)
                            is_authorized   = True

                if not is_authorized:
                    intruder_in_frame = True
                    breach_zones.add(zone_name)

                # Detection box with zone tag
                cv2.rectangle(frame, (left,top), (right,bottom), box_color, 2)
                cv2.rectangle(frame, (left,top-22), (right,top), box_color, -1)
                cv2.putText(frame, f"{name.upper()} | {zone_name}",
                            (left+4, top-6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 1, cv2.LINE_AA)

            # Cleanup liveness for faces no longer visible
            for n in [k for k in list(liveness_verified) if k not in current_frame_names]:
                liveness_verified.pop(n, None)
                blink_counters.pop(n, None)

            cur_t = time.time()
            if intruder_in_frame:
                for y_s in range(0, 480, 6):
                    cv2.line(frame, (0,y_s), (640,y_s), (239,68,68), 1)
                frame = cv2.addWeighted(frame, 0.85, np.full_like(frame,(30,0,0)), 0.15, 0)
                cv2.putText(frame, "BREACH: " + " | ".join(breach_zones),
                            (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (239,68,68), 2, cv2.LINE_AA)

                try:
                    import winsound
                    winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_ASYNC)
                except Exception:
                    pass

                bz_name = ", ".join(breach_zones)
                if area_was_clear or (cur_t - last_email_time) > email_cooldown:
                    fp = os.path.join(log_dir, f"intruder_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
                    cv2.imwrite(fp, cv2.cvtColor(clean_frame, cv2.COLOR_RGB2BGR))
                    if SENDER and PASSWORD and RECEIVER:
                        threading.Thread(
                            target=send_alert_email,
                            args=(fp, datetime.now().strftime('%H:%M:%S'), bz_name, SENDER, PASSWORD, RECEIVER)
                        ).start()
                    last_email_time = cur_t
                    last_log_time   = cur_t
                    area_was_clear  = False
                elif (cur_t - last_log_time) > log_cooldown:
                    fp = os.path.join(log_dir, f"intruder_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
                    cv2.imwrite(fp, cv2.cvtColor(clean_frame, cv2.COLOR_RGB2BGR))
                    last_log_time = cur_t

                alert_ph.markdown(
                    f'<div class="alert-banner">⚠ BREACH IN: {bz_name.upper()} · ALERT DISPATCHED</div>',
                    unsafe_allow_html=True
                )
            else:
                area_was_clear = True
                alert_ph.empty()

            feed_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width='stretch')

            try:
                df = pd.read_sql_query(
                    "SELECT worker_name AS 'WORKER', zone_name AS 'ZONE', entry_time AS 'TIME' "
                    "FROM attendance ORDER BY id DESC LIMIT 12", db_conn
                )
                log_ph.dataframe(df, width='stretch', hide_index=True)
            except Exception:
                pass

        video.release()
        db_conn.close()