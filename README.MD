<div align="center">

# 🛡️ SiteGuard OS
### Smart Construction Site Security Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge)

*Real-time facial recognition · Geofence enforcement · Anti-spoofing liveness detection · Automated SQL attendance · Email breach alerts*

---

</div>

## 📸 Preview

> **SiteGuard OS** is a production-grade security dashboard built for construction sites. It monitors a live camera feed, enforces a virtual restricted zone, identifies registered workers via facial recognition with blink-based liveness verification, logs attendance directly to a SQLite database, and dispatches instant email alerts with snapshot attachments when an intruder is detected.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎥 **Live Camera Feed** | Real-time HUD overlay with zone boundaries, corner brackets, and timestamp watermark |
| 🧠 **Facial Recognition** | Matches faces against enrolled workers using `face_recognition` (dlib under the hood) |
| 👁️ **Liveness Detection** | Eye Aspect Ratio (EAR) blink detection prevents photo/video spoofing attacks |
| 📐 **Geofencing** | Polygon-based restricted zone — only faces *inside* the zone are evaluated |
| 🗄️ **SQL Attendance Log** | Every verified entry is written to SQLite and displayed live in the dashboard |
| 📧 **Email Breach Alerts** | Automatic email with intruder snapshot sent via Gmail SMTP on unauthorized access |
| 🔴 **Intruder Scanlines** | Red scanline + tint overlay and on-screen text when a breach is active |
| 🖥️ **Industrial HUD UI** | Dark amber-accented Streamlit dashboard with metric tiles, status pills, and diagnostic panel |

---

## 🗂️ Project Structure

```
INTRUSION_DETECTION_SYSTEM/
│
├── app.py                        # ← Main Streamlit application (rename from dashboard.py)
├── construction_site_security.py # Core detection logic (standalone script)
├── intruder_detector.py          # Intruder detection module
├── dashboard.py                  # Original dashboard (superseded by app.py)
│
├── registered_workers/           # Worker face images for enrollment
│   └── Shubham_kumar.jpg         #   Filename (without ext) = worker's display name
│
├── known_faces/                  # Additional face reference directory
├── demo_media/
│   └── demo_video.mp4            # Demo video for offline testing
│
├── logs/                         # Auto-created: intruder snapshots (.jpg)
├── security_data.db              # Auto-created: SQLite attendance database
├── attendance_log.csv            # Legacy CSV attendance log
│
├── .streamlit/
│   └── config.toml               # Streamlit theme & server config
│
├── .env                          # 🔒 Secret credentials (never commit!)
├── .env.example                  # Template for .env setup
├── .gitignore                    # Excludes secrets, DB, logs, venv
├── requirements.txt              # Python dependencies
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- **Python 3.10+** (tested on 3.10)
- **Windows** recommended (uses `winsound` for alerts; Linux/macOS will skip audio gracefully)
- A webcam connected to the machine
- A Gmail account with an **App Password** enabled

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/siteguard-os.git
cd siteguard-os
```

### Step 2 — Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dlib (required before face_recognition)

`face_recognition` depends on `dlib`, which must be installed from a pre-built wheel on Windows:

```bash
# A compatible wheel is included in the repo root
pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
```

> For other Python versions or Linux/macOS, see the [dlib installation guide](https://github.com/davisking/dlib).

### Step 4 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Configure environment variables

```bash
# Copy the template
cp .env.example .env
```

Open `.env` and fill in your Gmail credentials:

```env
SENDER_EMAIL=your_gmail@gmail.com
SENDER_PASSWORD=xxxx_xxxx_xxxx_xxxx   # Gmail App Password
RECEIVER_EMAIL=security@yoursite.com
```

> **How to generate a Gmail App Password:**  
> Google Account → Security → 2-Step Verification → App Passwords → Generate

---

## 👷 Enrolling Workers

Place a clear, front-facing `.jpg` or `.png` photo of each worker inside the `registered_workers/` folder.

**The filename becomes the worker's display name on screen:**

```
registered_workers/
├── Shubham_kumar.jpg     →  "Shubham Kumar"
├── Rahul_Sharma.jpg      →  "Rahul Sharma"
└── Priya_Singh.png       →  "Priya Singh"
```

- Use underscores for spaces — they are auto-converted
- One face per image, well-lit, no sunglasses
- Restart the app after adding new workers (faces are cached with `@st.cache_resource`)

---

## 🚀 Running the App

```bash
streamlit run app.py
```

The dashboard opens at **http://localhost:8501**

1. Check the **☑ ARM SECURITY SYSTEM** toggle
2. Point your webcam at the restricted zone
3. Walk registered workers through — they must **blink** to pass liveness verification
4. Unauthorized faces trigger the red breach overlay + email alert

---

## 🧠 How It Works

```
Camera Frame
    │
    ▼
Resize to 640×480  →  draw HUD overlays
    │
    ▼
Downsample 0.25×  →  detect face locations + encodings + landmarks
    │
    ├─ Face center OUTSIDE restricted zone?  →  ignore
    │
    └─ Face center INSIDE restricted zone?
           │
           ├─ Match against known_encodings (tolerance 0.5)?
           │       │
           │       ├─ YES → EAR blink check (liveness)
           │       │           ├─ Blink confirmed  →  GREEN box + DB log
           │       │           └─ Awaiting blink   →  YELLOW "VERIFYING" box
           │       │
           │       └─ NO  →  RED box + INTRUDER flag
           │
           └─ INTRUDER flag set?
                   ├─ Scanline overlay + breach text
                   ├─ winsound alert (Windows)
                   └─ Save snapshot + send email (throttled)
```

### Liveness Detection (Anti-Spoofing)

The system uses the **Eye Aspect Ratio (EAR)** formula:

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
```

When EAR drops below `0.26` for at least 1 consecutive frame, a blink is registered. A worker is only marked **Authorized** after a confirmed blink — preventing photo or screen-based spoofing.

### Database Schema

```sql
CREATE TABLE attendance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_name TEXT    NOT NULL,
    entry_date  TEXT    NOT NULL,   -- YYYY-MM-DD
    entry_time  TEXT    NOT NULL    -- HH:MM:SS
);
```

---

## 🔧 Configuration

Key constants in `app.py` you can tune:

| Constant | Default | Description |
|---|---|---|
| `EAR_THRESHOLD` | `0.26` | EAR value below which a blink is detected |
| `EAR_CONSEC_FRAMES` | `1` | Frames below threshold required to register blink |
| `log_cooldown` | `10` sec | Minimum gap between intruder snapshot saves |
| `email_cooldown` | `60` sec | Minimum gap between breach email dispatches |
| `restricted_zone` | `[[150,100],[500,100],[550,400],[100,400]]` | Polygon vertices of the restricted zone |
| Face tolerance | `0.5` | `compare_faces` distance threshold (lower = stricter) |

---

## 🔒 Security & Privacy Notes

- **Never commit `.env`** — it's in `.gitignore` by default
- Worker face images are biometric data — keep `registered_workers/` private; it's gitignored
- The `logs/` folder contains intruder snapshots — gitignored
- `security_data.db` is gitignored (runtime data)
- For production deployments, consider encrypting the SQLite database

---

## 🛣️ Roadmap

- [ ] Multi-camera support (multiple `cv2.VideoCapture` streams)
- [ ] Streamlit Cloud deployment with webcam API fallback
- [ ] Worker registration UI (upload face image via the dashboard)
- [ ] Export attendance to Excel / CSV from the UI
- [ ] Telegram bot alert integration
- [ ] Night vision / low-light enhancement
- [ ] Docker containerization

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web dashboard framework |
| `opencv-python` | Camera capture & frame processing |
| `face_recognition` | Face detection, encoding & matching |
| `dlib` | Underlying ML engine for face_recognition |
| `numpy` | Array operations & EAR geometry |
| `Pillow` | Image loading for worker enrollment |
| `pandas` | Attendance dataframe display |
| `python-dotenv` | `.env` file loading |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/multi-camera`
3. Commit your changes: `git commit -m "feat: add multi-camera support"`
4. Push to the branch: `git push origin feature/multi-camera`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with 🧠 face_recognition · 👁️ dlib · 🎥 OpenCV · 🖥️ Streamlit

**SiteGuard OS** — *Keeping construction sites safe, one blink at a time.*

</div>