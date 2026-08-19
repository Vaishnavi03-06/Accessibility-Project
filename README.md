# Accessibility-Project — Real-Time Lecture to ISL Translator

> Converts live lecture audio into **Indian Sign Language (ISL)** using a modular pipeline of speech recognition, linguistic transformation, and avatar-based sign rendering, making academic lectures accessible to Deaf and hard-of-hearing students.

---

## Pipeline Overview

```
Lecture Video (MP4)
       │
       ▼
  FFmpeg (audio extraction)
       │
       ▼
  Whisper ASR  ──────────────────────► segments.json
  (speech → timestamped text)          (text + timestamps)
       │
       ▼
  Gloss Converter ───────────────────► gloss_output.json
  (English → ISL gloss)                (OSV reordered gloss)
  Rule-based + ISL grammar rules
       │
       ▼
  Motion Mapper ─────────────────────► motion_sequence.json
  (gloss → hand shape descriptors)     (hand shape + movement + location)
       │
       ▼
  Avatar Renderer (Three.js)
  (3D browser avatar with finger bone animation)
       │
       ▼
  [Phase 5] MediaPipe Pose + Hands Extraction
  ISLRTC videos → extracted_signs.json
  (real arm angles + finger values → avatar_3d.html patches)
```

Each stage communicates only through JSON — **loosely coupled**, so any component can be upgraded independently.

---

## Features

- **Real-time capable** : 3–5 second latency window per segment
- **ISL grammar rules** : OSV reordering, auxiliary verb dropping, negation and WH-word placement, function word removal
- **Motion library** : 200+ ISL signs mapped to hand shape descriptors
- **Phase 5 — ISLRTC pose extraction** : MediaPipe Pose + HandLandmarker extracts real arm angles and finger extension values directly from ISLRTC YouTube dictionary videos; avatar mirrors the actual signer's geometry
- **Auto URL discovery** : `scrape_islrtc.py` scrapes the ISLRTC dictionary website to find YouTube URLs for all lecture vocabulary automatically
- **3D browser avatar** : Three.js avatar with finger bone animation and extracted ISL pose angles; no installation needed
- **No-cache dev server** : `serve.py` disables browser caching so changes appear instantly
- **Modular architecture** : swap ASR, NLP, or avatar layer independently

---

## Project Structure

```
Accessibility-Project/
├── asr/
│   ├── asr_isl.py              # Whisper ASR — audio → segments.json
│   ├── audio.wav               # Extracted lecture audio
│   └── lecture.mp4             # Source lecture video
│
├── nlp/
│   └── gloss_converter.py      # English → ISL gloss converter
│
├── motion/
│   └── motion_mapper.py        # Gloss → motion descriptor mapper
│
├── avatar/
│   ├── avatar_integration.py   # Blender Python API integration
│   └── avatar_3d.html          # Browser 3D avatar (Three.js)
│
├── pose/
│   ├── mediapipe_extractor.py  # Downloads ISLRTC videos, extracts pose + hand landmarks
│   ├── update_motion_library.py# Patches motion_mapper.py from extracted_signs.json
│   ├── update_avatar.py        # Patches avatar_3d.html SIGNS dict with real angles
│   ├── scrape_islrtc.py        # Scrapes ISLRTC dictionary website for YouTube URLs
│   ├── update_islrtc_signs.py  # Patches ISLRTC_SIGNS dict in mediapipe_extractor.py
│   ├── visualise_extraction.py # Live MediaPipe overlay viewer
│   ├── extracted_signs.json    # Verified ISL arm angles + finger positions
│   └── videos/                 # Downloaded ISLRTC sign videos (auto-managed)
│
├── serve.py                    # No-cache HTTP server (avoids stale browser cache)
├── Makefile                    # Full pipeline automation
├── demo_viewer.html            # Full pipeline demo viewer
└── README.md
```

---

## Setup

### Requirements

```bash
Python 3.9+
pip install openai-whisper librosa numpy mediapipe opencv-python yt-dlp requests beautifulsoup4
```

FFmpeg must be installed separately:
- Mac: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`
- Windows: https://ffmpeg.org/download.html

---

## Makefile Commands

```bash
make audio      # Phase 1: extract audio from lecture.mp4
make asr        # Phase 2: transcribe with Whisper
make gloss      # Phase 3: convert to ISL gloss
make motion     # Phase 4: map to motion descriptors
make scrape     # Phase 5a: scrape ISLRTC website for sign URLs (run once)
make pose       # Phase 5b: extract ISL poses from ISLRTC videos (run once)
make serve      # Start no-cache local server → http://localhost:8000
make status     # Show which output files exist
make clean      # Remove lecture files (preserves ISL videos & pose data)
make clean-all  # Remove everything including ISL videos (full reset)
```

### Normal workflow (first time)

```bash
make audio && make asr && make gloss && make motion
make scrape   # finds ISLRTC video URLs for all lecture vocabulary
make pose     # downloads & extracts sign poses (~10–15 min, one-time)
make serve
```

### After changing lecture.mp4

```bash
make clean && make audio && make asr && make gloss && make motion && make serve
# make pose not needed — ISL videos and extracted poses are preserved
```

---

## How It Works

### 1. ASR — `asr/asr_isl.py`
Uses **OpenAI Whisper** (base model) to transcribe lecture audio into timestamped segments. Audio is resampled to 16 kHz mono using librosa before inference. Filler words (uh, um, like) are stripped and output is saved as structured JSON.

**Evaluation metric:** Word Error Rate (WER)

### 2. Gloss Conversion — `nlp/gloss_converter.py`
Converts cleaned English text to ISL gloss notation using rule-based linguistic transformation:

| Rule | Example |
|------|---------|
| OSV reordering | "I explain neural networks" → `NEURAL NETWORKS I EXPLAIN` |
| Drop aux verbs | "am, is, are, will, would" → dropped |
| Drop articles & function words | "the, an, a" → dropped |
| NOT to end | "I do not know" → `KNOW I NOT` |
| WH-word to end | "Where are you going?" → `YOU GO WHERE` |

**Evaluation metric:** BLEU score against human-annotated ISL gloss

### 3. Motion Mapping — `motion/motion_mapper.py`
Maps each gloss token to a structured motion descriptor:

```json
{
  "token": "KNOW",
  "hand_shape": "a_hand",
  "movement": "tap_temple",
  "location": "temple",
  "duration": 0.6,
  "non_manual": "neutral"
}
```

Library covers 200+ ISL signs. Unknown tokens fall back to fingerspelling.

### 4. Avatar Rendering — `avatar/avatar_3d.html`
Browser-based 3D avatar (Three.js / WebGL) with articulated finger bones. Runs in any browser with no installation. Load `nlp/gloss_output.json` when prompted to drive the animation.

Each sign entry stores real arm rotation angles extracted from ISLRTC pose data:

```javascript
HELLO: [[1,1,1,1,1], 'custom', [.5,.5,.5,.5,.5], 'neutral', 0.1,
        [-1.539, -0.344, -0.932, -0.021, 0.0],   // right arm angles
        [-0.12,   0.05, -0.88, -0.01,  0.0]]      // left arm angles
```

### 5. Phase 5 — ISLRTC Pose Extraction — `pose/`

The key innovation: instead of hand-crafting sign poses, the system **learns from real ISLRTC videos** automatically.

```
scrape_islrtc.py
  → scans divyangjan.depwd.gov.in/islrtc A–Z letter pages
  → finds YouTube video URLs for each lecture vocabulary word
  → saves to pose/islrtc_urls.json

mediapipe_extractor.py
  → downloads ISLRTC YouTube videos via yt-dlp
  → runs MediaPipe PoseLandmarker on every frame
      → shoulder / elbow / wrist 3D positions → arm rotation angles
  → runs MediaPipe HandLandmarker on every frame
      → 21 hand landmarks → finger extension values (0–1)
      → palm normal vector → palm orientation (hand_y)
  → uses peak-elevation frames for canonical sign pose
  → saves to pose/extracted_signs.json

update_avatar.py
  → reads extracted_signs.json
  → patches SIGNS dict in avatar_3d.html with real angles
  → fallback for signs with near-zero extraction (e.g. I → self pose)
```

**Coordinate conversion** — MediaPipe world coordinates to Three.js bone rotations:
- `upperArm_x = −arccos(u_y / |u|)` (elevation from straight-down)
- `upperArm_z = s · −|lateral / body_width|` (arm abduction)
- `foreArm_x = −arccos(dot(upper, fore) / |upper||fore|)` (elbow bend)
- `hand_z` from palm normal cross product (wrist rotation)

**Sign accuracy disclaimer** : Extracted angles are geometrically derived from 2D video and may differ from hand-crafted reference signs. Verified signs show directionally correct arm positions. Full accuracy requires stereo camera data or a dedicated ISL motion capture corpus.

---

## Limitations & Future Work

| Limitation | Planned Fix |
|------------|-------------|
| Rule-based gloss conversion | Replace with Seq2Seq transformer trained on English–ISL corpus |
| Whisper struggles with Indian accents | Fine-tune on Indian English academic speech / integrate Bhashini ASR |
| 2D-to-3D angle estimation error | Stereo cameras or depth sensors for exact pose extraction |
| No non-manual marker animation | Add facial expression blend shapes to avatar |
| Single speaker | Multi-speaker diarization |
| ~150 signs extracted | Expand ISLRTC_SIGNS dict with more vocabulary URLs |

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Audio extraction | FFmpeg |
| Speech recognition | OpenAI Whisper |
| Audio processing | librosa |
| NLP / gloss conversion | Python, rule-based |
| Pose extraction | MediaPipe PoseLandmarker + HandLandmarker |
| Video download | yt-dlp |
| Dictionary scraping | requests + BeautifulSoup4 |
| 3D avatar | Three.js (WebGL) |
| Blender integration | Blender Python API |

---

## Supervisor

Built under **Prof. PVM Rao**, Department of Design & Department of Mechanical Engineering, IIT Delhi.

---

## Authors

**Vaishnavi Rai**  
B.Tech CSE, IIT Delhi (4th year)  
[cs1230657@iitd.ac.in](mailto:cs1230657@iitd.ac.in)

**Avani Komalkar**  
B.Tech CSE, IIT Delhi (4th year)  
[cs5230325@iitd.ac.in](mailto:cs5230325@iitd.ac.in)

**Sanika Brahmankar**  
B.Tech CSE, IIT Delhi (4th year)  
[cs5230730@iitd.ac.in](mailto:cs5230730@iitd.ac.in)
