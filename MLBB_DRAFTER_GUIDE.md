# MLBB Drafter - Complete Project Guide

Real-time, non-invasive Mobile Legends: Bang Bang drafting assistant.

## What Is This

A computer-vision-powered tool that watches your MLBB draft screen, detects hero picks/bans via YOLOv8n-OBB, runs a lightweight GCN model for recommendations, and displays results on a web dashboard or native Android overlay.

**Zero game memory access.** Read-only screen capture via ADB.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   PC (Server)                       │
│                                                     │
│  ┌──────────┐   ┌──────────┐   ┌────────────────┐  │
│  │ ADB      │──▶│ YOLOv8n  │──▶│ GCN Model v2   │  │
│  │ Capture  │   │ Detector │   │ 132 heroes     │  │
│  │ (read)   │   │ (roles)  │   │ Counter boost  │  │
│  └──────────┘   └──────────┘   └───────┬────────┘  │
│                                        │            │
│                                ┌───────▼────────┐   │
│                                │  WebSocket     │   │
│                                │  Server        │   │
│                                └───────┬────────┘   │
└────────────────────────────────────────┼────────────┘
                                         │
                    ┌────────────────────┼────────────┐
                    ▼                    ▼            ▼
              ┌──────────┐      ┌──────────────┐  ┌────────┐
              │ Web      │      │ Android      │  │ Mobile │
              │ Dashboard│      │ Overlay App  │  │ Client │
              │ (HTML)   │      │ (Kotlin)     │  │ (WS)   │
              └──────────┘      └──────────────┘  └────────┘
```

**Pipeline:** Capture → Detect → Track → Recommend → Broadcast → Display

## Tech Stack

| Layer | Tech | Why |
|-------|------|-----|
| Screen capture | `adb exec-out screencap -p` | No phone app needed, read-only |
| Detection | YOLOv8n-OBB (ultralytics) | Real-time, oriented bounding boxes |
| Recommendation | PyTorch GCN (132 heroes) | Graph structure fits hero synergies |
| Communication | WebSocket (persistent) | Bidirectional, low-latency |
| Dashboard | HTML + CSS + JS | Universal, no install |
| Data | 37K+ tournament drafts | Real esports + ranked matches |

## Project Structure

```
mlbb_drafter/
├── server/                    # Python backend
│   ├── main.py               # Entry point, async capture loop
│   ├── config.py             # All configuration (env vars)
│   ├── capture/
│   │   └── adb_capture.py    # ADB screen capture (USB + wireless)
│   ├── detection/
│   │   ├── yolo_detector.py  # YOLOv8n-OBB hero detection
│   │   └── dummy_detector.py # Random detection for testing
│   ├── recommendation/
│   │   ├── gcn_model_v2.py   # Enhanced GCN (132 heroes, counter boost)
│   │   ├── gcn_model.py      # Original GCN (legacy, 30 heroes)
│   │   └── draft_state.py    # Draft state tracker
│   ├── data/
│   │   ├── loader.py         # Hero metadata loader
│   │   ├── tournament_loader.py # Tournament data loader
│   │   └── scraper.py        # Liquipedia async scraper
│   └── websocket/
│       └── server.py         # WebSocket broadcast server
├── shared/
│   ├── hero_meta.json        # 132 heroes metadata
│   └── constants.py          # Shared constants
├── web_dashboard/
│   ├── index.html            # Dashboard UI
│   ├── app.js                # WebSocket client
│   └── style.css             # Dashboard styles
├── training/
│   ├── train_gcn_enhanced.py # Main training script (uses ALL data)
│   ├── train_gcn.py          # Old training (deprecated)
│   ├── train_yolo.py         # YOLO training for detection
│   ├── fetch_all_data.py     # Fetch from all APIs (merge mode)
│   ├── fetch_hero_stats_api.py # Hero stats fetcher
│   ├── fetch_openmlbb_full.py # Comprehensive OpenMLBB fetcher
│   ├── fetch_mlbb_io.py      # mlbb.io ranked stats fetcher
│   ├── fetch_kaggle_datasets.py # Kaggle dataset downloader
│   ├── fetch_liquipedia.py   # Liquipedia tournament scraper
│   ├── process_drive_data.py # Process tournament data
│   ├── process_api_data.py   # Process API data
│   ├── add_temporal_features.py # Add time-aware features
│   ├── create_final_training_data.py # Consolidate all data
│   ├── backup_data.py        # Backup before updates
│   ├── generate_synthetic.py # Synthetic draft generation
│   ├── generate_images.py    # Synthetic image generation
│   ├── capture_screenshots.py # Real screenshot capture
│   └── data/                 # All training data
│       ├── gcn_model_v2.pt   # Trained model (132 heroes, 18.7K params)
│       ├── tournament_drafts.json # 37,373 tournament + ranked matches
│       ├── adjacency_real.json # 132x132 hero relationships
│       ├── hero_stats_temporal.csv # Temporal features
│       ├── hero_cooccurrence_tournament.json # Synergy stats
│       ├── synergy_matrix_tournament.json # Tournament synergy
│       ├── era_hero_stats.json # Per-patch hero stats (31 eras)
│       ├── patch_timeline.json # 54 patches
│       └── api_data/         # Raw API data (17 files)
├── tests/                    # Unit & integration tests
├── docs/                     # Documentation
├── drive_data/               # Tournament data from Google Drive
├── runs/                     # YOLO training outputs
├── pyproject.toml            # Python project config
├── requirements.txt          # Python dependencies
├── README.md                 # Main documentation
└── MLBB_DRAFTER_GUIDE.md     # This file
```

## Quick Start

### 1. Environment Setup

```bash
cd mlbb_drafter
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run with Dummy Detector (No YOLO needed)

```bash
# Start server with random detection for testing
python -m server.main

# Open dashboard in a new terminal
python -m http.server 8080 --directory web_dashboard
```

### 3. Run with YOLO Detector

```bash
# Train YOLO first (see Training section below)
# Then:
DETECTOR_TYPE=yolo YOLO_MODEL_PATH=runs/train/mlbb_hero_detect/weights/best.pt \
  python -m server.main
```

### 4. Wireless ADB (No USB Cable)

```bash
# On phone: Enable Developer Options > Wireless Debugging
# Get pairing code from phone settings

# Pair first (one-time)
adb pair 192.168.1.50:37913
# Enter pairing code when prompted

# Connect
adb connect 192.168.1.50:5555

# Run server with device
ADB_DEVICE=192.168.1.50:5555 DETECTOR_TYPE=yolo python -m server.main
```

### 5. Run Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

## Configuration

All config is in `server/config.py` via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ADB_HOST` | `localhost` | ADB server host |
| `ADB_PORT` | `5037` | ADB server port |
| `ADB_DEVICE` | `""` | Device serial or `IP:port` for wireless |
| `WS_HOST` | `0.0.0.0` | WebSocket bind address |
| `WS_PORT` | `8765` | WebSocket port |
| `CAPTURE_FPS` | `5` | Screen capture frame rate |
| `DETECTOR_TYPE` | `dummy` | `dummy` or `yolo` |
| `YOLO_MODEL_PATH` | `runs/.../best.pt` | Trained YOLO model path |
| `YOLO_CONFIDENCE` | `0.5` | Detection confidence threshold |
| `YOLO_DEVICE` | `cpu` | `cpu` or `cuda:0` |
| `DASHBOARD_PORT` | `8080` | HTTP dashboard port |

## GCN Model (MOBARec-GCNFP v2)

Enhanced Graph Convolutional Network for hero recommendation.

### Architecture

```
Hero Features (10 dims) → Linear(10, 256) → GCN(256, 256) → ReLU → Linear(256, 1) → Sigmoid
```

- **Heroes:** 132 (full MLBB roster)
- **Features per hero:** 10
  - win_rate, pick_rate, ban_rate
  - meta_score, trend
  - weighted_win_rate, recency_score
  - total_games
  - synergy_win_rate, synergy_games
- **Graph:** 132x132 adjacency matrix (11,380 edges)
- **Parameters:** 18,744
- **Output:** Win probability [0, 1]

### How Recommendations Work

1. Build hero embedding matrix: `hero_features(132, 10)`
2. Apply GCN with adjacency matrix: `gcn_out = ReLU(A · H · W + b)`
3. Dynamic match embedding: `em0 = Σ(ally_heroes) - Σ(enemy_heroes)`
4. Score each available hero via projection layers
5. Counter boost: +15% for counter picks
6. Return top-k by score

### Training

```bash
# Train GCN on all data (recommended)
python training/train_gcn_enhanced.py --epochs 500

# With temporal weighting
python training/train_gcn_enhanced.py --epochs 500 --min-year 2021 --temporal-decay 0.5

# Output: training/data/gcn_model_v2.pt
```

### Adjacency Matrix

Built from hero co-occurrence in draft data:
- 132x132 matrix (132 heroes)
- Non-zero = heroes picked together in same draft
- Saved as `training/data/adjacency_real.json`

## YOLO Detection

### Model: YOLOv8n-OBB

- 3,083,685 parameters (pretrained on COCO)
- Oriented Bounding Box (OBB) for rotated hero icons
- 6 classes: fighter, assassin, mage, tank, support, marksman

### Label Format

YOLO OBB format (one line per detection):
```
class_index x1 y1 x2 y2 x3 y3 x4 y4
```
- 4 corner points (clockwise from top-left)
- Coordinates normalized to [0, 1]

### Position-Based Categorization

```
┌────────────┬──────────────┬────────────┐
│   ALLY     │    BANS      │   ENEMY    │
│  (left)    │  (center)    │  (right)   │
│  cx < mid  │ mid±10%      │  cx > mid  │
└────────────┴──────────────┴────────────┘
```

Left half → ally picks, right half → enemy picks, center → bans.

### Training

#### Step 1: Generate Synthetic Images

```bash
# Creates 5,000 images with hero icons at random positions
# 4,000 train / 1,000 val, augmented with rotation, noise, glow
python training/generate_images.py

# Output: training/data/synthetic_images/
```

#### Step 2: Train YOLO

```bash
# On GPU (recommended):
venv/bin/python -c "
from ultralytics import YOLO
YOLO('yolov8n-obb.pt').train(
    data='training/dataset.yaml',
    epochs=50,
    device=0,
    imgsz=640,
    batch=16
)"
# ~15 min on GPU

# On CPU (very slow, ~33 min/epoch):
venv/bin/python -c "
from ultralytics import YOLO
YOLO('yolov8n-obb.pt').train(
    data='training/dataset.yaml',
    epochs=5,
    device='cpu'
)"
```

#### Step 3: Capture Real Screenshots (Optional, for fine-tuning)

```bash
# Capture 100 real draft screenshots via wireless ADB
python training/capture_screenshots.py \
  --count 100 \
  --output training/data/real \
  --serial 192.168.1.50:5555

# Manually label with Roboflow or CVAT
# Retrain YOLO on real data to bridge domain gap
```

### Dataset Config (`training/dataset.yaml`)

```yaml
path: /home/alan/Documents/code/mlbb_drafter/training/data/synthetic_images
train: images/train
val: images/val
nc: 6
names:
  0: fighter
  1: assassin
  2: mage
  3: tank
  4: support
  5: marksman
```

## Data Pipeline

### Tournament Data

- Source: Google Drive tournament data (M1-M5, MPL, MDL)
- Processor: `training/process_drive_data.py`
- Output: `training/data/tournament_drafts.json` (37,373 matches)

### API Data

- **OpenMLBB:** Hero stats, counters, synergies (132 heroes)
- **Pren7/MLBB-Winrate:** Daily win/pick/ban rates (132 heroes)
- **p3hndrx:** Hero metadata (132 heroes)
- **Kaggle:** Ranked matches (10,664), M5/M7 tournament data

### Hero Metadata (`shared/hero_meta.json`)

```json
{
  "heroes": [
    {"id": 1, "name": "Miya", "real_name": "Miya, the Moonlight Archer", "role": "Marksman", "lanes": ["Gold"]},
    {"id": 2, "name": "Balmond", "real_name": "Balmond, the Dwarf Warrior", "role": "Fighter", "lanes": ["EXP", "Jungle"]},
    {"id": 3, "name": "Saber", "real_name": "Saber, the Triple Sweep", "role": "Assassin", "lanes": ["Jungle"]}
  ]
}
```

**Coverage:** 132 heroes with full metadata.

## Server Pipeline

### Frame Processing Flow

```
ADB capture → YOLO detect → Position categorize → Update draft state
    → GCN recommend → WebSocket broadcast → Dashboard update
```

### Key Optimizations

1. **Frame dropping:** If previous frame still processing, skip current frame
2. **Async ADB:** `run_in_executor()` prevents blocking the event loop
3. **Error handling:** Try/except around each frame with logging
4. **Configurable FPS:** `CAPTURE_FPS` env var (default 5)

### WebSocket Message Format

```json
{
  "type": "draft_update",
  "ally_picks": ["Lancelot", "Atlas"],
  "enemy_picks": ["Fanny"],
  "bans": ["Kagura"],
  "recommendations": {
    "top_picks": [
      {"hero": "Chou", "win_rate": 0.682},
      {"hero": "Granger", "win_rate": 0.651}
    ],
    "counter_picks": [
      {"hero": "Saber", "win_rate": 0.620}
    ],
    "synergy_picks": [
      {"hero": "Estes", "win_rate": 0.615}
    ]
  },
  "draft_complete": false
}
```

## Dashboard

### Running

```bash
# Option 1: Python HTTP server
cd web_dashboard && python -m http.server 8080

# Option 2: Node.js
npx serve web_dashboard -p 8080
```

Open `http://localhost:8080` in browser.

### Features

- Live draft updates via WebSocket
- Ally/Enemy/Ban hero lists
- Top 3 recommendations with win rates
- Counter picks (heroes that counter enemy team)
- Synergy picks (heroes that work well with your team)
- Strategy flags (Early Game, Late Game, Team Fight, etc.)
- Counter alerts
- Role lane assignments (EXP, Gold, Mid, Roam, Jungle)

## Anti-Ban Safety

### Hard Rules

1. **Never** access game memory or files
2. **Never** automate inputs (`input tap`, `input swipe`)
3. **Never** modify game data
4. Read-only `adb exec-out screencap -p` only
5. All ML runs on PC, not on device

### What This Tool Does

- Captures screenshots via ADB (read-only)
- Analyzes images with computer vision (YOLO)
- Provides recommendations via overlay dashboard
- Zero interaction with the game client

### What This Tool Never Does

- Reads/writes game files
- Sends input commands to the device
- Hooks into game processes
- Modifies game state in any way

## Testing

### Test Suite

```bash
pytest tests/ -v

# test_integration.py (14 tests)
#   - End-to-end pipeline, GCN forward pass, draft state
#
# test_recommendation.py (9 tests)
#   - MOBARecGCN init, forward pass, adjacency, recommend determinism
#
# test_detection.py (3 tests)
#   - DummyDetector init, detection
#
# test_yolo_detector.py (2 tests)
#   - YOLODetector init, empty detection
#
# test_capture.py (9 tests)
#   - ADBCapture init, buffer, methods, wireless ADB
#
# test_websocket.py (3 tests)
#   - WebSocketServer init, broadcast, start
#
# test_scraper.py (1 test)
#   - Liquipedia scraper init
#
# test_tournament_loader.py (4 tests)
#   - Tournament loader, CSV, co-occurrence, adjacency
```

### Adding Tests

```python
# tests/test_your_feature.py
import pytest

def test_feature():
    assert your_function() == expected
```

## Known Limitations

1. **YOLO trained on synthetic data** — Domain gap with real screenshots. Fine-tune on real data.
2. **No Android overlay yet** — Web dashboard only for now.
3. **No latency measurement** — WebSocket ping/pong not implemented.
4. **Liquipedia rate limited** — IP-level blocks from scraping.
5. **mlbb.io needs API key** — Free at parse.bot.

## Files Reference

| File | Purpose |
|------|---------|
| `server/main.py` | Entry point, async capture loop, pipeline orchestration |
| `server/config.py` | All configuration (env vars) |
| `server/capture/adb_capture.py` | ADB screen capture with wireless support |
| `server/detection/yolo_detector.py` | YOLOv8n-OBB hero detection |
| `server/detection/dummy_detector.py` | Random detection for testing |
| `server/recommendation/gcn_model_v2.py` | Enhanced GCN (132 heroes, counter boost) |
| `server/recommendation/gcn_model.py` | Original GCN (legacy, 30 heroes) |
| `server/recommendation/draft_state.py` | Draft state tracker |
| `server/websocket/server.py` | WebSocket broadcast server |
| `server/data/loader.py` | Hero metadata loader |
| `server/data/tournament_loader.py` | Tournament data + co-occurrence |
| `server/data/scraper.py` | Liquipedia async scraper |
| `shared/hero_meta.json` | 132 heroes metadata |
| `shared/constants.py` | Shared constants |
| `web_dashboard/index.html` | Dashboard UI |
| `web_dashboard/app.js` | WebSocket client |
| `web_dashboard/style.css` | Dashboard styles |
| `training/train_gcn_enhanced.py` | Main training script |
| `training/train_gcn.py` | Legacy training (deprecated) |
| `training/train_yolo.py` | YOLO training script |
| `training/fetch_all_data.py` | Master data fetcher |
| `training/fetch_hero_stats_api.py` | Hero stats fetcher |
| `training/fetch_openmlbb_full.py` | Comprehensive OpenMLBB fetcher |
| `training/fetch_mlbb_io.py` | mlbb.io ranked stats fetcher |
| `training/fetch_kaggle_datasets.py` | Kaggle dataset downloader |
| `training/fetch_liquipedia.py` | Liquipedia tournament scraper |
| `training/process_drive_data.py` | Tournament data processor |
| `training/process_api_data.py` | API data processor |
| `training/add_temporal_features.py` | Temporal features processor |
| `training/create_final_training_data.py` | Data consolidator |
| `training/generate_synthetic.py` | Synthetic draft generator |
| `training/generate_images.py` | Synthetic image generator |
| `training/capture_screenshots.py` | Real screenshot capture |
| `training/dataset.yaml` | YOLO dataset config |
| `tests/` | Unit & integration tests |
