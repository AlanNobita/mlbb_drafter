# MLBB Drafter - Complete Project Guide

Real-time, non-invasive Mobile Legends: Bang Bang drafting assistant.

## What Is This

A computer-vision-powered tool that watches your MLBB draft screen, detects hero picks/bans via YOLOv8n-OBB, runs a lightweight GCN model for recommendations, and displays results on a web dashboard or native Android overlay.

**Zero game memory access.** Read-only screen capture via ADB.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   PC (Server)                   │
│                                                 │
│  ┌──────────┐   ┌──────────┐   ┌────────────┐  │
│  │ ADB      │──▶│ YOLOv8n  │──▶│ GCN Model  │  │
│  │ Capture  │   │ Detector │   │ Recommend  │  │
│  │ (read)   │   │ (roles)  │   │ (top 3)    │  │
│  └──────────┘   └──────────┘   └─────┬──────┘  │
│                                      │          │
│                              ┌───────▼────────┐ │
│                              │  WebSocket     │ │
│                              │  Server        │ │
│                              └───────┬────────┘ │
└──────────────────────────────────────┼──────────┘
                                       │
                    ┌──────────────────┼────────────┐
                    ▼                  ▼            ▼
              ┌──────────┐    ┌──────────────┐  ┌────────┐
              │ Web      │    │ Android      │  │ Mobile │
              │ Dashboard│    │ Overlay App  │  │ Client │
              │ (HTML)   │    │ (Kotlin)     │  │ (WS)   │
              └──────────┘    └──────────────┘  └────────┘
```

## Tech Stack

| Layer | Tech | Why |
|-------|------|-----|
| Screen capture | `adb exec-out screencap -p` | No phone app needed, read-only |
| Detection | YOLOv8n-OBB (ultralytics) | Real-time, oriented bounding boxes |
| Recommendation | PyTorch GCN (single-layer) | Graph structure fits hero synergies |
| Communication | WebSocket (persistent) | Bidirectional, low-latency |
| Dashboard | HTML + CSS + JS | Universal, no install |
| Future overlay | Kotlin + SYSTEM_ALERT_WINDOW | Native Android floating window |

## Project Structure

```
mlbb_drafter/
├── server/                    # Python backend
│   ├── main.py               # Entry point, async capture loop
│   ├── config.py             # All configuration (env vars)
│   ├── capture/
│   │   ├── __init__.py
│   │   └── adb_capture.py    # ADB screen capture (USB + wireless)
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── yolo_detector.py  # YOLOv8n-OBB hero detection
│   │   └── dummy_detector.py # Random detection for testing
│   ├── recommendation/
│   │   ├── __init__.py
│   │   ├── gcn_model.py      # MOBARec-GCNFP model (47k params)
│   │   └── draft_state.py    # Draft state tracker
│   ├── websocket/
│   │   ├── __init__.py
│   │   └── server.py         # WebSocket broadcast server
│   └── data/
│       ├── __init__.py
│       ├── loader.py          # Hero metadata loader
│       ├── tournament_loader.py # Tournament data + co-occurrence
│       └── scraper.py         # Liquipedia async scraper
├── shared/
│   └── hero_meta.json         # 30-hero MVP metadata
├── web_dashboard/
│   ├── index.html             # Dashboard UI
│   ├── app.js                 # WebSocket client + UI updates
│   └── style.css              # Dashboard styles
├── training/
│   ├── generate_synthetic.py  # 10k synthetic draft records
│   ├── generate_images.py     # 5k synthetic hero icon images
│   ├── train_yolo.py          # YOLO training script
│   ├── train_gcn.py           # GCN training script
│   ├── capture_screenshots.py # Real screenshot capture via ADB
│   ├── dataset.yaml           # YOLO dataset config (6 classes)
│   └── data/
│       ├── synthetic_drafts.csv    # 10k draft records
│       ├── adjacency_matrix.pt     # 30x30 hero co-occurrence
│       ├── gcn_model_v2.pt         # Trained GCN weights
│       └── synthetic_images/       # 5k images (4k train / 1k val)
│           ├── images/train/
│           ├── images/val/
│           ├── labels/train/
│           └── labels/val/
├── tests/                     # 46 unit tests
├── pyproject.toml
├── requirements.txt
└── PRODUCTION_GUIDE.md
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
python server/main.py

# Open dashboard
# Serve web_dashboard/ on port 8080 (any HTTP server)
cd web_dashboard && python -m http.server 8080
```

### 3. Run with YOLO Detector

```bash
# Train YOLO first (see Training section below)
# Then:
DETECTOR_TYPE=yolo YOLO_MODEL_PATH=runs/train/mlbb_hero_detect/weights/best.pt \
  python server/main.py
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
ADB_DEVICE=192.168.1.50:5555 DETECTOR_TYPE=yolo python server/main.py
```

### 5. Run Tests

```bash
source venv/bin/activate
pytest tests/ -v
# 46 tests, ~4.6s
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

## GCN Model (MOBARec-GCNFP)

Single-layer Graph Convolutional Network for hero recommendation.

### Architecture

```
Hero IDs → Embedding(30, 128) → GCN(128→128) → ReLU → Linear(128→64) → Score
```

- **Parameters:** 47,070 (under 60k budget)
- **Inference time:** 0.12ms on CPU
- **Input:** Draft state (ally picks, enemy picks, bans)
- **Output:** Top-k hero recommendations with scores

### Dynamic Match Embedding

```
em0 = Σ(efriendly) - Σ(eopponent)
```

The model computes a match embedding from picked heroes, then scores available heroes based on the graph structure (co-occurrence adjacency matrix).

### How Recommendations Work

1. Build hero embedding matrix: `hero_embedding(30, 128)`
2. Apply GCN with adjacency matrix: `gcn_out = ReLU(A · H · W + b)`
3. Score each available hero via projection layers
4. Return top-k by score

### Training

```bash
# Train GCN on synthetic drafts (10k records)
python training/train_gcn.py

# Output: training/data/gcn_model_v2.pt
```

### Adjacency Matrix

Built from hero co-occurrence in draft data:
- 30x30 matrix (30 heroes in MVP)
- Non-zero = heroes picked together in same draft
- Saved as `training/data/adjacency_matrix.pt`

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

### Synthetic Draft Generation

```bash
python training/generate_synthetic.py
# Output: training/data/synthetic_drafts.csv (10k records)
# Columns: friendly_picks, enemy_picks, bans, win_rate
```

### Tournament Data

- Source: [LiTianYeoh/MLBB_Tournament_Analysis](https://github.com/LiTianYeoh/MLBB_Tournament_Analysis)
- Loader: `server/data/tournament_loader.py`
- Builds co-occurrence matrix for adjacency tensor

### Liquipedia Scraper

```bash
# Async scraper with rate limiting (2s delay)
# Fetches hero stats, win rates, synergies
# Entry: server/data/scraper.py
```

### Hero Metadata (`shared/hero_meta.json`)

```json
{
  "heroes": [
    {"id": 1, "name": "Lancelot", "role": "Assassin", "lanes": ["Jungle"]},
    {"id": 6, "name": "Chou", "role": "Fighter", "lanes": ["EXP", "Roam"]},
    {"id": 9, "name": "Kagura", "role": "Mage", "lanes": ["Mid"]},
    {"id": 16, "name": "Atlas", "role": "Tank", "lanes": ["Roam"]},
    {"id": 21, "name": "Angela", "role": "Support", "lanes": ["Roam"]},
    {"id": 26, "name": "Granger", "role": "Marksman", "lanes": ["Gold"]}
  ]
}
```

**MVP:** 30 heroes (5 assassins, 4 fighters, 4 mages, 5 tanks, 5 supports, 5 marksmen). Expand to 503+ via scraper.

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
    "counter_picks": [],
    "synergy_picks": []
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
- Analyzes images with computer vision
- Displays recommendations on a separate screen
- Zero interaction with the game client

### What This Tool Never Does

- Reads/writes game files
- Sends input commands to the device
- Hooks into game processes
- Modifies game state in any way

## Testing

### Test Suite (46 tests)

```bash
pytest tests/ -v

# test_capture.py (9 tests)
#   - ADBCapture init, buffer, methods
#   - Wireless ADB support (device_serial, connect, disconnect, is_connected)
#
# test_detection.py (3 tests)
#   - DummyDetector init, detection
#
# test_yolo_detector.py (2 tests)
#   - YOLODetector init, empty detection
#
# test_recommendation.py (9 tests)
#   - MOBARecGCN init, forward pass, adjacency, recommend determinism
#
# test_integration.py (14 tests)
#   - End-to-end pipeline, adjacency matrix, data pipeline
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

1. **30-hero MVP** — Not full 503 hero pool. Expand via scraper rerun.
2. **Synthetic training data** — Domain gap with real screenshots. Fine-tune on real data.
3. **YOLO trained on CPU** — Extremely slow (~33 min/epoch). Use GPU.
4. **No tournament dataset** — Needs download from GitHub source.
5. **No Android overlay yet** — Web dashboard only for now.
6. **No latency measurement** — WebSocket ping/pong not implemented.

## Roadmap

### Phase 1: MVP Pipeline (Current)
- [x] ADB capture (USB + wireless)
- [x] Dummy detector for testing
- [x] YOLO detector skeleton
- [x] GCN model (47k params)
- [x] WebSocket server
- [x] Web dashboard
- [x] Synthetic data generation
- [x] 46 unit tests

### Phase 2: Training
- [ ] Capture 100+ real screenshots
- [ ] Train YOLO on GPU
- [ ] Fine-tune on real data
- [ ] Download tournament dataset
- [ ] Train GCN on real data
- [ ] Measure Recall@1, NDCG@3

### Phase 3: Production
- [ ] Android overlay app (Kotlin)
- [ ] Latency benchmarking (200ms target)
- [ ] Memory profiling (<500MB target)
- [ ] Error recovery / reconnection
- [ ] Hero expansion to 503+

### Phase 4: Advanced
- [ ] Counter-pick logic (enemy-specific)
- [ ] Synergy detection (team composition)
- [ ] Meta-aware recommendations (patch updates)
- [ ] Multi-device support

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Full pipeline latency | <200ms | Not measured |
| GCN inference | <100ms | 0.12ms |
| Memory usage | <500MB | Not measured |
| Capture FPS | 5-10 | 5 (configurable) |
| GCN parameters | <60k | 47,070 |
| Dashboard updates | Real-time | WebSocket |

## FAQ

### Q: Will this get me banned?

**No.** The tool only reads your screen via ADB. It never interacts with the game client, never modifies files, never sends input commands. It's visually equivalent to having a friend watch your screen and give advice.

### Q: Do I need a USB cable?

**No.** Wireless ADB is supported. Enable Developer Options > Wireless Debugging on your phone, then `adb pair` + `adb connect`.

### Q: What hardware do I need?

- **PC:** Any modern CPU (GCN inference is 0.12ms). YOLO benefits from GPU.
- **Phone:** Any Android phone with USB debugging or wireless debugging.

### Q: Can I use this on emulator?

**Yes.** Most Android emulators (BlueStacks, LDPlayer, MEmu) support ADB. Set `ADB_DEVICE` to the emulator's ADB port.

### Q: How do I expand to all 503 heroes?

1. Run `server/data/scraper.py` to fetch full hero data from Liquipedia
2. Update `shared/hero_meta.json` with full roster
3. Regenerate synthetic images with `training/generate_images.py`
4. Retrain YOLO and GCN on expanded data

## Files Reference

| File | Purpose |
|------|---------|
| `server/main.py` | Entry point, async capture loop, pipeline orchestration |
| `server/config.py` | All configuration (env vars) |
| `server/capture/adb_capture.py` | ADB screen capture with wireless support |
| `server/detection/yolo_detector.py` | YOLOv8n-OBB hero detection |
| `server/detection/dummy_detector.py` | Random detection for testing |
| `server/recommendation/gcn_model.py` | MOBARec-GCNFP (47k params) |
| `server/recommendation/draft_state.py` | Draft state tracker |
| `server/websocket/server.py` | WebSocket broadcast server |
| `server/data/loader.py` | Hero metadata loader |
| `server/data/tournament_loader.py` | Tournament data + co-occurrence |
| `server/data/scraper.py` | Liquipedia async scraper |
| `shared/hero_meta.json` | 30-hero MVP metadata |
| `web_dashboard/index.html` | Dashboard UI |
| `web_dashboard/app.js` | WebSocket client |
| `web_dashboard/style.css` | Dashboard styles |
| `training/generate_synthetic.py` | Synthetic draft data generator |
| `training/generate_images.py` | Synthetic hero icon images |
| `training/train_yolo.py` | YOLO training script |
| `training/train_gcn.py` | GCN training script |
| `training/capture_screenshots.py` | Real screenshot capture |
| `training/dataset.yaml` | YOLO dataset config |
| `tests/` | 46 unit tests |
