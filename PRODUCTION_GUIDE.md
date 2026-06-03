# MLBB Drafter - Production Guide

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MLBB Drafter Pipeline                     │
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  Phone    │───>│  ADB Capture │───>│  YOLOv8n-OBB     │   │
│  │  (MLBB)   │    │  (Wireless)  │    │  Hero Detection   │   │
│  └──────────┘    └──────────────┘    └────────┬─────────┘   │
│                                                │              │
│                                    ┌───────────▼──────────┐  │
│                                    │  Draft State Tracker  │  │
│                                    │  (Ally/Enemy/Bans)    │  │
│                                    └───────────┬──────────┘  │
│                                                │              │
│                                    ┌───────────▼──────────┐  │
│                                    │  GCN Model v2         │  │
│                                    │  (132 heroes, 18.7K)  │  │
│                                    │  Counter boost +15%   │  │
│                                    └───────────┬──────────┘  │
│                                                │              │
│                                    ┌───────────▼──────────┐  │
│                                    │  WebSocket Server     │  │
│                                    │  (ws://0.0.0.0:8765)  │  │
│                                    └───────────┬──────────┘  │
│                                                │              │
│                                    ┌───────────▼──────────┐  │
│                                    │  Dashboard            │  │
│                                    │  - Top Picks          │  │
│                                    │  - Counter Picks      │  │
│                                    │  - Synergy Picks      │  │
│                                    │  - Strategy Flags     │  │
│                                    └──────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Pipeline:** Capture → Detect → Track → Recommend → Broadcast → Display

## Hardware Setup: Wireless ADB Connection

### Prerequisites
- Android phone with MLBB installed
- PC running Ubuntu/Linux (or Windows with ADB)
- Both devices on the same WiFi network

### Step 1: Enable Wireless Debugging on Phone

1. Open **Settings** → **Developer Options** (enable if hidden: tap Build Number 7 times)
2. Enable **Wireless Debugging**
3. Tap **Wireless Debugging** to open settings
4. Note the **IP Address & Port** (e.g., `192.168.1.50:5555`)

### Step 2: Pair Phone with PC (Android 11+)

```bash
# On phone: Tap "Pair device with pairing code"
# Note the pairing code and port (e.g., 192.168.1.50:37000)

# On PC:
adb pair 192.168.1.50:37000
# Enter pairing code when prompted
```

### Step 3: Connect to Phone

```bash
# Connect using the IP:port from Step 1
adb connect 192.168.1.50:5555

# Verify connection
adb devices
# Should show: 192.168.1.50:5555   device
```

### Step 4: Start Server with Wireless Device

```bash
# Set the device serial
export ADB_DEVICE="192.168.1.50:5555"

# Start the server
cd /home/alan/Documents/code/mlbb_drafter
venv/bin/python -m server.main
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `unable to connect` | Ensure same WiFi network, check firewall |
| `connection refused` | Re-enable wireless debugging on phone |
| `unauthorized` | Accept "Allow USB debugging" popup on phone |
| Slow capture | Reduce `CAPTURE_FPS` env var (default: 5) |

## Training Report

### Vision Model (YOLOv8n-OBB)

| Metric | Value | Status |
|--------|-------|--------|
| Model | YOLOv8n-obb.pt | Pretrained (COCO) |
| Parameters | 3,083,685 | Lightweight |
| Training Images | 5,000 synthetic | 6 hero classes, 640x640 |
| Augmentation | Rotation ±15°, Gaussian noise, glow | Applied |
| Dataset | `training/data/synthetic_images/` | 4000 train / 1000 val |
| mAP50 | TBD | Requires GPU training |

**Note:** YOLO training requires GPU or extended CPU runtime (~33 min/epoch on CPU). Run with:
```bash
# GPU (recommended, ~5 min total)
venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov8n-obb.pt').train(data='training/dataset.yaml', epochs=50, device=0)"

# CPU (slow, ~16 hours)
venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov8n-obb.pt').train(data='training/dataset.yaml', epochs=30, device='cpu')"
```

### Neural Engine (GCN Model v2)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Parameters | 18,744 | <100,000 | ✅ |
| Hero Coverage | 132 | 132 | ✅ |
| Input Features | 10 | - | ✅ |
| Hidden Dim | 256 | - | ✅ |
| Output | Win probability [0,1] | - | ✅ |
| Training Data | 37,373 drafts | 30,000+ | ✅ |
| Training Time | ~5 min (500 epochs) | <10min | ✅ |
| Inference Time | ~1ms | <100ms | ✅ |
| Counter Boost | +15% | - | ✅ |
| Temporal Features | Yes | - | ✅ |

## Operational Guide

### Starting the Server

```bash
cd /home/alan/Documents/code/mlbb_drafter

# Optional: Set environment variables
export DETECTOR_TYPE="dummy"  # or "yolo" when trained
export ADB_DEVICE="192.168.1.50:5555"  # wireless ADB
export WS_PORT="8765"
export CAPTURE_FPS="5"

# Start server
venv/bin/python -m server.main
```

### Opening the Dashboard

1. Open browser to `http://localhost:8080`
2. Dashboard connects via WebSocket to `ws://localhost:8765`
3. Real-time draft updates appear automatically

### Dashboard Features

- **Live Draft Display:** Shows ally picks, enemy picks, bans
- **Top Recommendations:** 3 best hero picks with predicted win rates
- **Counter Picks:** Heroes that counter the enemy team composition
- **Synergy Picks:** Heroes that synergize with your team
- **Strategy Flags:** Early Game, Late Game, Team Fight, Split Push, Pick Off, Push Tower
- **Counter Alerts:** Warnings when enemy has counter-picks
- **Role Lanes:** EXP Lane, Gold Lane, Mid Lane, Roam, Jungle assignments

### Running Tests

```bash
# Full test suite
venv/bin/python -m pytest tests/ -v

# Specific test file
venv/bin/python -m pytest tests/test_recommendation.py -v
```

## Anti-Ban Safety Confirmation

### Read-Only Architecture

The MLBB Drafter is designed as a **strictly visual, read-only** tool:

| Component | Access | Risk Level |
|-----------|--------|------------|
| ADB Capture | `screencap` only | ✅ Zero risk |
| ADB Input | **NEVER used** | ✅ Zero risk |
| Game Memory | **NEVER accessed** | ✅ Zero risk |
| Game Files | **NEVER modified** | ✅ Zero risk |
| Automation | **NEVER performed** | ✅ Zero risk |

### What the System Does NOT Do

- ❌ Does not send `input tap` or `input swipe` commands
- ❌ Does not read game memory or process data
- ❌ Does not modify game files or settings
- ❌ Does not automate gameplay actions
- ❌ Does not interact with the game client directly

### What the System DOES Do

- ✅ Captures screen pixels via `adb exec-out screencap -p`
- ✅ Analyzes images using computer vision (YOLO)
- ✅ Provides recommendations via overlay dashboard
- ✅ All processing happens on the PC, not the phone

### Data Flow Safety

```
Phone Screen → ADB Screenshot → PC Analysis → Dashboard Display
    (read)       (read-only)      (local)       (local)
```

No data is sent back to the phone. No game interaction occurs.

## Capturing Real Screenshots

To bridge the "domain gap" and improve detection accuracy:

```bash
# Capture 100 screenshots during a live draft
venv/bin/python training/capture_screenshots.py --count 100 --output training/data/real/

# Then fine-tune the YOLO model
venv/bin/python training/train_yolo.py --data training/dataset.yaml --epochs 50
```

## Model Files

| File | Description |
|------|-------------|
| `training/data/gcn_model_v2.pt` | Trained GCN weights (18.7K params, 132 heroes) |
| `training/data/adjacency_real.json` | 132x132 hero relationship graph |
| `training/data/tournament_drafts.json` | 37,373 tournament + ranked matches |
| `training/data/hero_stats_temporal.csv` | 122 heroes with temporal features |
| `runs/train/mlbb_hero_detect/weights/best.pt` | YOLO detection weights |
| `shared/hero_meta.json` | 132 heroes metadata |

## Data Sources

| Source | Data | Count | Status |
|--------|------|-------|--------|
| Google Drive | Tournament matches (M1-M5, MPL) | 13,171 | ✅ Integrated |
| Kaggle Match Results | Ranked matches (patch 1.7.58/1.7.68) | 10,664 | ✅ Integrated |
| Kaggle M5/M7 | Tournament drafts | 73 | ✅ Integrated |
| OpenMLBB API | Hero stats, counters, synergies | 132 heroes | ✅ Integrated |
| Pren7/MLBB-Winrate | Daily win/pick/ban rates | 132 heroes | ✅ Integrated |
| p3hndrx API | Hero metadata | 132 heroes | ✅ Integrated |
| Liquipedia | Tournament drafts | Rate-limited | ⚠️ Partial |
| mlbb.io | Rank-specific stats | Needs API key | ⏳ Pending |

**Total Training Data:** 37,373 drafts from 150+ tournaments (2017-2024)

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ADB_DEVICE` | `""` | Wireless ADB IP:port |
| `DETECTOR_TYPE` | `"dummy"` | `"dummy"` or `"yolo"` |
| `WS_HOST` | `"0.0.0.0"` | WebSocket bind address |
| `WS_PORT` | `"8765"` | WebSocket port |
| `CAPTURE_FPS` | `"5"` | Capture frame rate |
| `YOLO_MODEL_PATH` | `"runs/train/.../best.pt"` | YOLO weights path |
| `YOLO_CONFIDENCE` | `"0.5"` | Detection threshold |
| `DASHBOARD_PORT` | `"8080"` | Dashboard HTTP port |

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Full pipeline latency | <200ms | Not measured |
| GCN inference | <100ms | ~1ms |
| Memory usage | <500MB | Not measured |
| Capture FPS | 5-10 | 5 (configurable) |
| GCN parameters | <100K | 18,744 |
| Training data | 30K+ drafts | 37,373 |
| Hero coverage | 132+ | 132 |

## Deployment Checklist

- [ ] ADB installed and working
- [ ] Phone connected via USB or wireless ADB
- [ ] Python dependencies installed
- [ ] GCN model trained (`training/data/gcn_model_v2.pt` exists)
- [ ] YOLO model trained (optional, for production detection)
- [ ] Dashboard accessible at `http://localhost:8080`
- [ ] WebSocket server running on port 8765
- [ ] Tests passing (`pytest tests/ -v`)
