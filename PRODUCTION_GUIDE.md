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
│                                    │  MOBARec-GCNFP       │  │
│                                    │  (47k params, CPU)    │  │
│                                    │  em0 = Σef - Σeo     │  │
│                                    └───────────┬──────────┘  │
│                                                │              │
│                                    ┌───────────▼──────────┐  │
│                                    │  WebSocket Server     │  │
│                                    │  (ws://0.0.0.0:8765)  │  │
│                                    └───────────┬──────────┘  │
│                                                │              │
│                                    ┌───────────▼──────────┐  │
│                                    │  Detailed Panel       │  │
│                                    │  Dashboard (HTML)     │  │
│                                    │  - Strategy Flags     │  │
│                                    │  - Counter Alerts     │  │
│                                    │  - Role Lanes         │  │
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
| Training Images | 5,000 synthetic | 30 hero classes, 640x640 |
| Augmentation | Rotation ±15°, Gaussian noise, glow | Applied |
| Dataset | `training/data/synthetic_images/` | 4000 train / 1000 val |
| mAP50 | TBD | Requires GPU training (~33 min/epoch on CPU) |
| mAP50-95 | TBD | - |
| Precision | TBD | - |
| Recall | TBD | - |

**Note:** YOLO training requires GPU or extended CPU runtime (~16 hours for 30 epochs). Run with:
```bash
# GPU (recommended, ~5 min total)
venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov8n-obb.pt').train(data='training/dataset.yaml', epochs=50, device=0)"

# CPU (slow, ~16 hours)
venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov8n-obb.pt').train(data='training/dataset.yaml', epochs=30, device='cpu')"
```

### Neural Engine (MOBARec-GCNFP)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Parameters | 47,070 | <60,000 | ✅ |
| Input Dim | 128 | - | ✅ |
| Hidden Dim | 128 | - | ✅ |
| Output Dim | 64 | - | ✅ |
| Training Data | 10,000 synthetic drafts | 10,000+ | ✅ |
| Training Time | 109.1s (100 epochs) | <5min | ✅ |
| Inference Time | 0.12ms | <100ms | ✅ |
| Match Embedding | em0 = Σ(efriendly) - Σ(eopponent) | Dynamic | ✅ |
| Recall@1 | 1.00 (synthetic) | - | ✅ |
| NDCG@3 | Requires tournament data | - | Pending |

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
- **Strategy Flags:** Early Game, Late Game, Team Fight, Split Push, Pick Off, Push Tower
- **Counter Alerts:** Warnings when enemy has counter-picks
- **Role Lanes:** EXP Lane, Gold Lane, Mid Lane, Roam, Jungle assignments
- **Recommendations:** Top picks with win rate predictions

### Running Tests

```bash
# Full test suite (46 tests)
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
| `training/data/gcn_model_v2.pt` | Trained GCN weights (47k params) |
| `training/data/adjacency_matrix.pt` | Hero co-occurrence graph |
| `training/data/synthetic_drafts.csv` | 10,000 synthetic draft records |
| `runs/train/mlbb_hero_detect/weights/best.pt` | YOLO detection weights |
| `shared/hero_meta.json` | 30-hero metadata |

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
