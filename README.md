# MLBB Drafter

Real-time Mobile Legends: Bang Bang drafting assistant with GCN-based hero recommendations.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

## What This Does

MLBB Drafter watches your MLBB draft screen via ADB, detects hero picks/bans using computer vision, runs a Graph Convolutional Network (GCN) model for recommendations, and displays results on a real-time web dashboard.

**Key Features:**
- ADB screen capture (USB + wireless) - read-only, no game interaction
- YOLOv8n-OBB hero icon detection
- GCN model trained on 37,000+ tournament and ranked matches
- Counter-pick and synergy recommendations
- Real-time WebSocket dashboard
- Temporal meta-aware predictions (patch/era awareness)

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
├── server/                    # Python backend server
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
├── training/                 # Model training & data pipeline
│   ├── train_gcn_enhanced.py # Main training script (ALL data)
│   ├── train_gcn.py          # Legacy training (deprecated)
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
├── shared/
│   ├── hero_meta.json        # 132 heroes metadata
│   └── constants.py          # Shared constants
├── web_dashboard/            # Web dashboard (HTML/JS/CSS)
│   ├── index.html            # Dashboard UI
│   ├── app.js                # WebSocket client
│   └── style.css             # Dashboard styles
├── tests/                    # Unit & integration tests
├── docs/                     # Documentation
├── drive_data/               # Tournament data from Google Drive
├── runs/                     # YOLO training outputs
├── pyproject.toml            # Python project config
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Android phone** with USB debugging or wireless debugging enabled
- **ADB** installed (`apt install adb` or download from Google)
- **MLBB** installed on the phone
- Both devices on the **same WiFi network** (for wireless ADB)

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/mlbb_drafter.git
cd mlbb_drafter

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Connect Phone (Wireless ADB)

```bash
# On phone: Enable Developer Options > Wireless Debugging
# Tap "Pair device with pairing code" and note the pairing code

# Pair (one-time)
adb pair 192.168.1.50:37913
# Enter pairing code when prompted

# Connect
adb connect 192.168.1.50:5555

# Verify
adb devices
# Should show: 192.168.1.50:5555   device
```

### 3. Start Server

```bash
# With dummy detector (no YOLO needed, for testing)
python -m server.main

# With YOLO detector (requires trained model)
DETECTOR_TYPE=yolo YOLO_MODEL_PATH=runs/train/mlbb_hero_detect/weights/best.pt \
  python -m server.main

# With wireless ADB device
ADB_DEVICE=192.168.1.50:5555 python -m server.main
```

### 4. Open Dashboard

```bash
# In a new terminal
python -m http.server 8080 --directory web_dashboard
```

Open **http://localhost:8080** in your browser.

### 5. Run Tests

```bash
pytest tests/ -v
```

## Configuration

All configuration is done via environment variables in `server/config.py`:

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

## Training the Model

### GCN Model (Recommendation Engine)

The GCN model is trained on 37,000+ real tournament and ranked matches.

```bash
# Train with recommended settings
python3 training/train_gcn_enhanced.py \
  --epochs 500 \
  --lr 0.00005 \
  --hidden-dim 256 \
  --batch-size 64 \
  --min-year 2021 \
  --temporal-decay 0.5

# Output: training/data/gcn_model_v2.pt
```

**Training Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 500 | Training epochs |
| `--lr` | 0.00005 | Learning rate |
| `--hidden-dim` | 256 | Hidden dimension size |
| `--batch-size` | 64 | Batch size |
| `--min-year` | 2017 | Filter drafts before this year |
| `--temporal-decay` | 0.5 | Time decay (0=equal, 1=only recent) |
| `--val-split` | 0.2 | Validation split ratio |
| `--patience` | 20 | Early stopping patience |
| `--output` | `training/data/gcn_model_v2.pt` | Output path |

**What the model learns:**
- Hero win rates, pick rates, ban rates
- Synergy patterns (heroes that work well together)
- Counter-pick relationships (heroes that beat others)
- Temporal meta trends (patch-aware predictions)
- Tournament draft patterns from professional play

### YOLO Model (Hero Detection)

```bash
# Generate synthetic training images
python3 training/generate_images.py

# Train YOLO
python3 training/train_yolo.py --data training/dataset.yaml --epochs 100

# For better results, capture real screenshots
python3 training/capture_screenshots.py --count 100 --serial 192.168.1.50:5555
```

### Fetch Latest Data

```bash
# Fetch from all APIs (never overwrites existing data)
python3 training/fetch_all_data.py

# Fetch hero stats from OpenMLBB
python3 training/fetch_hero_stats_api.py

# Fetch from mlbb.io (requires API key from parse.bot)
python3 training/fetch_mlbb_io.py --api-key YOUR_KEY

# Download Kaggle datasets
python3 training/fetch_kaggle_datasets.py --download-all

# Process tournament data
python3 training/process_drive_data.py

# Add temporal features
python3 training/add_temporal_features.py

# Create consolidated training file
python3 training/create_final_training_data.py
```

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

### Data Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources                              │
├─────────────────────────────────────────────────────────────┤
│  Google Drive → process_drive_data.py → tournament_drafts.json │
│  Kaggle → fetch_kaggle_datasets.py → tournament_drafts.json    │
│  OpenMLBB → fetch_hero_stats_api.py → hero_stats_openmlbb.json│
│  Pren7 → fetch_all_data.py → hero_winrate.json                │
│  Liquipedia → fetch_liquipedia.py → tournament_drafts.json     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Processing                                │
├─────────────────────────────────────────────────────────────┤
│  process_api_data.py → adjacency_real.json (132x132 matrix)  │
│  add_temporal_features.py → hero_stats_temporal.csv           │
│  create_final_training_data.py → MLBB_TRAINING_DATA_FINAL.json│
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Training                                  │
├─────────────────────────────────────────────────────────────┤
│  train_gcn_enhanced.py → gcn_model_v2.pt (132 heroes)       │
└─────────────────────────────────────────────────────────────┘
```

## Model Details

### GCN v2 Architecture

```
Hero Features (10 dims) → Linear(10, 256) → GCN(256, 256) → ReLU → Linear(256, 1) → Sigmoid
```

- **Heroes:** 132 (full MLBB roster)
- **Features per hero:** 10 (win_rate, pick_rate, ban_rate, meta_score, trend, weighted_wr, recency, total_games, synergy_wr, synergy_games)
- **Graph:** 132x132 adjacency matrix (hero relationships)
- **Parameters:** 18,744
- **Output:** Win probability [0, 1]

### How Recommendations Work

1. **Build hero features:** Load win rates, pick rates, ban rates from API data
2. **Apply GCN:** Propagate information through hero relationship graph
3. **Dynamic embedding:** `em0 = Σ(ally_heroes) - Σ(enemy_heroes)`
4. **Score each available hero:** Project through linear layers
5. **Counter boost:** Boost counter picks by up to +15% based on API counter data
6. **Return top-k:** Top recommendations with predicted win rates

### Counter-Pick System

The model uses two sources for counter-pick data:
- **GCN adjacency matrix:** Learned from tournament co-occurrence patterns
- **API counter data:** 264 counter relationships from OpenMLBB (660 entries total)

Counter picks are boosted by up to +15% in the final recommendation score.

## Dashboard Features

- **Live Draft Display:** Shows ally picks, enemy picks, bans in real-time
- **Top Recommendations:** 3 best hero picks with predicted win rates
- **Counter Picks:** Heroes that counter the enemy team composition
- **Synergy Picks:** Heroes that synergize with your team
- **Strategy Flags:** Early Game, Late Game, Team Fight, Split Push, Pick Off, Push Tower
- **Counter Alerts:** Warnings when enemy has counter-picks
- **Role Lanes:** EXP Lane, Gold Lane, Mid Lane, Roam, Jungle assignments

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

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_recommendation.py -v

# Run with coverage
pytest tests/ -v --cov=server --cov-report=term-missing
```

**Test Suite:**
- `test_integration.py` - End-to-end pipeline tests
- `test_recommendation.py` - GCN model tests
- `test_detection.py` - YOLO detector tests
- `test_capture.py` - ADB capture tests
- `test_websocket.py` - WebSocket server tests
- `test_tournament_loader.py` - Data pipeline tests

## Manual Setup Required

### 1. mlbb.io Parse API Key (Free)

The mlbb.io API provides rank-specific hero statistics (Mythic/Legend/Epic win rates).

1. Go to [parse.bot](https://parse.bot) and create a free account
2. Get your API key
3. Run:
   ```bash
   python3 training/fetch_mlbb_io.py --api-key YOUR_KEY
   ```

**Note:** Free tier gives 100 credits/month, 5 requests/minute.

### 2. Kaggle API (Optional)

For downloading MLBB tournament datasets from Kaggle:

1. Create a [Kaggle account](https://www.kaggle.com/account/login)
2. Go to [Kaggle API Settings](https://www.kaggle.com/settings/api) and generate a token
3. Either:
   - Set environment variable: `export KAGGLE_API_TOKEN=your_token`
   - Or save to file: `~/.kaggle/access_token`
4. Run:
   ```bash
   python3 training/fetch_kaggle_datasets.py --download-all
   ```

### 3. GPU for YOLO Training (Recommended)

YOLO training is very slow on CPU (~33 min/epoch). For better performance:

- Use NVIDIA GPU with CUDA
- Install CUDA toolkit and cuDNN
- Set `YOLO_DEVICE=cuda:0` when training

### 4. Real Screenshots for YOLO Fine-tuning

The YOLO model is trained on synthetic images. For better detection:

1. Capture 100+ real screenshots during live drafts:
   ```bash
   python3 training/capture_screenshots.py --count 100 --serial 192.168.1.50:5555
   ```
2. Label the screenshots using [Roboflow](https://roboflow.com) or [CVAT](https://cvat.ai)
3. Retrain YOLO on real data

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `adb: device not found` | Check USB cable, enable USB debugging, or use wireless ADB |
| `unable to connect` | Ensure same WiFi network, check firewall |
| `connection refused` | Re-enable wireless debugging on phone |
| `unauthorized` | Accept "Allow USB debugging" popup on phone |
| Slow capture | Reduce `CAPTURE_FPS` env var (default: 5) |
| YOLO training slow | Use GPU, or reduce epochs |
| Model not loading | Ensure `training/data/gcn_model_v2.pt` exists |

## Roadmap

### Phase 1: MVP Pipeline ✅
- [x] ADB capture (USB + wireless)
- [x] Dummy detector for testing
- [x] YOLO detector skeleton
- [x] GCN model v2 (132 heroes)
- [x] WebSocket server
- [x] Web dashboard
- [x] 37K+ tournament drafts
- [x] Counter-pick system
- [x] Synergy recommendations
- [x] Temporal meta awareness

### Phase 2: Enhancement 🔄
- [x] Integrate all API data sources
- [x] Add temporal features for meta awareness
- [x] Add synergy/co-occurrence learning
- [ ] Get mlbb.io ranked stats (needs API key)
- [ ] Train YOLO on GPU with real data
- [ ] Measure Recall@1, NDCG@3

### Phase 3: Production
- [ ] Android overlay app (Kotlin)
- [ ] Latency benchmarking (200ms target)
- [ ] Memory profiling (<500MB target)
- [ ] Error recovery / reconnection

### Phase 4: Advanced
- [ ] Multi-device support
- [ ] Hero pool expansion (503+ heroes)
- [ ] Live patch updates
- [ ] Match history tracking

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

## FAQ

### Q: Will this get me banned?

**No.** The tool only reads your screen via ADB. It never interacts with the game client, never modifies files, never sends input commands. It's visually equivalent to having a friend watch your screen and give advice.

### Q: Do I need a USB cable?

**No.** Wireless ADB is supported. Enable Developer Options > Wireless Debugging on your phone, then `adb pair` + `adb connect`.

### Q: What hardware do I need?

- **PC:** Any modern CPU (GCN inference is ~1ms). YOLO benefits from GPU.
- **Phone:** Any Android phone with USB debugging or wireless debugging.
- **Network:** Both devices on same WiFi (for wireless ADB).

### Q: Can I use this on emulator?

**Yes.** Most Android emulators (BlueStacks, LDPlayer, MEmu) support ADB. Set `ADB_DEVICE` to the emulator's ADB port.

### Q: How do I retrain the model?

```bash
# Fetch latest data
python3 training/fetch_all_data.py

# Retrain
python3 training/train_gcn_enhanced.py --epochs 500

# The new model is saved to training/data/gcn_model_v2.pt
```

### Q: What's the difference between v1 and v2 models?

- **v1 (gcn_model.py):** 30 heroes, 47K params, synthetic data only
- **v2 (gcn_model_v2.py):** 132 heroes, 18.7K params, 37K+ real drafts, counter-boost, temporal features

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [LiTianYeoh/MLBB_Tournament_Analysis](https://github.com/LiTianYeoh/MLBB_Tournament_Analysis) - Tournament data
- [OpenMLBB API](https://openmlbb.org) - Hero stats and counters
- [Pren7/MLBB-Winrate](https://github.com/Pren7/MLBB-Winrate) - Win rate data
- [Kaggle MLBB Datasets](https://www.kaggle.com) - Tournament and ranked match data
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - Graph neural networks
