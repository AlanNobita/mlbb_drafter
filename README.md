# MLBB Drafter

Real-time MLBB drafting assistant with GCN-based recommendations.

## Architecture

- **Server**: Python monolithic server with capture → detect → recommend → broadcast pipeline
- **Dashboard**: Web-based real-time dashboard with WebSocket updates
- **Model**: MOBARec-GCNFP v2 (Enhanced GCN with full API data integration)

## Project Structure

```
mlbb_drafter/
├── server/                    # Python server
│   ├── capture/              # ADB screen capture
│   ├── detection/            # YOLO hero detection
│   ├── recommendation/       # GCN recommendation engine
│   │   ├── gcn_model.py      # Original GCN (legacy)
│   │   ├── gcn_model_v2.py   # Enhanced GCN with all API data
│   │   └── draft_state.py    # Draft state tracker
│   ├── data/                 # Hero metadata loader
│   └── websocket/            # WebSocket server
├── training/                 # Model training scripts
│   ├── train_gcn_enhanced.py # Main training script (uses ALL data)
│   ├── train_gcn.py          # Old training (deprecated)
│   ├── train_yolo.py         # YOLO training for detection
│   ├── fetch_all_data.py     # Fetch from all APIs (merge mode)
│   ├── process_drive_data.py # Process tournament data
│   ├── add_temporal_features.py # Add time-aware features
│   ├── process_api_data.py   # Process API-specific data
│   ├── create_final_training_data.py # Create consolidated file
│   ├── backup_data.py        # Backup before updates
│   └── data/                 # All training data
│       ├── gcn_model_v2.pt   # Trained model (132 heroes)
│       ├── tournament_drafts.json # 13,171 tournament matches
│       ├── adjacency_real.json # 132x132 hero relationships
│       ├── hero_stats_temporal.csv # Temporal features
│       ├── hero_cooccurrence_tournament.json # Synergy stats
│       ├── synergy_matrix_tournament.json # Tournament synergy
│       ├── era_hero_stats.json # Per-patch hero stats
│       └── api_data/         # Raw API data
├── drive_data/               # Tournament data from Google Drive
├── web_dashboard/            # Web dashboard (HTML/JS/CSS)
└── tests/                    # Unit tests
```

## Quick Start

### 1. Setup Virtual Environment

```bash
cd mlbb_drafter
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Train GCN with ALL data (recommended)
python3 training/train_gcn_enhanced.py --epochs 100

# The trained model is saved to training/data/gcn_model_v2.pt
```

### 3. Run Tests

```bash
pytest tests/ -v
```

### 4. Start Server

```bash
python -m server.main
```

### 5. Start Dashboard

```bash
python -m http.server 8080 --directory web_dashboard
```

### 6. Open Dashboard

Open http://localhost:8080 in your browser.

## Training Commands

### GCN Model (Recommendation) - Main Training

```bash
python3 training/train_gcn_enhanced.py --epochs 100
```

**Data used:**
- `tournament_drafts.json` — 13,171 real tournament matches
- `hero_stats_temporal.csv` — 122 heroes with temporal features
- `adjacency_real.json` — 132x132 hero relationship graph
- `hero_winrate.json` — 132 heroes win/pick/ban rates
- `hero_cooccurrence_tournament.json` — Synergy stats (122 heroes)
- `synergy_matrix_tournament.json` — Tournament synergy matrix

**Options:**
- `--epochs 200` — More training iterations
- `--lr 0.0005` — Slower learning rate
- `--batch-size 64` — Larger batches
- `--val-split 0.2` — 20% validation split
- `--output path/to/model.pt` — Custom output path

### YOLO Model (Hero Detection)

```bash
python3 training/train_yolo.py --data training/dataset.yaml --epochs 100
```

**Note:** Uses synthetic images. For better detection, use real MLBB screenshots.

### Fetch Latest Data

```bash
# Fetch from all APIs (never overwrites existing data)
python3 training/fetch_all_data.py

# Process tournament data
python3 training/process_drive_data.py

# Add temporal features
python3 training/add_temporal_features.py
```

## Model Details

### GCN v2 Architecture

- **Heroes**: 132 (full MLBB roster)
- **Features**: 10 per hero
  - win_rate, pick_rate, ban_rate
  - meta_score, trend
  - weighted_win_rate, recency_score
  - total_games
  - synergy_win_rate, synergy_games
- **Graph**: 132x132 adjacency matrix (11,380 edges)
- **Parameters**: 43,137
- **Output**: Win probability [0, 1]

### Data Sources

| Source | Data | Count |
|--------|------|-------|
| Google Drive | Tournament matches | 13,171 |
| OpenMLBB API | Hero stats, counters | 132 heroes |
| MLBB Winrate API | Current win/pick/ban | 132 heroes |
| p3hndrx API | Hero metadata | 132 heroes |

## Features

- **ADB Screen Capture**: Captures screen from Android device via ADB
- **Hero Detection**: YOLOv8n-OBB for hero icon detection
- **GCN Recommendations**: Enhanced GCN with all API data integration
- **Real-time Dashboard**: WebSocket-based live updates
- **Temporal Features**: Time-aware predictions based on meta trends
- **Synergy Learning**: Uses tournament synergy and co-occurrence data

## Configuration

Environment variables:

- `WS_HOST`: WebSocket server host (default: 0.0.0.0)
- `WS_PORT`: WebSocket server port (default: 8765)
- `CAPTURE_FPS`: Capture frame rate (default: 5)
- `USE_DUMMY_DETECTOR`: Use dummy detector (default: true)

## Next Steps

- [x] Train GCN on real tournament data
- [x] Integrate all API data sources
- [x] Add temporal features for meta awareness
- [x] Add synergy/co-occurrence learning
- [ ] Train YOLOv8n-OBB on MLBB hero icons
- [ ] Build Android overlay app (Kotlin)
- [ ] Optimize GCN for real-time inference
