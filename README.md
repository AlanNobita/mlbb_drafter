# MLBB Drafter

Real-time MLBB drafting assistant with GCN-based recommendations.

## Architecture

- **Server**: Python monolithic server with capture → detect → recommend → broadcast pipeline
- **Dashboard**: Web-based real-time dashboard with WebSocket updates
- **Model**: MOBARec-GCNFP (Lightweight GCN for draft recommendation)

## Project Structure

```
mlbb_drafter/
├── server/                    # Python server
│   ├── capture/              # ADB screen capture
│   ├── detection/            # YOLO hero detection (dummy for MVP)
│   ├── recommendation/       # GCN recommendation engine
│   ├── data/                 # Hero metadata loader
│   └── websocket/            # WebSocket server
├── shared/                   # Shared data (hero_meta.json)
├── web_dashboard/            # Web dashboard (HTML/JS/CSS)
├── training/                 # Model training scripts
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

### 2. Run Tests

```bash
pytest tests/ -v
```

### 3. Start Server

```bash
python -m server.main
```

### 4. Start Dashboard

```bash
python -m http.server 8080 --directory web_dashboard
```

### 5. Open Dashboard

Open http://localhost:8080 in your browser.

## Features

- **ADB Screen Capture**: Captures screen from Android device via ADB
- **Hero Detection**: YOLOv8n-OBB for hero icon detection (dummy for MVP)
- **GCN Recommendations**: MOBARec-GCNFP model for draft recommendations
- **Real-time Dashboard**: WebSocket-based live updates

## Configuration

Environment variables:

- `WS_HOST`: WebSocket server host (default: 0.0.0.0)
- `WS_PORT`: WebSocket server port (default: 8765)
- `CAPTURE_FPS`: Capture frame rate (default: 5)
- `USE_DUMMY_DETECTOR`: Use dummy detector (default: true)

## Next Steps

- [ ] Train YOLOv8n-OBB on MLBB hero icons
- [ ] Implement Liquipedia scraper for live meta data
- [ ] Integrate MLBB-API for hero statistics
- [ ] Build Android overlay app (Kotlin)
- [ ] Optimize GCN training on real tournament data
