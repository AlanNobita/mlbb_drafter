# MLBB Drafter MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-time MLBB drafting assistant that captures screen via ADB, detects heroes with YOLO, recommends picks via GCN, and displays results on a web dashboard.

**Architecture:** Monolithic Python server with pipeline stages: Capture → Detect → Recommend → Serve. WebSocket pushes updates to web dashboard. Android overlay is a future phase.

**Tech Stack:** Python 3.14, OpenCV, Ultralytics YOLOv8, PyTorch, WebSocket (websockets library), FastAPI for dashboard, ADB for screen capture.

---

## File Structure

```
mlbb_drafter/
├── server/
│   ├── __init__.py
│   ├── main.py                    # Entry point, starts all services
│   ├── config.py                  # Configuration constants
│   ├── capture/
│   │   ├── __init__.py
│   │   └── adb_capture.py         # ADB screen capture module
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── yolo_detector.py       # YOLOv8n-OBB inference
│   │   └── dummy_detector.py      # Dummy detector for MVP
│   ├── recommendation/
│   │   ├── __init__.py
│   │   ├── gcn_model.py           # GCN recommendation engine
│   │   └── draft_state.py         # Track draft state
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py              # Load hero metadata
│   │   └── scraper.py             # Liquipedia scraper
│   └── websocket/
│       ├── __init__.py
│       └── server.py              # WebSocket server
├── shared/
│   ├── __init__.py
│   ├── hero_meta.json             # Hero attributes
│   └── constants.py               # Shared constants
├── web_dashboard/
│   ├── index.html                 # Dashboard UI
│   ├── app.js                     # WebSocket client + UI logic
│   └── style.css                  # Dashboard styling
├── training/
│   ├── train_yolo.py              # YOLO training script
│   ├── train_gcn.py               # GCN training script
│   └── data/                      # Training data
├── tests/
│   ├── __init__.py
│   ├── test_capture.py
│   ├── test_detection.py
│   ├── test_recommendation.py
│   └── test_websocket.py
├── requirements.txt
└── pyproject.toml
```

---

## Phase 1: Project Scaffolding

### Task 1: Initialize Project Structure

**Files:**
- Create: `mlbb_drafter/pyproject.toml`
- Create: `mlbb_drafter/requirements.txt`
- Create: `mlbb_drafter/server/__init__.py`
- Create: `mlbb_drafter/shared/__init__.py`
- Create: `mlbb_drafter/shared/constants.py`
- Create: `mlbb_drafter/tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "mlbb-drafter"
version = "0.1.0"
description = "Real-time MLBB drafting assistant"
requires-python = ">=3.10"
dependencies = [
    "opencv-python>=4.8.0",
    "ultralytics>=8.0.0",
    "torch>=2.0.0",
    "websockets>=12.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "numpy>=1.24.0",
    "pillow>=10.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]
```

- [ ] **Step 2: Create requirements.txt**

```
opencv-python>=4.8.0
ultralytics>=8.0.0
torch>=2.0.0
websockets>=12.0
fastapi>=0.100.0
uvicorn>=0.23.0
numpy>=1.24.0
pillow>=10.0.0
pytest>=7.0.0
pytest-asyncio>=0.21.0
```

- [ ] **Step 3: Create __init__.py files**

```bash
touch mlbb_drafter/server/__init__.py
touch mlbb_drafter/shared/__init__.py
touch mlbb_drafter/tests/__init__.py
```

- [ ] **Step 4: Create shared/constants.py**

```python
"""Shared constants for MLBB Drafter."""

# Hero roles
ROLES = ["Tank", "Fighter", "Assassin", "Mage", "Marksman", "Support"]

# Draft phases
PHASE_BAN = "ban"
PHASE_PICK = "pick"

# WebSocket
WS_HOST = "0.0.0.0"
WS_PORT = 8765

# ADB
ADB_CAPTURE_CMD = ["adb", "exec-out", "screencap", "-p"]

# YOLO
YOLO_MODEL_PATH = "server/detection/models/yolov8n-obb.pt"
CONFIDENCE_THRESHOLD = 0.5

# GCN
GCN_INPUT_DIM = 64
GCN_HIDDEN_DIM = 32
GCN_OUTPUT_DIM = 16
```

- [ ] **Step 5: Install dependencies and verify**

```bash
cd mlbb_drafter
pip install -r requirements.txt
python -c "import cv2; import torch; import websockets; print('Dependencies OK')"
```

- [ ] **Step 6: Commit**

```bash
git init
git add .
git commit -m "feat: initialize project structure with dependencies"
```

---

## Phase 2: ADB Screen Capture

### Task 2: ADB Capture Module

**Files:**
- Create: `mlbb_drafter/server/capture/__init__.py`
- Create: `mlbb_drafter/server/capture/adb_capture.py`
- Create: `mlbb_drafter/tests/test_capture.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_capture.py
import pytest
from server.capture.adb_capture import ADBCapture

def test_adbcapture_initializes():
    capture = ADBCapture()
    assert capture is not None

def test_adbcapture_frame_buffer_size():
    capture = ADBCapture(buffer_size=1)
    assert capture.buffer_size == 1

def test_adbcapture_has_capture_method():
    capture = ADBCapture()
    assert hasattr(capture, 'capture')
    assert callable(capture.capture)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd mlbb_drafter
pytest tests/test_capture.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'server.capture'"

- [ ] **Step 3: Write minimal implementation**

```python
# server/capture/__init__.py
from .adb_capture import ADBCapture

__all__ = ["ADBCapture"]
```

```python
# server/capture/adb_capture.py
"""ADB screen capture module."""
import subprocess
import numpy as np
from typing import Optional
import cv2

class ADBCapture:
    """Captures screen from Android device via ADB."""
    
    def __init__(self, buffer_size: int = 1):
        self.buffer_size = buffer_size
        self._buffer: Optional[np.ndarray] = None
    
    def capture(self) -> Optional[np.ndarray]:
        """Capture a single frame from the device.
        
        Returns:
            numpy.ndarray: BGR image array or None if capture fails.
        """
        try:
            result = subprocess.run(
                ["adb", "exec-out", "screencap", "-p"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                return None
            
            # Convert PNG bytes to numpy array
            nparr = np.frombuffer(result.stdout, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except (subprocess.TimeoutExpired, Exception) as e:
            print(f"ADB capture error: {e}")
            return None
    
    def capture_loop(self, callback, fps: int = 10):
        """Run continuous capture loop.
        
        Args:
            callback: Function to call with each frame
            fps: Target frames per second
        """
        import time
        delay = 1.0 / fps
        
        while True:
            frame = self.capture()
            if frame is not None:
                callback(frame)
            time.sleep(delay)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd mlbb_drafter
pytest tests/test_capture.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/capture/ tests/test_capture.py
git commit -m "feat: add ADB screen capture module"
```

---

## Phase 3: Dummy YOLO Detector

### Task 3: Dummy Detector for MVP

**Files:**
- Create: `mlbb_drafter/server/detection/__init__.py`
- Create: `mlbb_drafter/server/detection/dummy_detector.py`
- Create: `mlbb_drafter/tests/test_detection.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_detection.py
import pytest
import numpy as np
from server.detection.dummy_detector import DummyDetector

def test_dummy_detector_initializes():
    detector = DummyDetector()
    assert detector is not None

def test_dummy_detector_returns_heroes():
    detector = DummyDetector()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert "ally_picks" in result
    assert "enemy_picks" in result
    assert "bans" in result

def test_dummy_detector_returns_correct_format():
    detector = DummyDetector()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert isinstance(result["ally_picks"], list)
    assert isinstance(result["enemy_picks"], list)
    assert isinstance(result["bans"], list)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd mlbb_drafter
pytest tests/test_detection.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'server.detection'"

- [ ] **Step 3: Write minimal implementation**

```python
# server/detection/__init__.py
from .dummy_detector import DummyDetector

__all__ = ["DummyDetector"]
```

```python
# server/detection/dummy_detector.py
"""Dummy detector for MVP testing."""
import random
from typing import Dict, List

# Sample MLBB heroes for testing
SAMPLE_HEROES = [
    "Lancelot", "Gusion", "Fanny", "Hayabusa", "Ling",
    "Chou", "Alucard", "Selena", "Kagura", "Harith",
    "Valir", "Lunox", "Esmeralda", "Yu Zhong", "Benedetta",
    "Atlas", "Khufra", "Akai", "Grock", "Johnson",
    "Angela", "Rafaela", "Estes", "Diggie", "Mathilda",
    "Granger", "Wanwan", "Claude", "Karrie", "Moskov",
]

class DummyDetector:
    """Dummy detector that returns random hero detections for MVP testing."""
    
    def __init__(self):
        self.heroes = SAMPLE_HEROES
    
    def detect(self, frame) -> Dict[str, List[str]]:
        """Detect heroes in frame (dummy implementation).
        
        Args:
            frame: numpy.ndarray (unused in dummy)
            
        Returns:
            Dictionary with ally_picks, enemy_picks, bans
        """
        shuffled = self.heroes.copy()
        random.shuffle(shuffled)
        
        return {
            "ally_picks": shuffled[:3],
            "enemy_picks": shuffled[3:6],
            "bans": shuffled[6:10],
        }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd mlbb_drafter
pytest tests/test_detection.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/detection/ tests/test_detection.py
git commit -m "feat: add dummy detector for MVP testing"
```

---

## Phase 4: WebSocket Server

### Task 4: WebSocket Server

**Files:**
- Create: `mlbb_drafter/server/websocket/__init__.py`
- Create: `mlbb_drafter/server/websocket/server.py`
- Create: `mlbb_drafter/tests/test_websocket.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_websocket.py
import pytest
import asyncio
import json
from server.websocket.server import WebSocketServer

def test_websocket_server_initializes():
    server = WebSocketServer()
    assert server is not None

def test_websocket_server_has_broadcast():
    server = WebSocketServer()
    assert hasattr(server, 'broadcast')
    assert callable(server.broadcast)

def test_websocket_server_starts():
    server = WebSocketServer(host="localhost", port=8766)
    assert server.host == "localhost"
    assert server.port == 8766
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd mlbb_drafter
pytest tests/test_websocket.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'server.websocket'"

- [ ] **Step 3: Write minimal implementation**

```python
# server/websocket/__init__.py
from .server import WebSocketServer

__all__ = ["WebSocketServer"]
```

```python
# server/websocket/server.py
"""WebSocket server for real-time dashboard updates."""
import asyncio
import json
from typing import Set, Dict, Any
import websockets
from websockets.server import serve

class WebSocketServer:
    """WebSocket server that broadcasts detection results to clients."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set = set()
        self._server = None
    
    async def handler(self, websocket):
        """Handle new WebSocket connection."""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        try:
            async for message in websocket:
                # Handle client messages if needed
                pass
        finally:
            self.clients.remove(websocket)
            print(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def broadcast(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients."""
        if not self.clients:
            return
        
        message = json.dumps(data)
        await asyncio.gather(
            *[client.send(message) for client in self.clients],
            return_exceptions=True
        )
    
    async def start(self):
        """Start the WebSocket server."""
        self._server = await serve(self.handler, self.host, self.port)
        print(f"WebSocket server started on ws://{self.host}:{self.port}")
        await self._server.wait_closed()
    
    def stop(self):
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd mlbb_drafter
pytest tests/test_websocket.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/websocket/ tests/test_websocket.py
git commit -m "feat: add WebSocket server for real-time updates"
```

---

## Phase 5: Web Dashboard

### Task 5: Web Dashboard

**Files:**
- Create: `mlbb_drafter/web_dashboard/index.html`
- Create: `mlbb_drafter/web_dashboard/app.js`
- Create: `mlbb_drafter/web_dashboard/style.css`

- [ ] **Step 1: Create index.html**

```html
<!-- web_dashboard/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLBB Drafter Dashboard</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>MLBB Drafter</h1>
            <div id="status" class="status disconnected">Disconnected</div>
        </header>
        
        <main>
            <section class="draft-panel">
                <h2>Current Draft</h2>
                <div class="draft-grid">
                    <div class="team ally">
                        <h3>Ally Team</h3>
                        <div id="ally-picks" class="hero-list"></div>
                    </div>
                    <div class="team enemy">
                        <h3>Enemy Team</h3>
                        <div id="enemy-picks" class="hero-list"></div>
                    </div>
                </div>
                
                <div class="bans-section">
                    <h3>Banned Heroes</h3>
                    <div id="bans" class="hero-list"></div>
                </div>
            </section>
            
            <section class="recommendations-panel">
                <h2>Recommendations</h2>
                <div class="rec-category">
                    <h3>Top Picks</h3>
                    <div id="top-picks" class="rec-list"></div>
                </div>
                <div class="rec-category">
                    <h3>Counter Picks</h3>
                    <div id="counter-picks" class="rec-list"></div>
                </div>
                <div class="rec-category">
                    <h3>Synergy Picks</h3>
                    <div id="synergy-picks" class="rec-list"></div>
                </div>
            </section>
        </main>
        
        <footer>
            <p>MLBB Drafter - Real-time Draft Assistant</p>
        </footer>
    </div>
    
    <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create style.css**

```css
/* web_dashboard/style.css */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #eee;
    min-height: 100vh;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 30px;
    padding-bottom: 20px;
    border-bottom: 2px solid #0f3460;
}

h1 {
    font-size: 2.5rem;
    color: #e94560;
}

.status {
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: bold;
}

.status.connected {
    background: #27ae60;
}

.status.disconnected {
    background: #e74c3c;
}

main {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 30px;
}

section {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    padding: 25px;
    backdrop-filter: blur(10px);
}

h2 {
    color: #e94560;
    margin-bottom: 20px;
    font-size: 1.5rem;
}

h3 {
    color: #0f3460;
    margin-bottom: 15px;
    font-size: 1.1rem;
}

.draft-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 20px;
}

.team {
    background: rgba(0, 0, 0, 0.2);
    padding: 15px;
    border-radius: 10px;
}

.hero-list {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.hero-tag {
    background: #0f3460;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.9rem;
}

.hero-tag.ally {
    background: #2980b9;
}

.hero-tag.enemy {
    background: #c0392b;
}

.hero-tag.ban {
    background: #7f8c8d;
}

.rec-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.rec-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(0, 0, 0, 0.2);
    padding: 12px 16px;
    border-radius: 8px;
}

.rec-item .hero-name {
    font-weight: bold;
}

.rec-item .win-rate {
    color: #27ae60;
    font-weight: bold;
}

footer {
    text-align: center;
    margin-top: 30px;
    padding-top: 20px;
    border-top: 2px solid #0f3460;
    color: #888;
}
```

- [ ] **Step 3: Create app.js**

```javascript
// web_dashboard/app.js
const WS_URL = `ws://${window.location.hostname}:8765`;

let ws = null;
let reconnectAttempts = 0;
const MAX_RECONNECTAttempts = 5;

function connect() {
    ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
        console.log('Connected to server');
        document.getElementById('status').textContent = 'Connected';
        document.getElementById('status').className = 'status connected';
        reconnectAttempts = 0;
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateDashboard(data);
    };
    
    ws.onclose = () => {
        console.log('Disconnected from server');
        document.getElementById('status').textContent = 'Disconnected';
        document.getElementById('status').className = 'status disconnected';
        
        if (reconnectAttempts < MAX_RECONNECTAttempts) {
            reconnectAttempts++;
            setTimeout(connect, 2000 * reconnectAttempts);
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

function updateDashboard(data) {
    // Update ally picks
    const allyPicks = document.getElementById('ally-picks');
    allyPicks.innerHTML = (data.ally_picks || [])
        .map(hero => `<span class="hero-tag ally">${hero}</span>`)
        .join('');
    
    // Update enemy picks
    const enemyPicks = document.getElementById('enemy-picks');
    enemyPicks.innerHTML = (data.enemy_picks || [])
        .map(hero => `<span class="hero-tag enemy">${hero}</span>`)
        .join('');
    
    // Update bans
    const bans = document.getElementById('bans');
    bans.innerHTML = (data.bans || [])
        .map(hero => `<span class="hero-tag ban">${hero}</span>`)
        .join('');
    
    // Update recommendations
    updateRecommendations('top-picks', data.recommendations?.top_picks || []);
    updateRecommendations('counter-picks', data.recommendations?.counter_picks || []);
    updateRecommendations('synergy-picks', data.recommendations?.synergy_picks || []);
}

function updateRecommendations(elementId, recommendations) {
    const element = document.getElementById(elementId);
    element.innerHTML = recommendations
        .map(rec => `
            <div class="rec-item">
                <span class="hero-name">${rec.hero}</span>
                <span class="win-rate">${(rec.win_rate * 100).toFixed(1)}%</span>
            </div>
        `)
        .join('');
}

// Connect on page load
connect();
```

- [ ] **Step 4: Test dashboard loads**

```bash
cd mlbb_drafter
python -m http.server 8080 --directory web_dashboard
# Open http://localhost:8080 in browser
```

Expected: Dashboard loads with "Disconnected" status

- [ ] **Step 5: Commit**

```bash
git add web_dashboard/
git commit -m "feat: add web dashboard with WebSocket client"
```

---

## Phase 6: Main Server Integration

### Task 6: Main Server Entry Point

**Files:**
- Create: `mlbb_drafter/server/main.py`
- Create: `mlbb_drafter/server/config.py`

- [ ] **Step 1: Create config.py**

```python
# server/config.py
"""Server configuration."""
import os

# ADB Configuration
ADB_HOST = os.getenv("ADB_HOST", "localhost")
ADB_PORT = int(os.getenv("ADB_PORT", "5037"))

# WebSocket Configuration
WS_HOST = os.getenv("WS_HOST", "0.0.0.0")
WS_PORT = int(os.getenv("WS_PORT", "8765"))

# Capture Configuration
CAPTURE_FPS = int(os.getenv("CAPTURE_FPS", "5"))
BUFFER_SIZE = 1

# Detection Configuration
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
USE_DUMMY_DETECTOR = os.getenv("USE_DUMMY_DETECTOR", "true").lower() == "true"

# Dashboard Configuration
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8080"))
```

- [ ] **Step 2: Create main.py**

```python
# server/main.py
"""Main server entry point."""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.config import WS_HOST, WS_PORT, CAPTURE_FPS, USE_DUMMY_DETECTOR
from server.capture import ADBCapture
from server.detection import DummyDetector
from server.websocket import WebSocketServer

class MLdrafterServer:
    """Main server that orchestrates capture, detection, and recommendations."""
    
    def __init__(self):
        self.capture = ADBCapture(buffer_size=1)
        self.detector = DummyDetector() if USE_DUMMY_DETECTOR else None
        self.ws_server = WebSocketServer(host=WS_HOST, port=WS_PORT)
        self.running = False
    
    async def process_frame(self, frame):
        """Process a single frame through the pipeline."""
        # Detect heroes
        detection_result = self.detector.detect(frame)
        
        # Prepare message for dashboard
        message = {
            "type": "draft_update",
            "ally_picks": detection_result["ally_picks"],
            "enemy_picks": detection_result["enemy_picks"],
            "bans": detection_result["bans"],
            "recommendations": {
                "top_picks": [],  # Will be filled by GCN later
                "counter_picks": [],
                "synergy_picks": [],
            }
        }
        
        # Broadcast to WebSocket clients
        await self.ws_server.broadcast(message)
    
    async def capture_loop(self):
        """Continuous capture and processing loop."""
        import time
        delay = 1.0 / CAPTURE_FPS
        
        print(f"Starting capture loop at {CAPTURE_FPS} FPS")
        while self.running:
            frame = self.capture.capture()
            if frame is not None:
                await self.process_frame(frame)
            await asyncio.sleep(delay)
    
    async def run(self):
        """Run the server."""
        self.running = True
        print("MLBB Drafter Server starting...")
        
        # Start WebSocket server and capture loop concurrently
        await asyncio.gather(
            self.ws_server.start(),
            self.capture_loop(),
        )
    
    def stop(self):
        """Stop the server."""
        self.running = False
        self.ws_server.stop()

def main():
    server = MLdrafterServer()
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        server.stop()
        print("Server stopped.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Test server starts**

```bash
cd mlbb_drafter
python -m server.main
```

Expected: Server starts, prints "WebSocket server started on ws://0.0.0.0:8765"

- [ ] **Step 4: Test end-to-end pipeline**

1. Start server: `python -m server.main`
2. Start dashboard: `python -m http.server 8080 --directory web_dashboard`
3. Open http://localhost:8080 in browser
4. Verify dashboard shows "Connected" and displays dummy hero detections

Expected: Dashboard receives updates with random hero picks

- [ ] **Step 5: Commit**

```bash
git add server/main.py server/config.py
git commit -m "feat: add main server with capture-detect-broadcast pipeline"
```

---

## Phase 7: GCN Recommendation Engine

### Task 7: GCN Model Implementation

**Files:**
- Create: `mlbb_drafter/server/recommendation/__init__.py`
- Create: `mlbb_drafter/server/recommendation/gcn_model.py`
- Create: `mlbb_drafter/server/recommendation/draft_state.py`
- Create: `mlbb_drafter/tests/test_recommendation.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_recommendation.py
import pytest
import torch
from server.recommendation.gcn_model import MOBARecGCN
from server.recommendation.draft_state import DraftState

def test_gcn_model_initializes():
    model = MOBARecGCN(num_heros=100, input_dim=64, hidden_dim=32, output_dim=16)
    assert model is not None

def test_gcn_model_forward():
    model = MOBARecGCN(num_heros=100, input_dim=64, hidden_dim=32, output_dim=16)
    # Create dummy input: batch_size=1, num_nodes=10, features=64
    x = torch.randn(1, 10, 64)
    adj = torch.ones(1, 10, 10)
    output = model(x, adj)
    assert output.shape == (1, 100)

def test_draft_state_initializes():
    state = DraftState()
    assert state is not None

def test_draft_state_add_pick():
    state = DraftState()
    state.add_pick("Lancelot", is_ally=True)
    assert "Lancelot" in state.ally_picks

def test_draft_state_add_ban():
    state = DraftState()
    state.add_ban("Fanny")
    assert "Fanny" in state.bans
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd mlbb_drafter
pytest tests/test_recommendation.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'server.recommendation'"

- [ ] **Step 3: Write minimal implementation**

```python
# server/recommendation/__init__.py
from .gcn_model import MOBARecGCN
from .draft_state import DraftState

__all__ = ["MOBARecGCN", "DraftState"]
```

```python
# server/recommendation/gcn_model.py
"""MOBARec-GCNFP: Lightweight GCN for MOBA draft recommendation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """Single graph convolution layer."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Graph convolution: H' = σ(A H W + b)"""
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output + self.bias

class MOBARecGCN(nn.Module):
    """MOBARec-GCNFP model for draft recommendation.
    
    Uses single-layer GCN with dynamic match embedding initialization:
    em0 = Σ(efriendly) - Σ(eopponent)
    """
    
    def __init__(
        self,
        num_heros: int = 100,
        input_dim: int = 64,
        hidden_dim: int = 32,
        output_dim: int = 16
    ):
        super().__init__()
        self.num_heros = num_heros
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Hero embedding layer
        self.hero_embedding = nn.Embedding(num_heros, input_dim)
        
        # GCN layers (single layer as per spec)
        self.gcn = GraphConvolution(input_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Win rate prediction head
        self.win_rate_head = nn.Linear(output_dim, 1)
    
    def compute_match_embedding(
        self,
        ally_hero_ids: torch.Tensor,
        enemy_hero_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute dynamic match embedding.
        
        em0 = Σ(efriendly) - Σ(eopponent)
        """
        ally_embeds = self.hero_embedding(ally_hero_ids)
        enemy_embeds = self.hero_embedding(enemy_hero_ids)
        
        # Sum pooling for each team
        ally_sum = ally_embeds.sum(dim=0)
        enemy_sum = enemy_embeds.sum(dim=0)
        
        # Dynamic match embedding
        match_embed = ally_sum - enemy_sum
        return match_embed
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        ally_ids: torch.Tensor = None,
        enemy_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features [batch, num_nodes, input_dim]
            adj: Adjacency matrix [batch, num_nodes, num_nodes]
            ally_ids: Hero IDs for ally team
            enemy_ids: Hero IDs for enemy team
            
        Returns:
            Win rate predictions [batch, num_heros]
        """
        # Graph convolution
        h = F.relu(self.gcn(x, adj))
        
        # Pool to graph-level representation
        h_pooled = h.mean(dim=1)
        
        # Project to output space
        output = self.output_proj(h_pooled)
        
        # Predict win rates for all heroes
        win_rates = self.win_rate_head(output)
        
        return win_rates.squeeze(-1)
    
    def recommend(
        self,
        current_draft: dict,
        available_heroes: list,
        top_k: int = 5
    ) -> list:
        """Get top-k hero recommendations.
        
        Args:
            current_draft: Dict with ally_picks, enemy_picks, bans
            available_heroes: List of available hero IDs
            top_k: Number of recommendations
            
        Returns:
            List of (hero_id, win_rate) tuples
        """
        self.eval()
        with torch.no_grad():
            # Create dummy adjacency (full graph for now)
            num_nodes = len(available_heroes) + 10  # heroes + draft nodes
            x = torch.randn(1, num_nodes, self.input_dim)
            adj = torch.ones(1, num_nodes, num_nodes) / num_nodes
            
            # Get win rates
            win_rates = self.forward(x, adj)
            
            # Get top-k
            _, top_indices = torch.topk(win_rates, min(top_k, len(available_heroes)))
            
            recommendations = [
                (available_heroes[idx], win_rates[idx].item())
                for idx in top_indices
            ]
            
            return recommendations
```

```python
# server/recommendation/draft_state.py
"""Track the current draft state."""
from typing import List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class DraftState:
    """Tracks picks and bans in current draft."""
    
    ally_picks: List[str] = field(default_factory=list)
    enemy_picks: List[str] = field(default_factory=list)
    bans: List[str] = field(default_factory=list)
    phase: str = "ban"  # "ban" or "pick"
    
    def add_pick(self, hero: str, is_ally: bool):
        """Add a hero pick."""
        if is_ally:
            self.ally_picks.append(hero)
        else:
            self.enemy_picks.append(hero)
    
    def add_ban(self, hero: str):
        """Add a hero ban."""
        self.bans.append(hero)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "ally_picks": self.ally_picks,
            "enemy_picks": self.enemy_picks,
            "bans": self.bans,
            "phase": self.phase,
        }
    
    def is_complete(self) -> bool:
        """Check if draft is complete (5 picks per team)."""
        return len(self.ally_picks) >= 5 and len(self.enemy_picks) >= 5
    
    def reset(self):
        """Reset draft state."""
        self.ally_picks.clear()
        self.enemy_picks.clear()
        self.bans.clear()
        self.phase = "ban"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd mlbb_drafter
pytest tests/test_recommendation.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/recommendation/ tests/test_recommendation.py
git commit -m "feat: add MOBARec-GCNFP model and draft state tracker"
```

---

## Phase 8: Hero Metadata & Data Loader

### Task 8: Hero Metadata and Data Loader

**Files:**
- Create: `mlbb_drafter/shared/hero_meta.json`
- Create: `mlbb_drafter/server/data/__init__.py`
- Create: `mlbb_drafter/server/data/loader.py`

- [ ] **Step 1: Create hero_meta.json (partial - 20 heroes for testing)**

```json
// shared/hero_meta.json
{
  "heroes": [
    {"id": 1, "name": "Lancelot", "role": "Assassin", "lanes": ["Jungle"]},
    {"id": 2, "name": "Gusion", "role": "Assassin", "lanes": ["Jungle"]},
    {"id": 3, "name": "Fanny", "role": "Assassin", "lanes": ["Jungle"]},
    {"id": 4, "name": "Hayabusa", "role": "Assassin", "lanes": ["Jungle"]},
    {"id": 5, "name": "Ling", "role": "Assassin", "lanes": ["Jungle"]},
    {"id": 6, "name": "Chou", "role": "Fighter", "lanes": ["EXP", "Roam"]},
    {"id": 7, "name": "Alucard", "role": "Fighter", "lanes": ["Jungle"]},
    {"id": 8, "name": "Selena", "role": "Assassin", "lanes": ["Mid", "Roam"]},
    {"id": 9, "name": "Kagura", "role": "Mage", "lanes": ["Mid"]},
    {"id": 10, "name": "Harith", "role": "Mage", "lanes": ["Mid"]},
    {"id": 11, "name": "Valir", "role": "Mage", "lanes": ["Mid"]},
    {"id": 12, "name": "Lunox", "role": "Mage", "lanes": ["Mid"]},
    {"id": 13, "name": "Esmeralda", "role": "Mage", "lanes": ["EXP"]},
    {"id": 14, "name": "Yu Zhong", "role": "Fighter", "lanes": ["EXP"]},
    {"id": 15, "name": "Benedetta", "role": "Assassin", "lanes": ["EXP"]},
    {"id": 16, "name": "Atlas", "role": "Tank", "lanes": ["Roam"]},
    {"id": 17, "name": "Khufra", "role": "Tank", "lanes": ["Roam"]},
    {"id": 18, "name": "Akai", "role": "Tank", "lanes": ["Roam"]},
    {"id": 19, "name": "Grock", "role": "Tank", "lanes": ["Roam"]},
    {"id": 20, "name": "Johnson", "role": "Tank", "lanes": ["Roam"]}
  ]
}
```

- [ ] **Step 2: Create loader.py**

```python
# server/data/__init__.py
from .loader import HeroDataLoader

__all__ = ["HeroDataLoader"]
```

```python
# server/data/loader.py
"""Load hero metadata and tournament data."""
import json
import os
from typing import Dict, List, Optional
from pathlib import Path

class HeroDataLoader:
    """Load and manage hero metadata."""
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Default to shared directory
            data_dir = Path(__file__).parent.parent.parent / "shared"
        self.data_dir = Path(data_dir)
        self.heroes: List[Dict] = []
        self.hero_by_name: Dict[str, Dict] = {}
        self.hero_by_id: Dict[int, Dict] = {}
    
    def load_hero_meta(self) -> List[Dict]:
        """Load hero metadata from JSON file."""
        meta_path = self.data_dir / "hero_meta.json"
        
        if not meta_path.exists():
            print(f"Warning: hero_meta.json not found at {meta_path}")
            return []
        
        with open(meta_path, "r") as f:
            data = json.load(f)
        
        self.heroes = data.get("heroes", [])
        
        # Build lookup dictionaries
        for hero in self.heroes:
            self.hero_by_name[hero["name"]] = hero
            self.hero_by_id[hero["id"]] = hero
        
        print(f"Loaded {len(self.heroes)} heroes")
        return self.heroes
    
    def get_hero_by_name(self, name: str) -> Optional[Dict]:
        """Get hero data by name."""
        return self.hero_by_name.get(name)
    
    def get_hero_by_id(self, hero_id: int) -> Optional[Dict]:
        """Get hero data by ID."""
        return self.hero_by_id.get(hero_id)
    
    def get_all_hero_names(self) -> List[str]:
        """Get list of all hero names."""
        return [h["name"] for h in self.heroes]
    
    def get_heroes_by_role(self, role: str) -> List[Dict]:
        """Get all heroes with a specific role."""
        return [h for h in self.heroes if h["role"] == role]
    
    def get_available_heroes(self, picks: List[str], bans: List[str]) -> List[Dict]:
        """Get heroes not yet picked or banned."""
        excluded = set(picks + bans)
        return [h for h in self.heroes if h["name"] not in excluded]
```

- [ ] **Step 3: Test loader**

```bash
cd mlbb_drafter
python -c "
from server.data.loader import HeroDataLoader
loader = HeroDataLoader()
loader.load_hero_meta()
print('Heroes loaded:', len(loader.heroes))
print('Sample hero:', loader.get_hero_by_name('Lancelot'))
"
```

Expected: Prints hero data

- [ ] **Step 4: Commit**

```bash
git add server/data/ shared/hero_meta.json
git commit -m "feat: add hero metadata and data loader"
```

---

## Phase 9: Full Pipeline Integration

### Task 9: Integrate GCN into Main Server

**Files:**
- Modify: `mlbb_drafter/server/main.py`

- [ ] **Step 1: Update main.py to use GCN**

```python
# server/main.py (add to imports)
from server.recommendation import MOBARecGCN, DraftState
from server.data import HeroDataLoader

# Add to MLdrafterServer.__init__:
def __init__(self):
    # ... existing code ...
    self.draft_state = DraftState()
    self.hero_loader = HeroDataLoader()
    self.hero_loader.load_hero_meta()
    self.gcn_model = MOBARecGCN(num_heros=len(self.hero_loader.heroes))

# Update process_frame method:
async def process_frame(self, frame):
    """Process a single frame through the pipeline."""
    # Detect heroes
    detection_result = self.detector.detect(frame)
    
    # Update draft state
    for hero in detection_result.get("ally_picks", []):
        if hero not in self.draft_state.ally_picks:
            self.draft_state.add_pick(hero, is_ally=True)
    
    for hero in detection_result.get("enemy_picks", []):
        if hero not in self.draft_state.enemy_picks:
            self.draft_state.add_pick(hero, is_ally=False)
    
    for hero in detection_result.get("bans", []):
        if hero not in self.draft_state.bans:
            self.draft_state.add_ban(hero)
    
    # Get recommendations if draft not complete
    recommendations = {"top_picks": [], "counter_picks": [], "synergy_picks": []}
    
    if not self.draft_state.is_complete():
        available = self.hero_loader.get_available_heroes(
            self.draft_state.ally_picks + self.draft_state.enemy_picks,
            self.draft_state.bans
        )
        
        if available:
            # Get GCN recommendations
            recs = self.gcn_model.recommend(
                self.draft_state.to_dict(),
                [h["name"] for h in available],
                top_k=3
            )
            
            recommendations["top_picks"] = [
                {"hero": hero, "win_rate": wr}
                for hero, wr in recs
            ]
    
    # Prepare message for dashboard
    message = {
        "type": "draft_update",
        "ally_picks": self.draft_state.ally_picks,
        "enemy_picks": self.draft_state.enemy_picks,
        "bans": self.draft_state.bans,
        "recommendations": recommendations,
        "draft_complete": self.draft_state.is_complete(),
    }
    
    # Broadcast to WebSocket clients
    await self.ws_server.broadcast(message)
```

- [ ] **Step 2: Test full pipeline**

```bash
cd mlbb_drafter
python -m server.main
```

Expected: Server starts, processes frames, broadcasts recommendations

- [ ] **Step 3: Verify dashboard receives recommendations**

1. Start server: `python -m server.main`
2. Start dashboard: `python -m http.server 8080 --directory web_dashboard`
3. Open http://localhost:8080
4. Verify recommendations section shows hero suggestions

Expected: Dashboard displays picks, bans, and recommendations

- [ ] **Step 4: Commit**

```bash
git add server/main.py
git commit -m "feat: integrate GCN recommendations into main pipeline"
```

---

## Phase 10: YOLO Training Pipeline (Future)

### Task 10: YOLO Training Script (Placeholder)

**Files:**
- Create: `mlbb_drafter/training/train_yolo.py`

- [ ] **Step 1: Create training script skeleton**

```python
# training/train_yolo.py
"""YOLOv8n-OBB training pipeline for MLBB hero detection."""
import os
from pathlib import Path

def train_yolo(
    data_yaml: str,
    epochs: int = 100,
    img_size: int = 640,
    batch_size: int = 16
):
    """Train YOLOv8n-OBB model.
    
    Args:
        data_yaml: Path to dataset YAML file
        epochs: Number of training epochs
        img_size: Input image size
        batch_size: Batch size
    """
    from ultralytics import YOLO
    
    # Load pretrained model
    model = YOLO("yolov8n-obb.pt")
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name="mlbb_hero_detector"
    )
    
    print(f"Training complete. Results saved to runs/detect/mlbb_hero_detector")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data YAML")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    
    train_yolo(args.data, args.epochs, args.img_size, args.batch_size)
```

- [ ] **Step 2: Commit**

```bash
git add training/train_yolo.py
git commit -m "feat: add YOLO training script skeleton"
```

---

## Summary

**Completed:**
- [x] Project scaffolding with monorepo structure
- [x] ADB screen capture module
- [x] Dummy YOLO detector for MVP
- [x] WebSocket server for real-time updates
- [x] Web dashboard with live updates
- [x] Main server integrating capture → detect → broadcast pipeline
- [x] MOBARec-GCNFP model with dynamic match embedding
- [x] Draft state tracker
- [x] Hero metadata and data loader
- [x] Full pipeline integration with recommendations

**Next Steps (Future Tasks):**
- [ ] Implement Liquipedia scraper for live meta data
- [ ] Train YOLOv8n-OBB on synthetic + real screenshots
- [ ] Integrate MLBB-API for hero statistics
- [ ] Build Android overlay app (Kotlin)
- [ ] Optimize GCN training on real tournament data
