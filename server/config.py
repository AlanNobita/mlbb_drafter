"""Server configuration."""
import os

# ADB Configuration
ADB_HOST = os.getenv("ADB_HOST", "localhost")
ADB_PORT = int(os.getenv("ADB_PORT", "5037"))
ADB_DEVICE = os.getenv("ADB_DEVICE", "")  # e.g., "192.168.1.50:5555" for wireless

# WebSocket Configuration
WS_HOST = os.getenv("WS_HOST", "0.0.0.0")
WS_PORT = int(os.getenv("WS_PORT", "8765"))

# Capture Configuration
CAPTURE_FPS = int(os.getenv("CAPTURE_FPS", "5"))
BUFFER_SIZE = 1

# Detection Configuration
DETECTOR_TYPE = os.getenv("DETECTOR_TYPE", "dummy")  # "dummy" or "yolo"
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
USE_DUMMY_DETECTOR = os.getenv("USE_DUMMY_DETECTOR", "true").lower() == "true"

# YOLO Configuration
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "runs/train/mlbb_hero_detect/weights/best.pt")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.5"))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu")

# Dashboard Configuration
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8080"))
