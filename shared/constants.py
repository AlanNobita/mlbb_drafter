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
