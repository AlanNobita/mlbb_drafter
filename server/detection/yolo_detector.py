from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YOLODetector:
    """YOLOv8n-OBB based hero icon detector."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, device: str = "cpu"):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        if YOLO is None:
            raise ImportError("ultralytics required: pip install ultralytics")
        try:
            self.model = YOLO(str(self.model_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")
    
    def detect(self, image: np.ndarray) -> List[dict]:
        """Detect hero icons. Returns list of {role, confidence, bbox, center, area_ratio}."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        results = self.model.predict(image, conf=self.confidence_threshold, device=self.device, verbose=False)
        detections = []
        h, w = image.shape[:2]
        image_area = h * w
        
        for result in results:
            if not hasattr(result, 'obb') or result.obb is None:
                continue
            obb = result.obb
            for i in range(len(obb)):
                xyxyxyxy = obb.xyxyxyxy[i].cpu().numpy()
                conf = float(obb.conf[i])
                cls_id = int(obb.cls[i])
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                
                cx = float(np.mean(xyxyxyxy[:, 0]))
                cy = float(np.mean(xyxyxyxy[:, 1]))
                x_min, y_min = xyxyxyxy.min(axis=0)
                x_max, y_max = xyxyxyxy.max(axis=0)
                bbox_area = float((x_max - x_min) * (y_max - y_min))
                area_ratio = bbox_area / image_area
                
                detections.append({
                    "role": cls_name.lower(),
                    "confidence": conf,
                    "bbox": xyxyxyxy.flatten().tolist(),
                    "center": [cx, cy],
                    "area_ratio": area_ratio
                })
        return detections
    
    def detect_by_position(self, image: np.ndarray, screen_width: int, screen_height: int) -> dict:
        """Categorize detections by screen position. Left=ally, Right=enemy, Center=bans."""
        detections = self.detect(image)
        mid_x = screen_width / 2
        margin = screen_width * 0.1
        result = {"ally_picks": [], "enemy_picks": [], "bans": []}
        for det in detections:
            cx = det["center"][0]
            entry = {"role": det["role"], "confidence": det["confidence"]}
            if cx < mid_x - margin:
                result["ally_picks"].append(entry)
            elif cx > mid_x + margin:
                result["enemy_picks"].append(entry)
            else:
                result["bans"].append(entry)
        return result
