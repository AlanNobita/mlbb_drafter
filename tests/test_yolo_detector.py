import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock ultralytics before importing
mock_ultralytics = MagicMock()
mock_yolo_class = MagicMock()
mock_ultralytics.YOLO = mock_yolo_class
sys.modules['ultralytics'] = mock_ultralytics

from server.detection.yolo_detector import YOLODetector

class TestYOLODetector:
    @patch('server.detection.yolo_detector.YOLO')
    def test_init_loads_model(self, mock_yolo):
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        detector = YOLODetector("model.pt", confidence_threshold=0.6)
        assert detector.model == mock_model
        assert detector.confidence_threshold == 0.6
    
    @patch('server.detection.yolo_detector.YOLO')
    def test_detect_empty(self, mock_yolo):
        mock_model = Mock()
        mock_result = Mock()
        mock_result.obb = None
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        detector = YOLODetector("model.pt")
        detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert detections == []
