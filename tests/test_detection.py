"""Tests for dummy detector."""
import pytest
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
