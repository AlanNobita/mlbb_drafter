"""Tests for ADB capture module."""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def test_adbcapture_with_device_serial():
    capture = ADBCapture(device_serial="192.168.1.50:5555")
    assert capture.device_serial == "192.168.1.50:5555"


def test_adbcapture_default_no_serial():
    capture = ADBCapture()
    assert capture.device_serial is None


def test_adbcapture_has_connect_method():
    capture = ADBCapture()
    assert hasattr(capture, 'connect')
    assert callable(capture.connect)


def test_adbcapture_has_disconnect_method():
    capture = ADBCapture()
    assert hasattr(capture, 'disconnect')
    assert callable(capture.disconnect)


def test_adbcapture_has_is_connected_method():
    capture = ADBCapture()
    assert hasattr(capture, 'is_connected')
    assert callable(capture.is_connected)


def test_adbcapture_connect_no_serial_returns_true():
    """Without device_serial, connect() returns True (assumes USB)."""
    capture = ADBCapture()
    assert capture.connect() is True
