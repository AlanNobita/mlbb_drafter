"""Tests for WebSocket server."""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
