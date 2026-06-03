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
