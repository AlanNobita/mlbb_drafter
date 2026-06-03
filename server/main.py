"""Main server entry point."""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.config import (
    WS_HOST, WS_PORT, CAPTURE_FPS, USE_DUMMY_DETECTOR,
    DETECTOR_TYPE, YOLO_MODEL_PATH, YOLO_CONFIDENCE, YOLO_DEVICE,
    ADB_DEVICE,
)
from server.capture import ADBCapture
from server.detection import DummyDetector, YOLODetector
from server.websocket import WebSocketServer
from server.recommendation import MOBARecGCN, DraftState
from server.data import HeroDataLoader


class MLdrafterServer:
    """Main server that orchestrates capture, detection, and recommendations."""
    
    def __init__(self):
        self.capture = ADBCapture(buffer_size=1, device_serial=ADB_DEVICE or None)
        self.detector = self._create_detector()
        self.ws_server = WebSocketServer(host=WS_HOST, port=WS_PORT)
        self.running = False
        self._processing = False  # Frame dropping flag
        
        # Initialize data loader and GCN model
        self.hero_loader = HeroDataLoader()
        self.hero_loader.load_hero_meta()
        self.gcn_model = MOBARecGCN(num_heros=len(self.hero_loader.heroes))
        
        # Draft state tracker
        self.draft_state = DraftState()
    
    def _create_detector(self):
        """Create detector based on DETECTOR_TYPE config."""
        if DETECTOR_TYPE == "yolo":
            print(f"Using YOLO detector: {YOLO_MODEL_PATH} (conf={YOLO_CONFIDENCE}, device={YOLO_DEVICE})")
            return YOLODetector(
                model_path=YOLO_MODEL_PATH,
                confidence_threshold=YOLO_CONFIDENCE,
                device=YOLO_DEVICE,
            )
        if USE_DUMMY_DETECTOR or DETECTOR_TYPE == "dummy":
            print("Using dummy detector")
            return DummyDetector()
        raise ValueError(f"Unknown DETECTOR_TYPE: {DETECTOR_TYPE!r}. Use 'dummy' or 'yolo'.")
    
    async def process_frame(self, frame):
        """Process a single frame through the pipeline."""
        try:
            # Detect heroes
            if isinstance(self.detector, YOLODetector):
                detection_result = self.detector.detect_by_position(frame, frame.shape[1], frame.shape[0])
                # Extract hero role names from YOLO dicts
                for key in ("ally_picks", "enemy_picks", "bans"):
                    detection_result[key] = [
                        entry["role"] if isinstance(entry, dict) else entry
                        for entry in detection_result.get(key, [])
                    ]
            else:
                detection_result = self.detector.detect(frame)
            
            # Update draft state from detection
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
        except Exception as e:
            print(f"Error processing frame: {e}")
    
    async def capture_loop(self):
        """Continuous capture and processing loop with frame dropping."""
        delay = 1.0 / CAPTURE_FPS
        loop = asyncio.get_event_loop()
        
        print(f"Starting capture loop at {CAPTURE_FPS} FPS")
        while self.running:
            # Skip frame if previous still processing (frame dropping)
            if self._processing:
                await asyncio.sleep(delay)
                continue
            
            # Run blocking ADB capture in thread executor
            frame = await loop.run_in_executor(None, self.capture.capture)
            
            if frame is not None:
                self._processing = True
                try:
                    await self.process_frame(frame)
                finally:
                    self._processing = False
            
            await asyncio.sleep(delay)
    
    async def run(self):
        """Run the server."""
        self.running = True
        print("MLBB Drafter Server starting...")
        print(f"Loaded {len(self.hero_loader.heroes)} heroes")
        
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
