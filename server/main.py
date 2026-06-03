"""Main server entry point."""
import asyncio
import sys
import os
import json
from pathlib import Path

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

import torch


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
        
        # Load trained GCN model
        self.gcn_model = self._load_gcn_model()
        
        # Draft state tracker
        self.draft_state = DraftState()
    
    def _load_gcn_model(self):
        """Load trained GCN model with weights and adjacency matrix."""
        model_path = Path(__file__).parent.parent / "training" / "data" / "gcn_model_v2.pt"
        
        if not model_path.exists():
            print(f"Warning: No trained model found at {model_path}")
            print("  Using untrained model (recommendations will be random)")
            num_heros = len(self.hero_loader.heroes)
            model = MOBARecGCN(num_heros=num_heros)
            self.hero_name_to_id = {}
            self.hero_features = None
            return model
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            
            num_heros = checkpoint.get("num_heros", 132)
            model = MOBARecGCN(
                num_heros=num_heros,
                input_dim=checkpoint.get("input_dim", 128),
                hidden_dim=checkpoint.get("hidden_dim", 128),
                output_dim=checkpoint.get("output_dim", 64),
                hero_feature_dim=checkpoint.get("hero_feature_dim", 8)
            )
            
            # Load trained weights
            model.load_state_dict(checkpoint["model_state_dict"])
            
            # Load adjacency matrix
            if "adjacency_matrix" in checkpoint:
                model.set_adjacency_matrix(checkpoint["adjacency_matrix"])
                print(f"  Loaded adjacency matrix: {checkpoint['adjacency_matrix'].shape}")
            
            # Store hero mapping and features for recommendations
            self.hero_name_to_id = checkpoint.get("hero_to_id", {})
            self.hero_features = checkpoint.get("hero_features", None)
            
            print(f"  Loaded trained GCN model ({num_heros} heroes, {sum(p.numel() for p in model.parameters()):,} params)")
            
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("  Falling back to untrained model")
            num_heros = len(self.hero_loader.heroes)
            model = MOBARecGCN(num_heros=num_heros)
            self.hero_name_to_id = {}
            self.hero_features = None
            return model
    
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
                        hero_name_to_id=self.hero_name_to_id,
                        hero_features=self.hero_features,
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
