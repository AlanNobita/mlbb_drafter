"""Main server entry point."""
import asyncio
import sys
import os
import json
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.config import (
    WS_HOST, WS_PORT,
    # CAPTURE_FPS, USE_DUMMY_DETECTOR,  # unused in manual mode
    # DETECTOR_TYPE, YOLO_MODEL_PATH, YOLO_CONFIDENCE, YOLO_DEVICE,
    # ADB_DEVICE, CAPTURE_TYPE, SYNTHETIC_FOLDER, SYNTHETIC_LOOP,
)
# from server.capture import ADBCapture, SyntheticCapture  # unused in manual mode
# from server.detection import DummyDetector, YOLODetector, DraftStateTracker, is_lobby_frame  # unused in manual mode
from server.websocket import WebSocketServer
from server.recommendation import DraftState
from server.recommendation.scoring import HeroScorer
from server.recommendation.scoring_data import ScoringData
from server.data import HeroDataLoader


class MLdrafterServer:
    """Main server: receives manual draft entry via WebSocket, sends recommendations.

    Manual mode only. Auto-detection (YOLO, ADB capture) is commented out below
    for future re-enable when real-game labeled data is available.
    """

    def __init__(self):
        # === AUTO-DETECTION (disabled in manual mode, re-enable for YOLO) ===
        # self.capture = self._create_capture()
        # self.detector = self._create_detector()
        # self._processing = False  # Frame dropping flag
        # self._manual_mode = False  # Set True after first manual entry, suppresses auto-detect

        self.ws_server = WebSocketServer(host=WS_HOST, port=WS_PORT)
        self.running = False

        # Initialize data loader and scoring engine
        self.hero_loader = HeroDataLoader()
        self.hero_loader.load_hero_meta()

        # New rule-based scoring engine (replaces GCN)
        self.scoring_data = ScoringData()
        self.scorer = HeroScorer(self.scoring_data)

        # Hero name list for available heroes
        self.hero_names = [h["name"] for h in self.hero_loader.heroes]

        # Draft state tracker
        self.draft_state = DraftState()
        # Sticky detection state tracker (3-frame confirmation) - unused in manual mode
        # self.state_tracker = DraftStateTracker(confirm_threshold=3)

        # Wire WebSocket callbacks
        self.ws_server.set_on_connect(self._build_hero_list_message)
        self.ws_server.set_on_message(self._handle_client_message)

    # === AUTO-DETECTION (disabled in manual mode) ===
    # def _create_detector(self):
    #     """Create detector based on DETECTOR_TYPE config."""
    #     if DETECTOR_TYPE == "yolo":
    #         print(f"Using YOLO detector: {YOLO_MODEL_PATH} (conf={YOLO_CONFIDENCE}, device={YOLO_DEVICE})")
    #         return YOLODetector(
    #             model_path=YOLO_MODEL_PATH,
    #             confidence_threshold=YOLO_CONFIDENCE,
    #             device=YOLO_DEVICE,
    #         )
    #     if USE_DUMMY_DETECTOR or DETECTOR_TYPE == "dummy":
    #         print("Using dummy detector")
    #         return DummyDetector()
    #     raise ValueError(f"Unknown DETECTOR_TYPE: {DETECTOR_TYPE!r}. Use 'dummy' or 'yolo'.")
    #
    # def _create_capture(self):
    #     """Create frame capture source based on CAPTURE_TYPE config.
    #
    #     Returns ADBCapture (real phone) or SyntheticCapture (folder of PNGs).
    #     """
    #     if CAPTURE_TYPE == "synthetic":
    #         print(f"Using synthetic capture from: {SYNTHETIC_FOLDER}")
    #         return SyntheticCapture(folder=SYNTHETIC_FOLDER, loop=SYNTHETIC_LOOP)
    #     print(f"Using ADB capture (device: {ADB_DEVICE or 'USB'})")
    #     return ADBCapture(buffer_size=1, device_serial=ADB_DEVICE or None)

    # === AUTO-DETECTION FRAME PIPELINE (disabled in manual mode) ===
    # async def process_frame(self, frame):
    #     """Process a single frame through the pipeline."""
    #     try:
    #         # Skip auto-detect when user is in manual entry mode
    #         if self._manual_mode:
    #             return
    #
    #         # Detect heroes
    #         if isinstance(self.detector, YOLODetector):
    #             detection_result = self.detector.detect_by_position(
    #                 frame, frame.shape[1], frame.shape[0]
    #             )
    #         else:
    #             # DummyDetector: convert to new format
    #             dummy_result = self.detector.detect(frame)
    #             detection_result = {
    #                 "ally_picks": [{"role": h, "confidence": 1.0, "slot": f"ally_pick_{i+1}"}
    #                                for i, h in enumerate(dummy_result.get("ally_picks", []))],
    #                 "enemy_picks": [{"role": h, "confidence": 1.0, "slot": f"enemy_pick_{i+1}"}
    #                                 for i, h in enumerate(dummy_result.get("enemy_picks", []))],
    #                 "ally_bans": [],
    #                 "enemy_bans": [{"role": h, "confidence": 1.0, "slot": f"enemy_ban_{i+1}"}
    #                                for i, h in enumerate(dummy_result.get("bans", []))],
    #                 "ignored": [],
    #             }
    #
    #         # Lobby filter: skip frames with too few detections
    #         if isinstance(self.detector, YOLODetector) and is_lobby_frame(detection_result, min_detections=3):
    #             return
    #
    #         # Update sticky state tracker (3-frame confirmation)
    #         self.state_tracker.update(detection_result)
    #
    #         # Pull confirmed heroes from tracker into draft_state
    #         confirmed = self.state_tracker.get_confirmed()
    #
    #         for slot, hero in confirmed.items():
    #             if slot.startswith("ally_ban_"):
    #                 if hero not in self.draft_state.bans:
    #                     self.draft_state.add_ban(hero)
    #             elif slot.startswith("enemy_ban_"):
    #                 if hero not in self.draft_state.bans:
    #                     self.draft_state.add_ban(hero)
    #             elif slot.startswith("ally_pick_"):
    #                 if hero not in self.draft_state.ally_picks:
    #                     self.draft_state.add_pick(hero, is_ally=True)
    #             elif slot.startswith("enemy_pick_"):
    #                 if hero not in self.draft_state.enemy_picks:
    #                     self.draft_state.add_pick(hero, is_ally=False)
    #
    #         # Get recommendations if draft not complete
    #         recommendations = {"top_picks": [], "counter_picks": [], "synergy_picks": []}
    #
    #         if not self.draft_state.is_complete():
    #             available_names = [
    #                 h["name"] for h in self.hero_loader.heroes
    #                 if h["name"] not in self.draft_state.ally_picks
    #                 and h["name"] not in self.draft_state.enemy_picks
    #                 and h["name"] not in self.draft_state.bans
    #             ]
    #
    #             if available_names:
    #                 # Determine needed lane
    #                 needed_lane = self._get_needed_lane(
    #                     self.draft_state.ally_picks
    #                 )
    #
    #                 # Get rule-based recommendations
    #                 recs = self.scorer.generate_recommendations(
    #                     available_heroes=available_names,
    #                     ally_picks=self.draft_state.ally_picks,
    #                     enemy_picks=self.draft_state.enemy_picks,
    #                     needed_lane=needed_lane,
    #                     enemy_bans=self.draft_state.bans,
    #                     top_k=3
    #                 )
    #
    #                 recommendations["top_picks"] = [
    #                     {"hero": hero, "score": score}
    #                     for hero, score in recs
    #                 ]
    #
    #                 # Counter picks: heroes that counter enemies
    #                 counter_picks = self._get_counter_picks(
    #                     self.draft_state.enemy_picks,
    #                     available_names
    #                 )
    #                 recommendations["counter_picks"] = counter_picks
    #
    #                 # Synergy picks: heroes that synergize with allies
    #                 synergy_picks = self._get_synergy_picks(
    #                     self.draft_state.ally_picks,
    #                     available_names
    #                 )
    #                 recommendations["synergy_picks"] = synergy_picks
    #
    #         # Prepare message for dashboard
    #         message = {
    #             "type": "draft_update",
    #             "ally_picks": self.draft_state.ally_picks,
    #             "enemy_picks": self.draft_state.enemy_picks,
    #             "bans": self.draft_state.bans,
    #             "recommendations": recommendations,
    #             "draft_complete": self.draft_state.is_complete(),
    #             "pending": self.state_tracker.get_pending(),
    #             "progress": self.state_tracker.get_progress(),
    #         }
    #
    #         # Broadcast to WebSocket clients
    #         await self.ws_server.broadcast(message)
    #     except Exception as e:
    #         print(f"Error processing frame: {e}")

    # async def capture_loop(self):
    #     """Continuous capture and processing loop with frame dropping."""
    #     delay = 1.0 / CAPTURE_FPS
    #     loop = asyncio.get_event_loop()
    #
    #     print(f"Starting capture loop at {CAPTURE_FPS} FPS")
    #     while self.running:
    #         # Skip frame if previous still processing (frame dropping)
    #         if self._processing:
    #             await asyncio.sleep(delay)
    #             continue
    #
    #         # Run blocking ADB capture in thread executor
    #         frame = await loop.run_in_executor(None, self.capture.capture)
    #
    #         if frame is not None:
    #             self._processing = True
    #             try:
    #                 await self.process_frame(frame)
    #             finally:
    #                 self._processing = False
    #
    #         await asyncio.sleep(delay)

    async def run(self):
        """Run the server (manual mode: WebSocket only)."""
        self.running = True
        print("MLBB Drafter Server starting (MANUAL MODE)...")
        print(f"Loaded {len(self.hero_loader.heroes)} heroes")
        print("Open http://localhost:8080 in your browser to use the dashboard.")

        # Manual mode: only the WebSocket server runs (no capture loop)
        await self.ws_server.start()

    def stop(self):
        """Stop the server."""
        self.running = False
        self.ws_server.stop()
    
    def _load_counter_data(self):
        """Load counter data from API."""
        counter_path = Path(__file__).parent.parent / "training" / "data" / "api_data" / "hero_counters_openmlbb.json"
        if counter_path.exists():
            with open(counter_path) as f:
                return json.load(f)
        return {}

    def _build_hero_list_message(self) -> dict:
        """Build initial WS message: hero list for client autocomplete."""
        return {"type": "hero_list", "heroes": self.hero_names}

    def _handle_client_message(self, data: dict) -> None:
        """Handle inbound WebSocket messages from dashboard clients.

        Supported message types:
          - manual_entry: replace draft_state with manually-entered picks/bans
          - reset_draft: clear all state

        Picks in manual_entry can be either:
          - a list of hero names: ["Tigreal", "Lancelot", ...]
          - a list of {hero, lane?} dicts: [{"hero": "Freya", "lane": "exp"}, ...]
        The lane field is forwarded to the scoring engine so flex heroes
        (e.g. Freya playing EXP, not Jungle) don't poison the lane-taken check.
        """
        msg_type = data.get("type")
        if msg_type == "manual_entry":
            self._apply_manual_entry(
                ally_picks=data.get("ally_picks", []),
                enemy_picks=data.get("enemy_picks", []),
                bans=data.get("bans", []),
                lane_needed=data.get("lane_needed"),
            )
        elif msg_type == "reset_draft":
            self.draft_state.reset()
            # self.state_tracker.reset()  # unused in manual mode
            # self._manual_mode = False   # unused in manual mode
        else:
            print(f"Unknown WS message type: {msg_type!r}")

    def _normalize_picks(self, picks: list) -> list:
        """Coerce pick entries into (hero, lane|None) tuples.

        Accepts either a hero name string or a {"hero": ..., "lane": ...} dict.
        """
        out = []
        for p in picks:
            if isinstance(p, str):
                if p:
                    out.append((p, None))
            elif isinstance(p, dict):
                hero = p.get("hero")
                if hero:
                    out.append((hero, p.get("lane")))
        return out

    def _apply_manual_entry(
        self,
        ally_picks: list,
        enemy_picks: list,
        bans: list,
        lane_needed: Optional[str] = None,
    ) -> None:
        """Replace draft state with manual entry and broadcast recommendations."""
        self.draft_state.reset()
        for hero, lane in self._normalize_picks(ally_picks):
            self.draft_state.add_pick(hero, is_ally=True, lane=lane)
        for hero, lane in self._normalize_picks(enemy_picks):
            self.draft_state.add_pick(hero, is_ally=False, lane=lane)
        for h in bans:
            if h:
                self.draft_state.add_ban(h)
        # Schedule broadcast (we're not in async context here)
        asyncio.get_event_loop().create_task(
            self._broadcast_recommendations(lane_needed=lane_needed)
        )

    async def _broadcast_recommendations(self, lane_needed: Optional[str] = None) -> None:
        """Compute and broadcast per-lane recommendations for current draft_state.

        Returns recommendations shaped for the draft-ace frontend:
            recommendations: { gold: { heroes: [...], fallbackActive: bool }, ... }
        Each hero has: { heroId, heroName, composite_score, breakdown: {synergy, counter, laneMatch}, fallback? }
        """
        LANES = ["gold", "mid", "exp", "jungle", "roam"]
        LANE_INTERNAL = {
            "gold": "Gold",
            "mid": "Mid",
            "exp": "EXP",
            "jungle": "Jungle",
            "roam": "Roam",
        }

        recommendations: dict = {
            lane: {"heroes": [], "fallbackActive": False} for lane in LANES
        }

        if not self.draft_state.is_complete():
            available_names = [
                h["name"] for h in self.hero_loader.heroes
                if h["name"] not in self.draft_state.ally_heroes()
                and h["name"] not in self.draft_state.enemy_heroes()
                and h["name"] not in self.draft_state.bans
            ]
            if available_names:
                lanes_to_compute = (
                    [lane_needed] if lane_needed in LANES else LANES
                )
                # Pass through the lanes the user has explicitly locked.
                # Scoring engine will use this to decide if a lane is "taken"
                # (so Freya on EXP doesn't make the scorer think Jungle is taken).
                locked_ally = self.draft_state.ally_lane_set()
                locked_enemy = self.draft_state.enemy_lane_set()
                for lane_key in lanes_to_compute:
                    lane_internal = LANE_INTERNAL[lane_key]
                    recs = self.scorer.generate_recommendations(
                        available_heroes=available_names,
                        ally_picks=self.draft_state.ally_heroes(),
                        enemy_picks=self.draft_state.enemy_heroes(),
                        needed_lane=lane_internal,
                        enemy_bans=self.draft_state.bans,
                        top_k=5,
                        locked_ally_lanes=locked_ally,
                        locked_enemy_lanes=locked_enemy,
                    )
                    heroes = []
                    ally_set = set(self.draft_state.ally_heroes())
                    enemy_set = set(self.draft_state.enemy_heroes())
                    for hero, score in recs:
                        breakdown = self._score_breakdown(
                            hero, lane_internal,
                            locked_ally_lanes=locked_ally,
                            locked_enemy_lanes=locked_enemy,
                        )
                        heroes.append(
                            self._build_recommendation_dict(
                                hero=hero,
                                score=score,
                                breakdown=breakdown,
                                ally_set=ally_set,
                                enemy_set=enemy_set,
                            )
                        )
                    recommendations[lane_key] = {
                        "heroes": heroes,
                        "fallbackActive": len(heroes) == 0,
                    }

        # Map picks/bans to draft-ace GameStatePayload shape.
        # The new PickSlot shape is {lane, heroName} — one slot per lane,
        # in the same order as the recommendation columns so the user's
        # eye can flow left-to-right.
        SLOT_ORDER = ["gold", "mid", "exp", "jungle", "roam"]

        def picks_for_team(pairs: list) -> list:
            by_lane = {lane: hero for hero, lane in pairs}
            return [
                {"lane": lane, "heroName": by_lane.get(lane)}
                for lane in SLOT_ORDER
            ]

        def bans_for_team(team: str) -> list:
            # All bans live in a single global list; split first 5 / last 5 by team
            offset = 0 if team == "blue" else 5
            return [
                {
                    "index": offset + i,
                    "heroName": self.draft_state.bans[offset + i] if offset + i < len(self.draft_state.bans) else None,
                    "team": team,
                }
                for i in range(5)
            ]

        message = {
            "type": "draft_update",
            "state": {
                "blue": picks_for_team(self.draft_state.ally_picks),
                "red": picks_for_team(self.draft_state.enemy_picks),
                "bans": bans_for_team("blue") + bans_for_team("red"),
                "recommendations": recommendations,
                "phase": "Manual" if not self.draft_state.is_complete() else "Complete",
                "intents": {lane: "General" for lane in LANES},
            },
            "draft_complete": self.draft_state.is_complete(),
        }
        await self.ws_server.broadcast(message)

    def _build_recommendation_dict(
        self,
        hero: str,
        score: float,
        breakdown: dict,
        ally_set: set,
        enemy_set: set,
    ) -> dict:
        """Build the per-recommendation dict sent to the frontend.

        Filters counters / synergies / countered_by to the CURRENT picks
        so the tooltip only shows relationships with the live draft.
        Extracted for unit testing the filter logic.
        """
        countered_enemies = sorted(
            enemy for enemy in enemy_set
            if self.scoring_data.get_counter_confidence(hero, enemy)[0] > 0
        )
        synergy_with_allies = sorted(
            ally for ally in ally_set
            if ally in self.scoring_data.get_synergies(hero)
            or hero in self.scoring_data.get_synergies(ally)
        )
        countered_by = sorted(
            enemy for enemy in enemy_set
            if enemy in self.scoring_data.get_heroes_that_beat(hero)
        )
        return {
            "heroId": hero.lower().replace(" ", "_"),
            "heroName": hero,
            "composite_score": round(score, 3),
            "breakdown": breakdown,
            "counters": countered_enemies,
            "synergies": synergy_with_allies,
            "countered_by": countered_by,
        }

    def _score_breakdown(
        self, hero: str, lane: str,
        locked_ally_lanes: Optional[set] = None,
        locked_enemy_lanes: Optional[set] = None,
    ) -> dict:
        """Compute partial score breakdown for a hero in a given lane.

        Returns dict with synergy, counter, laneMatch components (each 0-1).
        Used to populate the recommendation breakdown UI.
        """
        # Approximate partial scores by calling score_hero with restricted input
        try:
            full = self.scorer.score_hero(
                hero, self.draft_state.ally_heroes(), self.draft_state.enemy_heroes(),
                lane, self.draft_state.bans,
                locked_ally_lanes=locked_ally_lanes,
                locked_enemy_lanes=locked_enemy_lanes,
            )
        except Exception:
            return {"synergy": 0.0, "counter": 0.0, "laneMatch": 0.0}

        # Lane match: 1.0 if hero can play this lane, 0.0 otherwise
        can_play = self.scorer._hero_can_play_lane(hero, lane)
        lane_match = 1.0 if can_play else 0.0

        # Split full score into synergy/counter contributions
        try:
            no_synergy = self.scorer.score_hero(
                hero, [], self.draft_state.enemy_heroes(), lane, self.draft_state.bans,
                locked_ally_lanes=locked_ally_lanes,
                locked_enemy_lanes=locked_enemy_lanes,
            )
            no_counter = self.scorer.score_hero(
                hero, self.draft_state.ally_heroes(), [], lane, self.draft_state.bans,
                locked_ally_lanes=locked_ally_lanes,
                locked_enemy_lanes=locked_enemy_lanes,
            )
        except Exception:
            no_synergy = no_counter = full

        synergy = max(0.0, min(1.0, full - no_synergy))
        counter = max(0.0, min(1.0, full - no_counter))
        return {
            "synergy": round(synergy, 3),
            "counter": round(counter, 3),
            "laneMatch": round(lane_match, 3),
        }
    
    def _get_needed_lane(self, ally_picks: list) -> str:
        """Determine which lane still needs to be filled.

        With the lane-organized UI, picks arrive as `[(hero, lane), ...]`
        where each lane is explicit. The "needed lane" is the first lane
        the user has not yet filled. Falls back to "Mid" if the team is
        full (which means the user wants recs for all lanes).
        """
        SLOT_ORDER = ["gold", "mid", "exp", "jungle", "roam"]
        LANE_INTERNAL = {
            "gold": "Gold",
            "mid": "Mid",
            "exp": "EXP",
            "jungle": "Jungle",
            "roam": "Roam",
        }
        filled_lanes = {lane for _, lane in ally_picks if lane}
        for lane_key in SLOT_ORDER:
            if lane_key not in filled_lanes:
                return LANE_INTERNAL[lane_key]
        return "Mid"

    def _get_counter_picks(self, enemy_picks: list, available: list) -> list:
        """Get counter pick recommendations based on enemy team.

        Uses scoring data with double-check validation:
        - HIGH confidence (both sources agree) = strong counter
        - MEDIUM/LOW confidence = possible counter
        """
        if not enemy_picks:
            return []

        available_names = {h["name"] if isinstance(h, dict) else h for h in available}

        counter_scores = []
        for hero in available_names:
            total_conf = 0.0
            countered_enemies = []
            for enemy in enemy_picks:
                conf, level = self.scoring_data.get_counter_confidence(hero, enemy)
                if conf > 0:
                    total_conf += conf
                    countered_enemies.append(f"{enemy}({level})")
            if total_conf > 0:
                counter_scores.append((hero, total_conf, countered_enemies))

        counter_scores.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "hero": name,
                "confidence": round(score, 2),
                "counters": enemies
            }
            for name, score, enemies in counter_scores[:3]
        ]

    def _get_synergy_picks(self, ally_picks: list, available: list) -> list:
        """Get synergy pick recommendations based on ally team.

        Uses scoring data merged from openmlbb_heroes, mlbb_io_overviews, and hero_meta.
        """
        if not ally_picks:
            return []

        available_names = {h["name"] if isinstance(h, dict) else h for h in available}

        synergy_scores = []
        for hero in available_names:
            matched_synergies = []
            for ally in ally_picks:
                if hero in self.scoring_data.get_synergies(ally):
                    matched_synergies.append(ally)
            if matched_synergies:
                synergy_scores.append((hero, len(matched_synergies), matched_synergies))

        synergy_scores.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "hero": name,
                "synergy_count": count,
                "synergizes_with": allies
            }
            for name, count, allies in synergy_scores[:3]
        ]


def main():
    server = MLdrafterServer()
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        server.stop()
        print("Server stopped.")


if __name__ == "__main__":
    main()
