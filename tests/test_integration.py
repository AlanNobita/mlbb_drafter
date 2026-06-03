import pytest
import asyncio
import json
import numpy as np
import torch
from pathlib import Path
from unittest.mock import AsyncMock, patch

from server.recommendation.gcn_model import MOBARecGCN
from server.recommendation.draft_state import DraftState
from server.data.loader import HeroDataLoader
from server.data.tournament_loader import TournamentDataLoader
from server.detection.dummy_detector import DummyDetector
import server.config as config


class TestEndToEndPipeline:
    """Integration tests for the full drafter pipeline."""

    def test_dummy_detector_produces_detections(self):
        """Test dummy detector returns valid detections."""
        detector = DummyDetector()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert "ally_picks" in result
        assert "enemy_picks" in result
        assert len(result["ally_picks"]) > 0
        assert len(result["enemy_picks"]) > 0

    def test_gcn_model_forward_pass(self):
        """Test GCN model produces valid output via legacy mode."""
        model = MOBARecGCN(num_heros=30, input_dim=128, hidden_dim=128, output_dim=64)
        x = torch.randn(1, 30, 128)
        adj = torch.ones(1, 30, 30) / 30

        output = model(x=x, adj=adj)

        assert output.shape == (1, 30)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gcn_recommend_returns_valid_heroes(self):
        """Test recommendation returns valid hero names and win rates."""
        model = MOBARecGCN(num_heros=30)
        hero_names = [f"Hero_{i}" for i in range(30)]
        available = hero_names[:20]

        draft = {
            "ally_picks": ["Hero_0", "Hero_1", "Hero_2"],
            "enemy_picks": ["Hero_5", "Hero_6"],
            "bans": ["Hero_10", "Hero_11"],
        }

        recs = model.recommend(draft, available, top_k=5)

        assert len(recs) == 5
        for hero_name, win_rate in recs:
            assert hero_name in available
            assert isinstance(win_rate, float)

    def test_draft_state_tracker_update(self):
        """Test draft state tracker processes picks correctly."""
        state = DraftState()

        state.add_pick("Lancelot", is_ally=True)
        state.add_pick("Gusion", is_ally=False)
        state.add_pick("Fanny", is_ally=True)
        state.add_ban("Chou")

        d = state.to_dict()
        assert len(d["ally_picks"]) == 2
        assert len(d["enemy_picks"]) == 1
        assert len(d["bans"]) == 1

    def test_draft_state_is_complete(self):
        """Test draft completeness check."""
        state = DraftState()
        assert not state.is_complete()

        for i in range(5):
            state.add_pick(f"Hero_a{i}", is_ally=True)
            state.add_pick(f"Hero_e{i}", is_ally=False)

        assert state.is_complete()

    def test_hero_metadata_loading(self):
        """Test hero metadata loads correctly."""
        loader = HeroDataLoader()
        heroes = loader.load_hero_meta()

        assert len(heroes) > 0

        hero = heroes[0]
        assert "id" in hero
        assert "name" in hero
        assert "role" in hero

    def test_config_detector_type_default(self):
        """Test default detector type is dummy."""
        assert config.DETECTOR_TYPE == "dummy"

    @pytest.mark.asyncio
    async def test_websocket_broadcast_format(self):
        """Test WebSocket broadcast message format."""
        from server.websocket.server import WebSocketServer

        server = WebSocketServer()

        # Create mock websocket
        mock_ws = AsyncMock()
        server.clients.add(mock_ws)

        test_data = {
            "type": "draft_update",
            "ally_picks": ["Fighter", "Mage"],
            "enemy_picks": ["Assassin"],
            "bans": ["Tank"],
            "recommendations": [{"hero": "Support", "win_rate": 0.65}],
        }

        await server.broadcast(test_data)

        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "draft_update"
        assert len(sent_data["ally_picks"]) == 2


class TestAdjacencyMatrix:
    """Test adjacency matrix integration."""

    def test_adjacency_matrix_from_tournament_data(self):
        """Test adjacency matrix builds from tournament data."""
        loader = TournamentDataLoader()
        drafts = [
            {"friendly_picks": [0, 1, 2, 3, 4], "enemy_picks": [5, 6, 7, 8, 9], "bans": [], "win_rate": 0.6},
            {"friendly_picks": [0, 1, 5, 6, 7], "enemy_picks": [2, 3, 8, 9, 10], "bans": [], "win_rate": 0.5},
        ]

        cooc = loader.build_cooccurrence_matrix(drafts, num_heros=15)
        adj = loader.build_adjacency_tensor(cooc, threshold=0.1)

        assert adj.shape == (15, 15)
        assert adj[0][1] == 1.0  # Heroes 0 and 1 co-occur

    def test_gcn_model_set_adjacency_matrix(self):
        """Test GCN model accepts adjacency matrix without error."""
        model = MOBARecGCN(num_heros=30)

        adj = torch.zeros(30, 30)
        adj[0, 1] = 1.0
        adj[1, 0] = 1.0
        adj[2, 3] = 1.0
        adj[3, 2] = 1.0

        model.set_adjacency_matrix(adj)
        assert model._edge_index is not None

    def test_gcn_model_legacy_mode_with_adj(self):
        """Test GCN model works in legacy mode with dense adjacency."""
        model = MOBARecGCN(num_heros=30, input_dim=128, hidden_dim=128, output_dim=64)
        x = torch.randn(1, 30, 128)
        adj = torch.ones(1, 30, 30) / 30

        output = model(x=x, adj=adj)

        assert output.shape == (1, 30)
        assert not torch.isnan(output).any()


class TestDataPipeline:
    """Test data loading and processing pipeline."""

    def test_synthetic_data_loading(self):
        """Test synthetic draft data loads correctly."""
        csv_path = Path("training/data/synthetic_drafts.csv")
        if csv_path.exists():
            loader = TournamentDataLoader()
            drafts = loader.load_synthetic_drafts()
            assert len(drafts) == 10000
            assert "friendly_picks" in drafts[0]
            assert "win_rate" in drafts[0]

    def test_adjacency_matrix_saving(self):
        """Test adjacency matrix saves and loads correctly."""
        adj = torch.eye(10)
        save_path = Path("/tmp/test_adj.pt")
        torch.save(adj, save_path)

        loaded = torch.load(save_path, weights_only=True)
        assert torch.equal(adj, loaded)
        save_path.unlink()

    def test_draft_state_reset(self):
        """Test draft state reset clears all picks and bans."""
        state = DraftState()
        state.add_pick("Hero1", is_ally=True)
        state.add_pick("Hero2", is_ally=False)
        state.add_ban("Hero3")

        state.reset()
        d = state.to_dict()

        assert len(d["ally_picks"]) == 0
        assert len(d["enemy_picks"]) == 0
        assert len(d["bans"]) == 0

    def test_full_pipeline_flow(self):
        """Test capture -> detect -> recommend -> state update flow."""
        detector = DummyDetector()
        model = MOBARecGCN(num_heros=30)
        state = DraftState()

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        for hero in detections["ally_picks"]:
            state.add_pick(hero, is_ally=True)
        for hero in detections["enemy_picks"]:
            state.add_pick(hero, is_ally=False)

        all_detected = detections["ally_picks"] + detections["enemy_picks"] + detections["bans"]
        picked_banned = set(state.to_dict()["ally_picks"] + state.to_dict()["enemy_picks"] + state.to_dict()["bans"])
        available = [h for h in all_detected if h not in picked_banned]

        assert len(available) > 0

        recs = model.recommend(state.to_dict(), available, top_k=3)

        assert len(recs) > 0
        assert len(state.to_dict()["ally_picks"]) == 3
