import pytest
import json
import csv
from pathlib import Path
from server.data.tournament_loader import TournamentDataLoader

class TestTournamentDataLoader:
    def test_init(self):
        loader = TournamentDataLoader(data_dir="/tmp/test")
        assert loader.data_dir == Path("/tmp/test")

    def test_load_synthetic_drafts(self, tmp_path):
        # Create test CSV
        csv_file = tmp_path / "test_drafts.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["match_id", "friendly_picks", "enemy_picks", "bans", "win_rate"])
            writer.writeheader()
            writer.writerow({
                "match_id": 1,
                "friendly_picks": json.dumps([1, 2, 3, 4, 5]),
                "enemy_picks": json.dumps([6, 7, 8, 9, 10]),
                "bans": json.dumps([11, 12, 13, 14, 15, 16]),
                "win_rate": 0.65
            })

        loader = TournamentDataLoader(data_dir=str(tmp_path))
        drafts = loader.load_synthetic_drafts("test_drafts.csv")

        assert len(drafts) == 1
        assert drafts[0]["match_id"] == 1
        assert len(drafts[0]["friendly_picks"]) == 5

    def test_build_cooccurrence(self, tmp_path):
        loader = TournamentDataLoader()
        drafts = [
            {"friendly_picks": [0, 1, 2], "enemy_picks": [3, 4], "bans": [], "win_rate": 0.6},
            {"friendly_picks": [0, 3, 4], "enemy_picks": [1, 2], "bans": [], "win_rate": 0.5}
        ]
        cooc = loader.build_cooccurrence_matrix(drafts, num_heros=5)
        assert cooc.shape == (5, 5)
        assert cooc[0][1] > 0  # Heroes 0 and 1 co-occur on friendly team

    def test_build_adjacency(self):
        import torch
        loader = TournamentDataLoader()
        cooc = [[0.0, 0.5], [0.5, 0.0]]
        import numpy as np
        cooc_arr = np.array(cooc)
        adj = loader.build_adjacency_tensor(cooc_arr, threshold=0.1)
        assert isinstance(adj, torch.Tensor)
        assert adj.shape == (2, 2)
