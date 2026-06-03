from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import torch

class TournamentDataLoader:
    """Loads tournament draft data and builds adjacency matrices."""

    def __init__(self, data_dir: str = "training/data"):
        self.data_dir = Path(data_dir)

    def load_synthetic_drafts(self, filename: str = "synthetic_drafts.csv") -> List[Dict]:
        """Load synthetic draft data from CSV."""
        import csv
        filepath = self.data_dir / filename
        drafts = []
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                drafts.append({
                    "match_id": int(row["match_id"]),
                    "friendly_picks": json.loads(row["friendly_picks"]),
                    "enemy_picks": json.loads(row["enemy_picks"]),
                    "bans": json.loads(row["bans"]),
                    "win_rate": float(row["win_rate"])
                })
        return drafts

    def load_hero_meta(self, hero_meta_path: str = "shared/hero_meta.json") -> Dict[int, Dict]:
        """Load hero metadata indexed by hero ID."""
        with open(hero_meta_path, "r") as f:
            heroes = json.load(f)
        return {h["id"]: h for h in heroes}

    def build_cooccurrence_matrix(self, drafts: List[Dict], num_heros: int = 503) -> np.ndarray:
        """
        Build hero co-occurrence adjacency matrix from draft data.

        Two heroes are connected if they appear on the same team in any draft.
        """
        cooccurrence = np.zeros((num_heros, num_heros), dtype=np.float32)

        for draft in drafts:
            # Friendly team co-occurrence
            friendly = draft["friendly_picks"]
            for i in range(len(friendly)):
                for j in range(i + 1, len(friendly)):
                    h1, h2 = friendly[i], friendly[j]
                    if h1 < num_heros and h2 < num_heros:
                        cooccurrence[h1][h2] += 1.0
                        cooccurrence[h2][h1] += 1.0

            # Enemy team co-occurrence (counter relationships)
            enemy = draft["enemy_picks"]
            for i in range(len(enemy)):
                for j in range(i + 1, len(enemy)):
                    h1, h2 = enemy[i], enemy[j]
                    if h1 < num_heros and h2 < num_heros:
                        cooccurrence[h1][h2] += 1.0
                        cooccurrence[h2][h1] += 1.0

        # Normalize
        max_val = cooccurrence.max()
        if max_val > 0:
            cooccurrence /= max_val

        return cooccurrence

    def build_adjacency_tensor(self, cooccurrence: np.ndarray, threshold: float = 0.1) -> torch.Tensor:
        """Convert co-occurrence matrix to binary adjacency tensor."""
        adj = (cooccurrence > threshold).astype(np.float32)
        return torch.from_numpy(adj)

    def get_training_data(self, drafts: List[Dict], num_heros: int = 503) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert draft data to training tensors.

        Returns:
            friendly_picks: (N, 5) tensor of friendly hero IDs
            enemy_picks: (N, 5) tensor of enemy hero IDs
            win_rates: (N,) tensor of win rates
            bans: (N, 6) tensor of banned hero IDs
        """
        max_drafts = len(drafts)
        friendly = torch.zeros(max_drafts, 5, dtype=torch.long)
        enemy = torch.zeros(max_drafts, 5, dtype=torch.long)
        bans = torch.zeros(max_drafts, 6, dtype=torch.long)
        win_rates = torch.zeros(max_drafts, dtype=torch.float)

        for i, draft in enumerate(drafts):
            for j, hid in enumerate(draft["friendly_picks"][:5]):
                friendly[i, j] = hid
            for j, hid in enumerate(draft["enemy_picks"][:5]):
                enemy[i, j] = hid
            for j, hid in enumerate(draft["bans"][:6]):
                bans[i, j] = hid
            win_rates[i] = draft["win_rate"]

        return friendly, enemy, win_rates, bans


def main():
    loader = TournamentDataLoader()

    print("Loading synthetic drafts...")
    drafts = loader.load_synthetic_drafts()
    print(f"Loaded {len(drafts)} drafts")

    print("Building co-occurrence matrix...")
    cooccurrence = loader.build_cooccurrence_matrix(drafts, num_heros=30)  # Using 30 heroes for MVP
    print(f"Co-occurrence matrix shape: {cooccurrence.shape}")
    print(f"Non-zero entries: {(cooccurrence > 0).sum()}")

    print("Building adjacency tensor...")
    adj = loader.build_adjacency_tensor(cooccurrence, threshold=0.1)
    print(f"Adjacency tensor shape: {adj.shape}")
    print(f"Connected edges: {adj.sum().item()}")

    print("Converting to training tensors...")
    friendly, enemy, win_rates, bans = loader.get_training_data(drafts)
    print(f"Training data: {friendly.shape[0]} samples")

    # Save adjacency matrix
    output_path = Path("training/data/adjacency_matrix.pt")
    torch.save(adj, output_path)
    print(f"Saved adjacency matrix to {output_path}")


if __name__ == "__main__":
    main()
