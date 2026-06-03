#!/usr/bin/env python3
"""
GCN Model Training Script for MLBB Hero Recommendation

Usage:
    python training/train_gcn.py --data training/data/synthetic_drafts.csv --epochs 50
    python training/train_gcn.py --adj training/data/adjacency_matrix.pt --epochs 100
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.recommendation.gcn_model import MOBARecGCN


def load_training_data(csv_path: str, num_heros: int = 30) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load draft data from CSV and convert to tensors."""
    drafts = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            drafts.append({
                "friendly_picks": json.loads(row["friendly_picks"]),
                "enemy_picks": json.loads(row["enemy_picks"]),
                "bans": json.loads(row["bans"]),
                "win_rate": float(row["win_rate"])
            })

    n = len(drafts)
    friendly = torch.zeros(n, 5, dtype=torch.long)
    enemy = torch.zeros(n, 5, dtype=torch.long)
    win_rates = torch.zeros(n, dtype=torch.float)

    for i, d in enumerate(drafts):
        for j, hid in enumerate(d["friendly_picks"][:5]):
            friendly[i, j] = hid % num_heros
        for j, hid in enumerate(d["enemy_picks"][:5]):
            enemy[i, j] = hid % num_heros
        win_rates[i] = d["win_rate"]

    return friendly, enemy, win_rates


def build_adjacency_from_drafts(
    friendly: torch.Tensor,
    enemy: torch.Tensor,
    num_heros: int
) -> torch.Tensor:
    """Build adjacency matrix from co-pick patterns in training data.

    Heroes that appear together on the same team get a stronger edge.
    """
    adj = torch.zeros(num_heros, num_heros)
    for i in range(friendly.shape[0]):
        picks = friendly[i].tolist()
        for a in range(len(picks)):
            for b in range(a + 1, len(picks)):
                adj[picks[a], picks[b]] += 1
                adj[picks[b], picks[a]] += 1

    # Normalize and add self-loops
    adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)
    adj += torch.eye(num_heros)
    adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)
    return adj


def train_model(
    model: MOBARecGCN,
    train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 32,
    device: str = "cpu"
) -> List[Dict]:
    """Train the GCN model."""
    friendly, enemy, win_rates = train_data
    friendly = friendly.to(device)
    enemy = enemy.to(device)
    win_rates = win_rates.to(device)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(friendly, enemy, win_rates)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_friendly, batch_enemy, batch_winrates in dataloader:
            optimizer.zero_grad()

            # Forward pass using adjacency matrix mode (uses set adjacency matrix)
            outputs = model(friendly_picks=batch_friendly, enemy_picks=batch_enemy)

            # Model returns (batch_size, num_heros) - use mean across heroes as draft score
            pred = outputs.mean(dim=1)

            loss = criterion(pred, batch_winrates)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        history.append({"epoch": epoch + 1, "loss": avg_loss})

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    return history


def save_model(model: MOBARecGCN, path: str) -> None:
    """Save trained model weights."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train GCN Model for MLBB Hero Recommendation")
    parser.add_argument("--data", type=str, default="training/data/synthetic_drafts.csv", help="Training data CSV")
    parser.add_argument("--adj", type=str, default=None, help="Adjacency matrix path (.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-heros", type=int, default=30, help="Number of heroes")
    parser.add_argument("--output", type=str, default="training/data/gcn_model.pt", help="Output model path")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")

    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    friendly, enemy, win_rates = load_training_data(args.data, args.num_heros)
    print(f"Loaded {friendly.shape[0]} training samples")

    print(f"Initializing MOBARec-GCN model (num_heros={args.num_heros})...")
    model = MOBARecGCN(num_heros=args.num_heros)

    # Load or build adjacency matrix
    if args.adj:
        print(f"Loading adjacency matrix from {args.adj}...")
        adj = torch.load(args.adj, weights_only=True)
        model._edge_index = adj
        print(f"Adjacency matrix loaded: {adj.shape}")
    else:
        print("Building adjacency matrix from training data...")
        adj = build_adjacency_from_drafts(friendly, enemy, args.num_heros)
        model._edge_index = adj
        print(f"Adjacency matrix built: {adj.shape}")

    print(f"Training for {args.epochs} epochs...")
    history = train_model(model, (friendly, enemy, win_rates), args.epochs, args.lr, args.batch_size, args.device)

    save_model(model, args.output)
    print("Training complete!")


if __name__ == "__main__":
    main()
