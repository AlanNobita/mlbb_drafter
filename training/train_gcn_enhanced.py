#!/usr/bin/env python3
"""
Enhanced GCN Training Script for MLBB Hero Recommendation

Uses ALL data sources:
- Tournament drafts (real matches)
- Hero stats (win rate, pick rate, ban rate)
- Temporal features (recency, meta score, trend)
- API relations (counters, synergies, adjacency matrix)
- Co-occurrence data (which heroes win together)
- Synergy matrix (tournament synergy scores)
- Era-specific stats (per-patch hero performance)

Usage:
    python training/train_gcn_enhanced.py --epochs 100
    python training/train_gcn_enhanced.py --epochs 200 --lr 0.0005
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.recommendation.gcn_model_v2 import MOBARecGCN


DATA_DIR = Path(__file__).parent / "data"
API_DIR = DATA_DIR / "api_data"


def load_hero_mapping() -> Dict[str, int]:
    """Load hero name to ID mapping from API data."""
    with open(API_DIR / "hero_winrate.json") as f:
        hero_list = json.load(f)
    return {hero["name"]: i for i, hero in enumerate(hero_list)}


def normalize_case(name: str) -> str:
    """Normalize hero name casing for consistent matching."""
    normalizations = {
        "Yi Sun-shin": "Yi Sun-Shin",
        "X.Borg": "X.Borg",
        "Popol and Kupa": "Popol and Kupa",
    }
    return normalizations.get(name, name)


def load_hero_features() -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load hero features from temporal stats and API data."""
    hero_to_id = load_hero_mapping()
    num_heros = len(hero_to_id)
    
    # Normalize hero names in mapping
    hero_to_id_normalized = {normalize_case(name): idx for name, idx in hero_to_id.items()}
    
    # Load temporal stats
    temporal_stats = {}
    with open(DATA_DIR / "hero_stats_temporal.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            temporal_stats[normalize_case(row["hero_name"])] = row
    
    # Load API winrate data
    api_winrate = {}
    with open(API_DIR / "hero_winrate.json") as f:
        for hero in json.load(f):
            api_winrate[normalize_case(hero["name"])] = hero
    
    # Load co-occurrence data (synergy stats)
    cooccurrence = {}
    with open(DATA_DIR / "hero_cooccurrence_tournament.json") as f:
        raw_cooccur = json.load(f)
        for hero, pairs in raw_cooccur.items():
            # Calculate average synergy score for this hero
            if pairs:
                avg_win_rate = np.mean([p["win_rate"] for p in pairs.values()])
                total_games = sum(p["games_together"] for p in pairs.values())
                cooccurrence[normalize_case(hero)] = {
                    "avg_synergy_win_rate": avg_win_rate,
                    "total_synergy_games": total_games
                }
    
    # Build feature tensor: [win_rate, pick_rate, ban_rate, meta_score, trend, 
    #                        weighted_win_rate, recency_score, total_games,
    #                        synergy_win_rate, synergy_games]
    feature_dim = 10
    features = torch.zeros(num_heros, feature_dim)
    
    # Track missing heroes
    missing_heroes = []
    
    for name, idx in hero_to_id_normalized.items():
        temporal = temporal_stats.get(name, {})
        api = api_winrate.get(name, {})
        cooccur = cooccurrence.get(name, {})
        
        if not api:
            missing_heroes.append(name)
        
        # Parse API percentages, clamp to [0, 1]
        def parse_pct(s):
            if isinstance(s, str):
                val = float(s.replace("%", ""))
                return min(val / 100.0, 1.0)
            return min(float(s) if s else 0.0, 1.0)
        
        features[idx, 0] = parse_pct(api.get("winrate", 0))  # win_rate
        features[idx, 1] = parse_pct(api.get("pickrate", 0))  # pick_rate
        features[idx, 2] = parse_pct(api.get("banrate", 0))  # ban_rate
        features[idx, 3] = float(temporal.get("meta_score", 0))  # meta_score
        features[idx, 4] = float(temporal.get("trend", 0))  # trend
        features[idx, 5] = float(temporal.get("weighted_win_rate", 0))  # weighted_win_rate
        features[idx, 6] = float(temporal.get("recency_score", 0))  # recency_score
        features[idx, 7] = float(temporal.get("total_tournament_games", 0)) / 1000.0  # total_games
        features[idx, 8] = cooccur.get("avg_synergy_win_rate", 0.5)  # synergy_win_rate
        features[idx, 9] = cooccur.get("total_synergy_games", 0) / 10000.0  # synergy_games
    
    if missing_heroes:
        print(f"  Warning: {len(missing_heroes)} heroes missing from API data: {missing_heroes[:5]}...")
    
    # Normalize features to [0, 1] range
    for i in range(feature_dim):
        col = features[:, i]
        col_min = col.min()
        col_max = col.max()
        if col_max > col_min:
            features[:, i] = (col - col_min) / (col_max - col_min)
    
    return features, hero_to_id


def load_adjacency_matrix(hero_to_id: Dict[str, int]) -> torch.Tensor:
    """Load and combine adjacency matrix with synergy matrix.
    
    Combines:
    - adjacency_real.json (counter/synergy relationships)
    - synergy_matrix_tournament.json (tournament synergy scores)
    """
    # Load base adjacency
    with open(DATA_DIR / "adjacency_real.json") as f:
        adj_data = json.load(f)
    
    hero_names = adj_data["hero_names"]
    adj_list = adj_data["adjacency"]
    
    num_heros = len(hero_to_id)
    adj = torch.zeros(num_heros, num_heros)
    
    # Map from API hero names to our IDs
    api_name_to_our_id = {}
    for i, name in enumerate(hero_names):
        normalized = normalize_case(name)
        if normalized in hero_to_id:
            api_name_to_our_id[i] = hero_to_id[normalized]
    
    # Fill adjacency matrix from base relationships
    for i, row in enumerate(adj_list):
        if i in api_name_to_our_id:
            our_i = api_name_to_our_id[i]
            for j, val in enumerate(row):
                if j in api_name_to_our_id and val != 0:
                    our_j = api_name_to_our_id[j]
                    adj[our_i, our_j] = min(abs(val), 1.0)
    
    # Load synergy matrix and blend with adjacency
    with open(DATA_DIR / "synergy_matrix_tournament.json") as f:
        synergy_data = json.load(f)
    
    synergy_heroes = synergy_data["hero_names"]
    synergy_matrix = synergy_data["synergy_matrix"]
    
    # Map synergy hero names to our IDs
    synergy_name_to_our_id = {}
    for i, name in enumerate(synergy_heroes):
        normalized = normalize_case(name)
        if normalized in hero_to_id:
            synergy_name_to_our_id[i] = hero_to_id[normalized]
    
    # Add synergy scores to adjacency (blend 70% base + 30% synergy)
    for i, row in enumerate(synergy_matrix):
        if i in synergy_name_to_our_id:
            our_i = synergy_name_to_our_id[i]
            for j, val in enumerate(row):
                if j in synergy_name_to_our_id and val != 0:
                    our_j = synergy_name_to_our_id[j]
                    # Normalize synergy value to [0, 1]
                    synergy_val = min(abs(val), 1.0)
                    # Blend with existing adjacency value
                    current_val = adj[our_i, our_j].item()
                    adj[our_i, our_j] = 0.7 * current_val + 0.3 * synergy_val
    
    # Symmetrize: A = (A + A.T) / 2
    adj = (adj + adj.T) / 2.0
    
    # Add self-loops
    adj += torch.eye(num_heros)
    
    # Normalize: D^{-1/2} A D^{-1/2} (Kipf & Welling)
    d = adj.sum(dim=1)
    d_inv_sqrt = torch.pow(d, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    adj = torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
    
    return adj


def load_tournament_drafts(hero_to_id: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load tournament drafts and convert to tensors."""
    with open(DATA_DIR / "tournament_drafts.json") as f:
        drafts = json.load(f)
    
    num_heros = len(hero_to_id)
    friendly_list = []
    enemy_list = []
    win_rates = []
    
    skipped = 0
    for draft in drafts:
        # Map hero names to IDs (with case normalization)
        friendly_ids = []
        for name in draft["blue_picks"]:
            normalized = normalize_case(name)
            if normalized in hero_to_id:
                friendly_ids.append(hero_to_id[normalized])
        
        enemy_ids = []
        for name in draft["red_picks"]:
            normalized = normalize_case(name)
            if normalized in hero_to_id:
                enemy_ids.append(hero_to_id[normalized])
        
        # Skip if not enough heroes mapped
        if len(friendly_ids) < 5 or len(enemy_ids) < 5:
            skipped += 1
            continue
        
        # Win rate: 1.0 if blue wins, 0.0 if red wins
        win_rate = 1.0 if draft["winner"] == "t1" else 0.0
        
        friendly_list.append(friendly_ids[:5])
        enemy_list.append(enemy_ids[:5])
        win_rates.append(win_rate)
    
    if skipped > 0:
        print(f"  Warning: {skipped} drafts skipped due to unmapped heroes")
    
    friendly = torch.tensor(friendly_list, dtype=torch.long)
    enemy = torch.tensor(enemy_list, dtype=torch.long)
    win_rates = torch.tensor(win_rates, dtype=torch.float)
    
    return friendly, enemy, win_rates


def train_model(
    model: MOBARecGCN,
    friendly: torch.Tensor,
    enemy: torch.Tensor,
    win_rates: torch.Tensor,
    hero_features: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 32,
    device: str = "cpu",
    val_split: float = 0.1
) -> List[Dict]:
    """Train the enhanced GCN model with validation."""
    # Split into train/val
    n = len(friendly)
    n_val = int(n * val_split)
    n_train = n - n_val
    
    indices = torch.randperm(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    train_friendly = friendly[train_idx].to(device)
    train_enemy = enemy[train_idx].to(device)
    train_winrates = win_rates[train_idx].to(device)
    
    val_friendly = friendly[val_idx].to(device)
    val_enemy = enemy[val_idx].to(device)
    val_winrates = win_rates[val_idx].to(device)
    
    hero_features = hero_features.to(device)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    train_dataset = TensorDataset(train_friendly, train_enemy, train_winrates)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    history = []
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        n_batches = 0
        
        for batch_friendly, batch_enemy, batch_winrates in train_dataloader:
            optimizer.zero_grad()
            
            # Forward pass with hero features
            batch_size_actual = batch_friendly.shape[0]
            hero_feats_batch = hero_features.unsqueeze(0).expand(batch_size_actual, -1, -1)
            
            pred = model(
                friendly_picks=batch_friendly,
                enemy_picks=batch_enemy,
                hero_features=hero_feats_batch
            )
            
            loss = criterion(pred, batch_winrates)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = total_train_loss / max(n_batches, 1)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            hero_feats_val = hero_features.unsqueeze(0).expand(len(val_friendly), -1, -1)
            val_pred = model(val_friendly, val_enemy, hero_feats_val)
            val_loss = criterion(val_pred, val_winrates).item()
        
        scheduler.step(val_loss)
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train: {avg_train_loss:.6f}, Val: {val_loss:.6f}, Best: {best_val_loss:.6f}")
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with val_loss: {best_val_loss:.6f}")
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced GCN Model for MLBB Hero Recommendation")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output", type=str, default="training/data/gcn_model_v2.pt", help="Output model path")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ENHANCED GCN TRAINING - MLBB Hero Recommendation")
    print("Using ALL data sources for maximum performance")
    print("=" * 60)
    
    # Load all data
    print("\n[1/5] Loading hero mapping...")
    hero_to_id = load_hero_mapping()
    num_heros = len(hero_to_id)
    print(f"  {num_heros} heroes mapped")
    
    print("\n[2/5] Loading hero features (temporal + co-occurrence)...")
    hero_features, _ = load_hero_features()
    print(f"  Features shape: {hero_features.shape}")
    print(f"  Feature ranges: min={hero_features.min():.3f}, max={hero_features.max():.3f}")
    
    print("\n[3/5] Loading adjacency matrix (base + synergy)...")
    adj = load_adjacency_matrix(hero_to_id)
    print(f"  Adjacency shape: {adj.shape}")
    print(f"  Non-zero edges: {(adj > 0).sum().item()}")
    print(f"  NaN check: {torch.isnan(adj).any().item()}")
    
    print("\n[4/5] Loading tournament drafts...")
    friendly, enemy, win_rates = load_tournament_drafts(hero_to_id)
    print(f"  {friendly.shape[0]} training samples")
    print(f"  Win rate distribution: mean={win_rates.mean():.3f}, std={win_rates.std():.3f}")
    
    # Initialize model
    print("\n[5/5] Initializing Enhanced MOBARec-GCN v2 model...")
    model = MOBARecGCN(
        num_heros=num_heros,
        input_dim=128,
        hidden_dim=128,
        output_dim=64,
        hero_feature_dim=10  # Updated: 10 features now
    )
    model.set_adjacency_matrix(adj)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Hero feature dim: 10 (win_rate, pick_rate, ban_rate, meta_score,")
    print(f"                      trend, weighted_wr, recency, total_games,")
    print(f"                      synergy_wr, synergy_games)")
    
    # Train
    print(f"\nTraining for {args.epochs} epochs (val_split={args.val_split})...")
    history = train_model(
        model, friendly, enemy, win_rates, hero_features,
        args.epochs, args.lr, args.batch_size, args.device, args.val_split
    )
    
    # Save (exclude _adj_matrix from state_dict as it's a buffer, not a parameter)
    state_dict = {k: v for k, v in model.state_dict().items() if k != "_adj_matrix"}
    torch.save({
        "model_state_dict": state_dict,
        "hero_to_id": hero_to_id,
        "hero_features": hero_features,
        "adjacency_matrix": adj,
        "num_heros": num_heros,
        "input_dim": 128,
        "hidden_dim": 128,
        "output_dim": 64,
        "hero_feature_dim": 10,
        "history": history,
    }, args.output)
    
    print(f"\nModel saved to {args.output}")
    if history:
        print(f"Final train loss: {history[-1]['train_loss']:.6f}")
        print(f"Final val loss: {history[-1]['val_loss']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
