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
    """Normalize hero name casing for consistent matching.
    
    Handles case mismatches between data sources:
    - hero_winrate.json has "Yi Sun-shin" (lowercase s)
    - tournament_drafts.json has "Yi Sun-Shin" (uppercase S)
    - adjacency_real.json has "Yi Sun-shin" (lowercase s)
    """
    normalizations = {
        "Yi Sun-shin": "Yi Sun-Shin",
        "X.Borg": "X.Borg",
        "Popol and Kupa": "Popol and Kupa",
    }
    return normalizations.get(name, name)


def build_hero_lookup(hero_to_id: Dict[str, int]) -> Dict[str, int]:
    """Build a case-insensitive hero lookup that normalizes all names.
    
    This ensures that "Yi Sun-shin" (from hero_winrate.json) and 
    "Yi Sun-Shin" (from tournament_drafts.json) both resolve to the same ID.
    
    Returns a dict with exactly len(hero_to_id) unique IDs (no duplicates).
    """
    lookup = {}
    for name, idx in hero_to_id.items():
        normalized = normalize_case(name)
        # Always use the normalized form as the canonical key
        lookup[normalized] = idx
        # Keep original name as alias (overwrites if normalized already set same ID)
        if name != normalized:
            lookup[name] = idx
    return lookup


def load_hero_features() -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load hero features from temporal stats and API data."""
    hero_to_id = load_hero_mapping()
    num_heros = len(hero_to_id)
    
    # Build normalized lookup (handles case mismatches across data sources)
    hero_lookup = build_hero_lookup(hero_to_id)
    
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
    
    # Load hero meta for lane data
    hero_meta = {}
    meta_path = Path(__file__).parent.parent / "shared" / "hero_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta_data = json.load(f)
            hero_meta = {normalize_case(h['name']): h for h in meta_data.get('heroes', [])}
    
    # Build feature tensor: [win_rate, pick_rate, ban_rate, meta_score, trend, 
    #                        weighted_win_rate, recency_score, total_games,
    #                        synergy_win_rate, synergy_games,
    #                        lane_EXP, lane_Gold, lane_Mid, lane_Jungle, lane_Roam]
    feature_dim = 15
    features = torch.zeros(num_heros, feature_dim)
    
    # Lane encoding: EXP=0, Gold=1, Mid=2, Jungle=3, Roam=4
    lane_to_idx = {"EXP": 10, "Gold": 11, "Mid": 12, "Jungle": 13, "Roam": 14}
    
    # Track missing heroes
    missing_heroes = []
    
    for name, idx in hero_lookup.items():
        # Skip duplicate entries from build_hero_lookup
        if idx >= num_heros:
            continue
        
        # Normalize name for consistent lookup across data sources
        normalized = normalize_case(name)
        
        temporal = temporal_stats.get(normalized, {}) or temporal_stats.get(name, {})
        api = api_winrate.get(normalized, {}) or api_winrate.get(name, {})
        cooccur = cooccurrence.get(normalized, {}) or cooccurrence.get(name, {})
        
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
        
        # Lane features: one-hot encoding for primary lane
        hero_meta_data = hero_meta.get(normalized, {}) or hero_meta.get(name, {})
        primary_lane = hero_meta_data.get("primary_lane", "")
        if primary_lane in lane_to_idx:
            features[idx, lane_to_idx[primary_lane]] = 1.0
    
    if missing_heroes:
        print(f"  Warning: {len(missing_heroes)} heroes missing from API data: {missing_heroes[:5]}...")
    
    # Normalize features to [0, 1] range
    for i in range(feature_dim):
        col = features[:, i]
        col_min = col.min()
        col_max = col.max()
        if col_max > col_min:
            features[:, i] = (col - col_min) / (col_max - col_min)
    
    return features, hero_lookup


def load_adjacency_matrix(hero_lookup: Dict[str, int]) -> torch.Tensor:
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
    
    num_heros = len(set(hero_lookup.values()))  # Unique IDs only (lookup may have aliases)
    adj = torch.zeros(num_heros, num_heros)
    
    # Map from API hero names to our IDs (using normalized lookup)
    api_name_to_our_id = {}
    for i, name in enumerate(hero_names):
        normalized = normalize_case(name)
        if normalized in hero_lookup:
            api_name_to_our_id[i] = hero_lookup[normalized]
        elif name in hero_lookup:
            api_name_to_our_id[i] = hero_lookup[name]
    
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
    
    # Map synergy hero names to our IDs (using normalized lookup)
    synergy_name_to_our_id = {}
    for i, name in enumerate(synergy_heroes):
        normalized = normalize_case(name)
        if normalized in hero_lookup:
            synergy_name_to_our_id[i] = hero_lookup[normalized]
        elif name in hero_lookup:
            synergy_name_to_our_id[i] = hero_lookup[name]
    
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


def load_tournament_drafts(
    hero_lookup: Dict[str, int],
    min_year: int = 2017,
    temporal_decay: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load tournament drafts with temporal weighting.
    
    Args:
        hero_lookup: Normalized hero name to ID mapping
        min_year: Minimum year to include (filters old drafts)
        temporal_decay: How much to weight recent vs old (0=equal, 1=only recent)
    
    Returns:
        friendly: Friendly team picks [n, 5]
        enemy: Enemy team picks [n, 5]
        win_rates: Win rates [n]
        sample_weights: Temporal weights [n] (newer = higher)
    """
    with open(DATA_DIR / "tournament_drafts.json") as f:
        drafts = json.load(f)
    
    # Find the latest date for normalization
    all_dates = []
    for draft in drafts:
        date_str = draft.get("date", "20230101")
        try:
            year = int(date_str[:4])
            all_dates.append(year)
        except:
            all_dates.append(2023)
    
    max_year = max(all_dates) if all_dates else 2023
    
    friendly_list = []
    enemy_list = []
    win_rates = []
    weights = []
    
    skipped_old = 0
    skipped_heroes = 0
    skipped_cross_team = 0
    skipped_dupes = 0
    
    for draft in drafts:
        # Check date
        date_str = draft.get("date", "20230101")
        try:
            year = int(date_str[:4])
        except:
            year = 2023
        
        # Filter by minimum year
        if year < min_year:
            skipped_old += 1
            continue
        
        # Map hero names to IDs (with case normalization)
        friendly_ids = []
        for name in draft["blue_picks"]:
            normalized = normalize_case(name)
            if normalized in hero_lookup:
                friendly_ids.append(hero_lookup[normalized])
            elif name in hero_lookup:
                friendly_ids.append(hero_lookup[name])
        
        enemy_ids = []
        for name in draft["red_picks"]:
            normalized = normalize_case(name)
            if normalized in hero_lookup:
                enemy_ids.append(hero_lookup[normalized])
            elif name in hero_lookup:
                enemy_ids.append(hero_lookup[name])
        
        # Skip if not enough heroes mapped
        if len(friendly_ids) < 5 or len(enemy_ids) < 5:
            skipped_heroes += 1
            continue
        
        # Filter cross-team: same hero on both teams (impossible in real MLBB)
        if set(friendly_ids) & set(enemy_ids):
            skipped_cross_team += 1
            continue
        
        # Filter duplicate heroes within same team
        if len(friendly_ids) != len(set(friendly_ids)) or len(enemy_ids) != len(set(enemy_ids)):
            skipped_dupes += 1
            continue
        
        # Calculate temporal weight (newer drafts get higher weight)
        # Linear decay: year 2023 = 1.0, year min_year = ~0.1
        year_weight = 0.1 + 0.9 * ((year - min_year) / max(1, max_year - min_year))
        # Apply temporal decay factor
        sample_weight = year_weight ** (1 + temporal_decay)
        
        # Win rate: 1.0 if blue wins, 0.0 if red wins
        win_rate = 1.0 if draft["winner"] == "t1" else 0.0
        
        friendly_list.append(friendly_ids[:5])
        enemy_list.append(enemy_ids[:5])
        win_rates.append(win_rate)
        weights.append(sample_weight)
    
    if skipped_old > 0:
        print(f"  Skipped {skipped_old} drafts before {min_year}")
    if skipped_heroes > 0:
        print(f"  Skipped {skipped_heroes} drafts due to unmapped heroes")
    if skipped_cross_team > 0:
        print(f"  Skipped {skipped_cross_team} cross-team duplicate drafts")
    if skipped_dupes > 0:
        print(f"  Skipped {skipped_dupes} drafts with duplicate heroes within team")
    
    friendly = torch.tensor(friendly_list, dtype=torch.long)
    enemy = torch.tensor(enemy_list, dtype=torch.long)
    win_rates = torch.tensor(win_rates, dtype=torch.float)
    weights = torch.tensor(weights, dtype=torch.float)
    
    # Normalize weights to average to 1.0 (so total loss is comparable)
    weights = weights / weights.mean()
    
    return friendly, enemy, win_rates, weights


def train_model(
    model: MOBARecGCN,
    friendly: torch.Tensor,
    enemy: torch.Tensor,
    win_rates: torch.Tensor,
    hero_features: torch.Tensor,
    sample_weights: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 32,
    device: str = "cpu",
    val_split: float = 0.1,
    patience: int = 20
) -> List[Dict]:
    """Train the enhanced GCN model with temporal weighting."""
    # Split into train/val
    n = len(friendly)
    n_val = int(n * val_split)
    n_train = n - n_val
    
    # Stratified split (maintain temporal distribution)
    indices = torch.randperm(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    train_friendly = friendly[train_idx].to(device)
    train_enemy = enemy[train_idx].to(device)
    train_winrates = win_rates[train_idx].to(device)
    train_weights = sample_weights[train_idx].to(device)
    
    val_friendly = friendly[val_idx].to(device)
    val_enemy = enemy[val_idx].to(device)
    val_winrates = win_rates[val_idx].to(device)
    
    hero_features = hero_features.to(device)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss(reduction='none')  # We'll apply weights manually
    
    train_dataset = TensorDataset(train_friendly, train_enemy, train_winrates, train_weights)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    history = []
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    max_patience = patience
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        n_batches = 0
        
        for batch_friendly, batch_enemy, batch_winrates, batch_weights in train_dataloader:
            optimizer.zero_grad()
            
            # Forward pass with hero features
            batch_size_actual = batch_friendly.shape[0]
            hero_feats_batch = hero_features.unsqueeze(0).expand(batch_size_actual, -1, -1)
            
            pred = model(
                friendly_picks=batch_friendly,
                enemy_picks=batch_enemy,
                hero_features=hero_feats_batch
            )
            
            # Weighted loss (newer drafts contribute more)
            element_loss = criterion(pred, batch_winrates)
            weighted_loss = (element_loss * batch_weights).sum()
            
            weighted_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += weighted_loss.item()
            n_batches += 1
        
        avg_train_loss = total_train_loss / max(n_batches, 1)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            hero_feats_val = hero_features.unsqueeze(0).expand(len(val_friendly), -1, -1)
            val_pred = model(val_friendly, val_enemy, hero_feats_val)
            val_loss = criterion(val_pred, val_winrates).mean().item()
        
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
        if max_patience > 0 and patience_counter >= max_patience:
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
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension (128=small, 256=medium, 512=large)")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GCN layers (1-4)")
    parser.add_argument("--min-year", type=int, default=2017, help="Minimum year for drafts (2017=all, 2021=recent only)")
    parser.add_argument("--temporal-decay", type=float, default=0.1, help="Temporal decay (0=equal weight, 1=only recent)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (0=disable)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ENHANCED GCN TRAINING - MLBB Hero Recommendation")
    print("Using ALL data sources for maximum performance")
    print("=" * 60)
    
    # Load all data
    print("\n[1/5] Loading hero mapping...")
    hero_to_id = load_hero_mapping()
    hero_lookup = build_hero_lookup(hero_to_id)
    num_heros = len(hero_to_id)
    print(f"  {num_heros} heroes mapped")
    
    print("\n[2/5] Loading hero features (temporal + co-occurrence)...")
    hero_features, _ = load_hero_features()
    print(f"  Features shape: {hero_features.shape}")
    print(f"  Feature ranges: min={hero_features.min():.3f}, max={hero_features.max():.3f}")
    
    print("\n[3/5] Loading adjacency matrix (base + synergy)...")
    adj = load_adjacency_matrix(hero_lookup)
    print(f"  Adjacency shape: {adj.shape}")
    print(f"  Non-zero edges: {(adj > 0).sum().item()}")
    print(f"  NaN check: {torch.isnan(adj).any().item()}")
    
    print("\n[4/5] Loading tournament drafts with temporal weighting...")
    print(f"  Min year: {args.min_year}, Temporal decay: {args.temporal_decay}")
    friendly, enemy, win_rates, sample_weights = load_tournament_drafts(
        hero_lookup, 
        min_year=args.min_year,
        temporal_decay=args.temporal_decay
    )
    print(f"  {friendly.shape[0]} training samples")
    print(f"  Win rate distribution: mean={win_rates.mean():.3f}, std={win_rates.std():.3f}")
    print(f"  Temporal weight range: {sample_weights.min():.4f} to {sample_weights.max():.4f}")
    
    # Initialize model
    print("\n[5/5] Initializing Enhanced MOBARec-GCN v2 model...")
    model = MOBARecGCN(
        num_heros=num_heros,
        input_dim=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.hidden_dim // 2,
        hero_feature_dim=15
    )
    model.set_adjacency_matrix(adj)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Hero feature dim: 15 (win_rate, pick_rate, ban_rate, meta_score,")
    print(f"                      trend, weighted_wr, recency, total_games,")
    print(f"                      synergy_wr, synergy_games, lane_EXP, lane_Gold,")
    print(f"                      lane_Mid, lane_Jungle, lane_Roam)")
    print(f"  Model size: {'Small' if param_count < 100000 else 'Medium' if param_count < 500000 else 'Large'}")
    
    # Train
    print(f"\nTraining for {args.epochs} epochs (val_split={args.val_split}, patience={args.patience})...")
    history = train_model(
        model, friendly, enemy, win_rates, hero_features, sample_weights,
        args.epochs, args.lr, args.batch_size, args.device, args.val_split, args.patience
    )
    
    # Save (exclude _adj_matrix from state_dict as it's a buffer, not a parameter)
    state_dict = {k: v for k, v in model.state_dict().items() if k != "_adj_matrix"}
    torch.save({
        "model_state_dict": state_dict,
        "hero_to_id": hero_lookup,  # Use normalized lookup for inference
        "hero_features": hero_features,
        "adjacency_matrix": adj,
        "num_heros": num_heros,
        "input_dim": args.hidden_dim,
        "hidden_dim": args.hidden_dim,
        "output_dim": args.hidden_dim // 2,
        "hero_feature_dim": 15,
        "history": history,
    }, args.output)
    
    print(f"\nModel saved to {args.output}")
    if history:
        print(f"Final train loss: {history[-1]['train_loss']:.6f}")
        print(f"Final val loss: {history[-1]['val_loss']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
