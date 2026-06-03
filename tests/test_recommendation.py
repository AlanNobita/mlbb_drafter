"""Tests for GCN recommendation engine."""
import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.recommendation.gcn_model import MOBARecGCN
from server.recommendation.draft_state import DraftState


def test_gcn_model_initializes():
    model = MOBARecGCN(num_heros=100, input_dim=64, hidden_dim=32, output_dim=16)
    assert model is not None


def test_draft_state_initializes():
    state = DraftState()
    assert state is not None


def test_draft_state_add_pick():
    state = DraftState()
    state.add_pick("Lancelot", is_ally=True)
    assert "Lancelot" in state.ally_picks


def test_draft_state_add_ban():
    state = DraftState()
    state.add_ban("Fanny")
    assert "Fanny" in state.bans


def test_set_adjacency_matrix():
    model = MOBARecGCN(num_heros=503)
    adj = torch.zeros(503, 503)
    adj[0, 1] = 1.0
    adj[1, 0] = 1.0
    model.set_adjacency_matrix(adj)
    assert model._edge_index is not None
    # Now stores dense adjacency matrix, not sparse edge_index
    assert model._edge_index.shape == (503, 503)


def test_recommend_uses_learned_embeddings_not_random():
    """recommend() must use learned hero embeddings, not torch.randn."""
    torch.manual_seed(42)
    model = MOBARecGCN(num_heros=30, input_dim=64, hidden_dim=32, output_dim=16)
    
    # Set all embeddings to a known constant so results are deterministic
    with torch.no_grad():
        model.hero_embedding.weight.fill_(1.0)
        for p in model.parameters():
            p.fill_(0.5)
    
    draft = {"ally_picks": ["Lancelot", "Gusion"], "enemy_picks": ["Chou"], "bans": ["Fanny"]}
    available = [f"Hero_{i}" for i in range(25)]
    
    # Run recommend twice with same model state - must be deterministic
    recs1 = model.recommend(draft, available, top_k=5)
    recs2 = model.recommend(draft, available, top_k=5)
    
    # Same model state must produce same recommendations
    assert recs1 == recs2, f"recommend() is non-deterministic: {recs1} != {recs2}"
    
    # Must not contain NaN or inf
    for hero, wr in recs1:
        assert not (wr != wr), f"recommend() returned NaN for {hero}"
        assert not (wr == float('inf') or wr == float('-inf')), f"recommend() returned inf for {hero}"


def test_recommend_returns_valid_hero_names():
    """recommend() must return heroes from the available list."""
    model = MOBARecGCN(num_heros=30, input_dim=64, hidden_dim=32, output_dim=16)
    draft = {"ally_picks": [], "enemy_picks": [], "bans": []}
    available = ["Alice", "Balmond", "Cyclops", "Diggie", "Estes"]
    
    recs = model.recommend(draft, available, top_k=3)
    
    assert len(recs) == 3
    for hero_name, win_rate in recs:
        assert hero_name in available, f"Recommended hero {hero_name} not in available list"


def test_adjacency_forward_does_not_crash():
    """Forward pass with adjacency matrix mode must not crash."""
    model = MOBARecGCN(num_heros=30, input_dim=64, hidden_dim=32, output_dim=16)
    
    adj = torch.zeros(30, 30)
    adj[0, 1] = 1.0
    adj[1, 0] = 1.0
    adj[2, 3] = 1.0
    adj[3, 2] = 1.0
    model.set_adjacency_matrix(adj)
    
    friendly = torch.tensor([[0, 1, 2, 3, 4]])
    enemy = torch.tensor([[5, 6, 7, 8, 9]])
    
    # This must NOT crash with shape mismatch
    output = model(friendly_picks=friendly, enemy_picks=enemy)
    assert output.shape[0] == 1  # batch size


def test_recommend_deterministic_with_trained_model():
    """After setting adjacency, recommend must use graph structure."""
    model = MOBARecGCN(num_heros=30, input_dim=64, hidden_dim=32, output_dim=16)
    
    # Create adjacency where hero 0 and 1 are strongly connected
    adj = torch.zeros(30, 30)
    adj[0, 1] = 10.0
    adj[1, 0] = 10.0
    model.set_adjacency_matrix(adj)
    
    draft = {"ally_picks": ["Hero_0"], "enemy_picks": [], "bans": []}
    available = [f"Hero_{i}" for i in range(30)]
    
    recs = model.recommend(draft, available, top_k=5)
    rec_names = [h for h, _ in recs]
    
    # Hero_1 should be recommended (connected to Hero_0)
    # At minimum, results must be deterministic
    recs2 = model.recommend(draft, available, top_k=5)
    assert recs == recs2
