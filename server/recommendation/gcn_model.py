"""MOBARec-GCNFP: Lightweight GCN for MOBA draft recommendation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """Single graph convolution layer."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Graph convolution: H' = σ(A H W + b)"""
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output + self.bias


class MOBARecGCN(nn.Module):
    """MOBARec-GCNFP model for draft recommendation.
    
    Uses single-layer GCN with dynamic match embedding initialization:
    em0 = Σ(efriendly) - Σ(eopponent)
    """
    
    def __init__(
        self,
        num_heros: int = 100,
        input_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 64
    ):
        super().__init__()
        self.num_heros = num_heros
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Hero embedding layer
        self.hero_embedding = nn.Embedding(num_heros, input_dim)
        
        # GCN layers (single layer as per spec)
        self.gcn = GraphConvolution(input_dim, hidden_dim)
        
        # GCN backbone for adjacency matrix mode
        self.gcn_backbone = GraphConvolution(input_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Win rate prediction head (maps to num_heros)
        self.win_rate_head = nn.Linear(output_dim, num_heros)
        
        # Adjacency matrix support
        self._edge_index = None
    
    def set_adjacency_matrix(self, adj_matrix: torch.Tensor) -> None:
        """
        Set adjacency matrix for the GCN backbone.
        
        Args:
            adj_matrix: (num_heros, num_heros) tensor. Non-zero entries = edges.
        """
        assert adj_matrix.shape == (self.num_heros, self.num_heros), \
            f"Shape {adj_matrix.shape} doesn't match num_heros={self.num_heros}"
        edge_index = self._adj_to_edge_index(adj_matrix)
        self._edge_index = edge_index
    
    def _adj_to_edge_index(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Store dense adjacency matrix for GraphConvolution."""
        return adj_matrix
    
    def compute_match_embedding(
        self,
        ally_hero_ids: torch.Tensor,
        enemy_hero_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute dynamic match embedding.
        
        em0 = Σ(efriendly) - Σ(eopponent)
        """
        ally_embeds = self.hero_embedding(ally_hero_ids)
        enemy_embeds = self.hero_embedding(enemy_hero_ids)
        
        # Sum pooling for each team
        ally_sum = ally_embeds.sum(dim=0)
        enemy_sum = enemy_embeds.sum(dim=0)
        
        # Dynamic match embedding
        match_embed = ally_sum - enemy_sum
        return match_embed
    
    def forward(
        self,
        x: torch.Tensor = None,
        adj: torch.Tensor = None,
        ally_ids: torch.Tensor = None,
        enemy_ids: torch.Tensor = None,
        friendly_picks: torch.Tensor = None,
        enemy_picks: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Supports two modes:
        1. Legacy mode (x, adj): Raw node features and adjacency matrix
        2. Adjacency matrix mode (friendly_picks, enemy_picks): Uses set adjacency matrix
        
        Args:
            x: Node features [batch, num_nodes, input_dim] (legacy mode)
            adj: Adjacency matrix [batch, num_nodes, num_nodes] (legacy mode)
            ally_ids: Hero IDs for ally team (legacy mode)
            enemy_ids: Hero IDs for enemy team (legacy mode)
            friendly_picks: Hero indices [batch, num_friendly] (adjacency matrix mode)
            enemy_picks: Hero indices [batch, num_enemy] (adjacency matrix mode)
            
        Returns:
            Win rate predictions [batch, num_heros]
        """
        if self._edge_index is not None and friendly_picks is not None:
            # Adjacency matrix mode: process full graph once
            device = self.hero_embedding.weight.device
            x = self.hero_embedding.weight  # (num_heros, input_dim)
            adj = self._edge_index.to(device)  # dense (num_heros, num_heros)
            gcn_out = F.relu(self.gcn_backbone(x, adj))  # (num_heros, hidden_dim)
            
            # For each sample, extract embeddings of friendly+enemy heroes and average
            batch_size = friendly_picks.shape[0]
            gcn_out_list = []
            for b in range(batch_size):
                picked = torch.cat([friendly_picks[b], enemy_picks[b]], dim=0)
                picked_emb = gcn_out[picked]  # (num_picked, hidden_dim)
                gcn_out_list.append(picked_emb.mean(dim=0))
            gcn_out = torch.stack(gcn_out_list)  # (batch_size, hidden_dim)
            
            # Project to output space
            output = self.output_proj(gcn_out)
            win_rates = self.win_rate_head(output)
            return win_rates.squeeze(-1)
        elif x is not None and adj is not None:
            # Legacy mode: graph convolution on provided features
            h = F.relu(self.gcn(x, adj))
            h_pooled = h.mean(dim=1)
            output = self.output_proj(h_pooled)
            win_rates = self.win_rate_head(output)
            return win_rates.squeeze(-1)
        else:
            raise ValueError(
                "Provide either (x, adj) for legacy mode or "
                "(friendly_picks, enemy_picks) with a set adjacency matrix"
            )
    
    def recommend(
        self,
        current_draft: dict,
        available_heroes: list,
        top_k: int = 5
    ) -> list:
        """Get top-k hero recommendations.
        
        Args:
            current_draft: Dict with ally_picks, enemy_picks, bans
            available_heroes: List of available hero names
            top_k: Number of recommendations
            
        Returns:
            List of (hero_name, win_rate) tuples
        """
        self.eval()
        with torch.no_grad():
            num_available = len(available_heroes)
            if num_available == 0:
                return []
            
            # Build hero index mapping: available_heroes position -> model hero index
            # We use the hash of hero name to get a stable index into the embedding
            hero_indices = []
            for name in available_heroes:
                hero_indices.append(hash(name) % self.num_heros)
            hero_indices = torch.tensor(hero_indices, dtype=torch.long)
            
            # Get real embeddings for available heroes
            avail_emb = self.hero_embedding(hero_indices)  # (num_available, input_dim)
            
            # Use adjacency structure if available
            if self._edge_index is not None:
                adj = self._edge_index  # dense (num_heros, num_heros)
                gcn_out = F.relu(self.gcn_backbone(self.hero_embedding.weight, adj))
                avail_gcn = gcn_out[hero_indices]  # (num_available, hidden_dim)
            else:
                # Fallback: use GCN with identity adjacency (each hero only connected to self)
                identity = torch.eye(self.num_heros, device=self.hero_embedding.weight.device)
                gcn_out = F.relu(self.gcn_backbone(self.hero_embedding.weight, identity))
                avail_gcn = gcn_out[hero_indices]  # (num_available, hidden_dim)
            
            # Project to output space
            output = self.output_proj(avail_gcn)  # (num_available, output_dim)
            win_rates = self.win_rate_head(output)  # (num_available, num_heros)
            
            # We want the score for each available hero
            # Take the diagonal or use a simpler scoring
            # For now, use the mean of output as a single score per hero
            scores = output.mean(dim=-1)  # (num_available,)
            
            # Get top-k
            k = min(top_k, num_available)
            top_values, top_indices = torch.topk(scores, k)
            
            recommendations = []
            for i in range(k):
                idx = top_indices[i].item()
                recommendations.append(
                    (available_heroes[idx], top_values[i].item())
                )
            
            return recommendations
