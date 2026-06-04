"""MOBARec-GCNFP v2: Enhanced GCN with full API data integration.

DEPRECATED: The GCN model has been replaced by the rule-based HeroScorer
in server/recommendation/scoring.py. This file is kept for reference only.

The GCN model achieved ~57% accuracy on tournament draft prediction,
barely above the 56% baseline. The rule-based scorer is transparent,
explainable, and uses correct counter/synergy data from multiple sources.
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


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
    """MOBARec-GCNFP v2 model for draft recommendation.
    
    Enhanced with:
    - Hero features (win rate, pick rate, ban rate, meta score, trend)
    - Real adjacency matrix from API relations (counters, synergies)
    - Temporal features for time-aware predictions
    """
    
    def __init__(
        self,
        num_heros: int = 132,
        input_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 64,
        hero_feature_dim: int = 15
    ):
        super().__init__()
        self.num_heros = num_heros
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hero_feature_dim = hero_feature_dim
        
        # Hero embedding layer
        self.hero_embedding = nn.Embedding(num_heros, input_dim)
        
        # Hero feature projection (win rate, pick rate, ban rate, meta score, trend, etc.)
        # Now includes 5 lane features: EXP, Gold, Mid, Jungle, Roam
        self.hero_feature_proj = nn.Linear(hero_feature_dim, input_dim)
        
        # GCN layers
        self.gcn_backbone = GraphConvolution(input_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Win rate prediction head (outputs scalar)
        self.win_rate_head = nn.Linear(output_dim, 1)
        
        # Use a buffer for adjacency matrix (moves with model to device)
        self.register_buffer("_adj_matrix", None)
    
    def set_adjacency_matrix(self, adj_matrix: torch.Tensor) -> None:
        """Set adjacency matrix for the GCN backbone."""
        if adj_matrix.shape != (self.num_heros, self.num_heros):
            raise ValueError(
                f"Shape {adj_matrix.shape} doesn't match num_heros={self.num_heros}"
            )
        # Register as buffer so it moves with model to device
        self._adj_matrix = adj_matrix.clone()
    
    def forward(
        self,
        friendly_picks: torch.Tensor = None,
        enemy_picks: torch.Tensor = None,
        hero_features: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            friendly_picks: Hero indices [batch, num_friendly] (values 0..num_heros-1)
            enemy_picks: Hero indices [batch, num_enemy] (values 0..num_heros-1)
            hero_features: Optional hero features [batch, num_heros, hero_feature_dim]
            
        Returns:
            Win rate predictions [batch] in [0, 1] range
        """
        if self._adj_matrix is None:
            raise ValueError("Adjacency matrix not set. Call set_adjacency_matrix() first.")
        
        if friendly_picks is None or enemy_picks is None:
            raise ValueError("Provide friendly_picks and enemy_picks")
        
        device = self.hero_embedding.weight.device
        
        # Validate input shapes
        if friendly_picks.dim() != 2 or enemy_picks.dim() != 2:
            raise ValueError(
                f"Expected 2D tensors, got friendly={friendly_picks.dim()}D, enemy={enemy_picks.dim()}D"
            )
        
        # Get base embeddings
        x = self.hero_embedding.weight  # (num_heros, input_dim)
        
        # Add hero features if provided
        if hero_features is not None:
            # Project hero features to embedding space
            feat_proj = self.hero_feature_proj(hero_features)  # (batch, num_heros, input_dim)
            # Average across batch to get global feature contribution
            feat_global = feat_proj.mean(dim=0)  # (num_heros, input_dim)
            x = x + feat_global
        
        # Apply GCN with adjacency matrix (already normalized)
        adj = self._adj_matrix.to(device)
        gcn_out = F.relu(self.gcn_backbone(x, adj))  # (num_heros, hidden_dim)
        
        # For each sample, extract embeddings of friendly+enemy heroes
        batch_size = friendly_picks.shape[0]
        gcn_out_list = []
        
        for b in range(batch_size):
            # Create masks for picked heroes
            friendly_mask = torch.zeros(self.num_heros, device=device)
            enemy_mask = torch.zeros(self.num_heros, device=device)
            
            # Only count valid hero indices
            valid_friendly = friendly_picks[b][(friendly_picks[b] >= 0) & (friendly_picks[b] < self.num_heros)]
            valid_enemy = enemy_picks[b][(enemy_picks[b] >= 0) & (enemy_picks[b] < self.num_heros)]
            
            friendly_mask[valid_friendly] = 1.0
            enemy_mask[valid_enemy] = 1.0
            
            # Weighted sum: friendly heroes contribute positively, enemy negatively
            friendly_emb = (gcn_out * friendly_mask.unsqueeze(1)).sum(dim=0)
            enemy_emb = (gcn_out * enemy_mask.unsqueeze(1)).sum(dim=0)
            
            # Dynamic match embedding: ally - enemy
            match_emb = friendly_emb - enemy_emb
            gcn_out_list.append(match_emb)
        
        gcn_out = torch.stack(gcn_out_list)  # (batch_size, hidden_dim)
        
        # Project to output space
        output = self.output_proj(gcn_out)  # (batch_size, output_dim)
        win_rate_logit = self.win_rate_head(output).squeeze(-1)  # (batch_size,)
        
        # Apply sigmoid to get [0, 1] range
        win_rate = torch.sigmoid(win_rate_logit)
        
        return win_rate
    
    def recommend(
        self,
        current_draft: dict,
        available_heroes: list,
        hero_name_to_id: dict,
        hero_features: torch.Tensor = None,
        hero_meta: dict = None,
        top_k: int = 5,
        counter_data: dict = None,
        filter_lanes: bool = True,
        needed_lane: str = None
    ) -> list:
        """Get top-k hero recommendations.
        
        Args:
            current_draft: Dict with ally_picks, enemy_picks (hero names)
            available_heroes: List of available hero names
            hero_name_to_id: Mapping from hero name to model index
            hero_features: Optional hero features tensor
            hero_meta: Optional hero metadata with lane info
            top_k: Number of recommendations
            counter_data: Optional counter data
            filter_lanes: If True, prioritize heroes that fill needed lanes
            needed_lane: If set, override auto-detection and prioritize this lane (e.g. "Jungle", "EXP")
            counter_data: Optional counter data from hero_counters_openmlbb.json
            filter_lanes: If True, prioritize heroes that fill needed lanes
            
        Returns:
            List of (hero_name, win_rate) tuples
        """
        self.eval()
        
        if not available_heroes:
            return []
        
        if self._adj_matrix is None:
            raise ValueError("Adjacency matrix not set. Call set_adjacency_matrix() first.")
        
        # Counter data disabled — waiting for correct mlbb.io counter pick data
        counter_map = None
        
        # Load synergy data (heroes that work well with allies)
        synergy_map = None
        synergy_path = Path(__file__).parent.parent.parent / "training" / "data" / "api_data" / "mlbb_io_overviews.json"
        if synergy_path.exists():
            with open(synergy_path) as f:
                synergy_map = json.load(f)
        
        # Build ally synergy boost: which heroes synergize with our picks
        ally_names = current_draft.get("ally_picks", current_draft.get("friendly_picks", []))
        ally_synergy_boost = {}
        if synergy_map:
            for ally in ally_names:
                if ally in synergy_map:
                    for syn_name in synergy_map[ally].get("synergies", []):
                        # Don't boost heroes already picked
                        if syn_name not in ally_synergy_boost:
                            ally_synergy_boost[syn_name] = 0
                        ally_synergy_boost[syn_name] += 0.10  # +10% per synergy match
        
        # Extract enemy picks for counter lookup
        enemy_names = current_draft.get("enemy_picks", [])
        
        # Build enemy counter boost map: hero_name -> boost_score
        # For each enemy hero, find ALL heroes that counter them
        enemy_counter_boost = {}
        if counter_map:
            for enemy in enemy_names:
                if enemy in counter_map:
                    for counter in counter_map[enemy].get("counters", []):
                        counter_name = counter["hero"]
                        increase = counter.get("increase", 0.03)
                        # Scale increase to [0, 1] range (typical increase is 0.01-0.07)
                        boost = min(increase * 10, 1.0)
                        if counter_name not in enemy_counter_boost or boost > enemy_counter_boost[counter_name]:
                            enemy_counter_boost[counter_name] = boost
        
        # Convert hero names to IDs (handle both 'ally_picks' and 'friendly_picks' keys)
        friendly_names = current_draft.get("ally_picks", current_draft.get("friendly_picks", []))
        
        friendly_ids = []
        for name in friendly_names:
            if name in hero_name_to_id:
                friendly_ids.append(hero_name_to_id[name])
        
        enemy_ids = []
        for name in enemy_names:
            if name in hero_name_to_id:
                enemy_ids.append(hero_name_to_id[name])
        
        # Pad to fixed size with -1 (sentinel value, not valid hero index)
        while len(friendly_ids) < 5:
            friendly_ids.append(-1)
        while len(enemy_ids) < 5:
            enemy_ids.append(-1)
        
        # Determine needed lanes based on current ally picks
        needed_lanes = set()
        if needed_lane:
            # User explicitly told us which lane is needed — trust them
            needed_lanes = {needed_lane}
        elif filter_lanes and hero_meta:
            # Lanes in MLBB: EXP, Gold, Mid, Jungle, Roam
            all_lanes = {"EXP", "Gold", "Mid", "Jungle", "Roam"}
            used_lanes = set()
            # hero_meta can be a list of dicts or a dict of dicts
            meta_dict = {}
            if isinstance(hero_meta, list):
                meta_dict = {h["name"]: h for h in hero_meta}
            else:
                meta_dict = hero_meta
            for name in friendly_names:
                if name in meta_dict:
                    primary_lane = meta_dict[name].get("primary_lane", "")
                    if primary_lane:
                        used_lanes.add(primary_lane)
            needed_lanes = all_lanes - used_lanes
        
        device = self.hero_embedding.weight.device
        friendly_tensor = torch.tensor([friendly_ids[:5]], dtype=torch.long, device=device)
        enemy_tensor = torch.tensor([enemy_ids[:5]], dtype=torch.long, device=device)
        
        # Get predictions for all available heroes
        scores = []
        for hero_name in available_heroes:
            if hero_name not in hero_name_to_id:
                continue
            
            hero_id = hero_name_to_id[hero_name]
            
            # Try adding this hero to friendly team
            test_friendly = friendly_tensor.clone()
            # Find first empty slot (marked with -1)
            empty_slots = (test_friendly[0] == -1).nonzero(as_tuple=True)[0]
            if len(empty_slots) > 0:
                test_friendly[0, empty_slots[0]] = hero_id
            
            with torch.no_grad():
                pred = self.forward(test_friendly, enemy_tensor, hero_features)
                gcn_score = pred.item()
            
            # Hero meta strength (win rate, pick rate, ban rate from features)
            hero_id = hero_name_to_id.get(hero_name, -1)
            meta_score = 0.5
            if hero_features is not None and 0 <= hero_id < hero_features.shape[0]:
                feats = hero_features[hero_id]
                # Features: [win_rate, pick_rate, ban_rate, meta_score, trend, ...]
                meta_score = feats[0].item() * 0.5 + feats[1].item() * 0.2 + feats[2].item() * 0.1 + feats[3].item() * 0.2
            
            # Counter boost (draft-specific signal)
            counter_boost = enemy_counter_boost.get(hero_name, 0)
            
            # Synergy boost (team-specific signal)
            synergy_boost = min(ally_synergy_boost.get(hero_name, 0), 0.20)
            
            # Lane check: can this hero fill the needed lane?
            meta_for_lane = hero_meta
            if isinstance(meta_for_lane, list):
                meta_for_lane = {h["name"]: h for h in meta_for_lane} if meta_for_lane else {}
            hero_lane = meta_for_lane.get(hero_name, {}).get("primary_lane", "") if meta_for_lane else ""
            hero_all_lanes = meta_for_lane.get(hero_name, {}).get("lanes", [hero_lane]) if meta_for_lane else []
            needed_key = needed_lane.replace(" Lane", "") if needed_lane else None
            fills_lane = (needed_key and needed_key in hero_all_lanes) or (hero_lane in needed_lanes) if needed_lane else True
            
            # Scoring: meta is the base, counter/synergy are multipliers
            # Only give counter/synergy boost if hero fills the needed lane
            base_score = meta_score * 0.6 + gcn_score * 0.4
            if fills_lane and counter_boost > 0:
                base_score = base_score * (1.0 + counter_boost * 3.0)  # Up to +150% boost
            if fills_lane and synergy_boost > 0:
                base_score = base_score * (1.0 + synergy_boost * 2.0)  # Up to +40% boost
            
            boosted_score = min(base_score, 1.0)
            
            # Lane bonus: +20% auto-detected, +40% if user explicitly said this lane
            lane_bonus = 0
            if filter_lanes and hero_meta:
                meta_dict = hero_meta if isinstance(hero_meta, dict) else {h["name"]: h for h in hero_meta}
                if hero_name in meta_dict:
                    hero_lane = meta_dict[hero_name].get("primary_lane", "")
                    # Also check all possible lanes for this hero
                    hero_all_lanes = meta_dict[hero_name].get("lanes", [hero_lane])
                    # Normalize needed_lane: "EXP Lane" -> "EXP", "Jungle" stays "Jungle"
                    needed_key = needed_lane.replace(" Lane", "") if needed_lane else None
                    if hero_lane in needed_lanes or (needed_key and needed_key in hero_all_lanes):
                        lane_bonus = 0.40 if needed_lane else 0.20
            
            final_score = min(boosted_score + lane_bonus, 1.0)
            scores.append((hero_name, final_score))
        
        # Sort by boosted win rate
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
