"""Load hero metadata and tournament data."""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class HeroMetadata:
    id: int
    name: str
    real_name: str
    role: str
    roles: List[str] = field(default_factory=list)
    lanes: List[str] = field(default_factory=list)
    win_rate: float = 50.0
    pick_rate: float = 5.0
    tier: str = "B"


class HeroDataLoader:
    """Load and manage hero metadata."""
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Default to shared directory
            data_dir = Path(__file__).parent.parent.parent / "shared"
        self.data_dir = Path(data_dir)
        self.heroes: List[Dict] = []
        self.hero_by_name: Dict[str, Dict] = {}
        self.hero_by_id: Dict[int, Dict] = {}
    
    def load_hero_meta(self) -> List[Dict]:
        """Load hero metadata from JSON file."""
        meta_path = self.data_dir / "hero_meta.json"
        
        if not meta_path.exists():
            print(f"Warning: hero_meta.json not found at {meta_path}")
            return []
        
        with open(meta_path, "r") as f:
            data = json.load(f)
        
        self.heroes = data.get("heroes", [])
        
        # Build lookup dictionaries
        for hero in self.heroes:
            self.hero_by_name[hero["name"]] = hero
            self.hero_by_id[hero["id"]] = hero
        
        print(f"Loaded {len(self.heroes)} heroes")
        return self.heroes
    
    def get_hero_by_name(self, name: str) -> Optional[Dict]:
        """Get hero data by name."""
        return self.hero_by_name.get(name)
    
    def get_hero_by_id(self, hero_id: int) -> Optional[Dict]:
        """Get hero data by ID."""
        return self.hero_by_id.get(hero_id)
    
    def get_all_hero_names(self) -> List[str]:
        """Get list of all hero names."""
        return [h["name"] for h in self.heroes]
    
    def get_heroes_by_role(self, role: str) -> List[Dict]:
        """Get all heroes with a specific role."""
        return [h for h in self.heroes if h["role"] == role]
    
    def get_available_heroes(self, picks: List[str], bans: List[str]) -> List[Dict]:
        """Get heroes not yet picked or banned."""
        excluded = set(picks + bans)
        return [h for h in self.heroes if h["name"] not in excluded]
