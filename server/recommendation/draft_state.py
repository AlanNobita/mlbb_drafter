"""Track the current draft state."""
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class DraftState:
    """Tracks picks and bans in current draft."""
    
    ally_picks: List[str] = field(default_factory=list)
    enemy_picks: List[str] = field(default_factory=list)
    bans: List[str] = field(default_factory=list)
    phase: str = "ban"  # "ban" or "pick"
    
    def add_pick(self, hero: str, is_ally: bool):
        """Add a hero pick."""
        if is_ally:
            self.ally_picks.append(hero)
        else:
            self.enemy_picks.append(hero)
    
    def add_ban(self, hero: str):
        """Add a hero ban."""
        self.bans.append(hero)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "ally_picks": self.ally_picks,
            "enemy_picks": self.enemy_picks,
            "bans": self.bans,
            "phase": self.phase,
        }
    
    def is_complete(self) -> bool:
        """Check if draft is complete (5 picks per team)."""
        return len(self.ally_picks) >= 5 and len(self.enemy_picks) >= 5
    
    def reset(self):
        """Reset draft state."""
        self.ally_picks.clear()
        self.enemy_picks.clear()
        self.bans.clear()
        self.phase = "ban"
