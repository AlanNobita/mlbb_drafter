"""Dummy detector for MVP testing."""
import random
from typing import Dict, List

# Sample MLBB heroes for testing
SAMPLE_HEROES = [
    "Lancelot", "Gusion", "Fanny", "Hayabusa", "Ling",
    "Chou", "Alucard", "Selena", "Kagura", "Harith",
    "Valir", "Lunox", "Esmeralda", "Yu Zhong", "Benedetta",
    "Atlas", "Khufra", "Akai", "Grock", "Johnson",
    "Angela", "Rafaela", "Estes", "Diggie", "Mathilda",
    "Granger", "Wanwan", "Claude", "Karrie", "Moskov",
]


class DummyDetector:
    """Dummy detector that returns random hero detections for MVP testing."""
    
    def __init__(self):
        self.heroes = SAMPLE_HEROES
    
    def detect(self, frame) -> Dict[str, List[str]]:
        """Detect heroes in frame (dummy implementation).
        
        Args:
            frame: numpy.ndarray (unused in dummy)
            
        Returns:
            Dictionary with ally_picks, enemy_picks, bans
        """
        shuffled = self.heroes.copy()
        random.shuffle(shuffled)
        
        return {
            "ally_picks": shuffled[:3],
            "enemy_picks": shuffled[3:6],
            "bans": shuffled[6:10],
        }
