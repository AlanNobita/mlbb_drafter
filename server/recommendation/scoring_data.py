"""Data loader for hero scoring system.

Loads counter, synergy, and stats data from multiple JSON sources.
Implements double-check validation for counter relationships.

Data Sources:
    - hero_counters_openmlbb.json: "counters" = heroes this hero BEATS (this hero counters them)
    - hero_meta.json: "counters" = heroes that BEAT this hero (this hero is countered_by them)
    - openmlbb_heroes.json: "weak" = heroes that beat this hero, "strong" = heroes this hero beats, "assist" = synergies
    - mlbb_io_overviews.json: synergies (hero names only)
    - mlbb_io_stats.json: win_rate, pick_rate, ban_rate, role, lane, speciality
"""
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


class ScoringData:
    """Loads and indexes all hero data for scoring."""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "training" / "data" / "api_data"
        else:
            data_dir = Path(data_dir)

        self.heroes: Set[str] = set()
        # hero -> heroes this hero counters (beats)
        self._counters: Dict[str, List[str]] = {}
        # hero -> heroes that counter (beat) this hero
        self._countered_by: Dict[str, List[str]] = {}
        # hero -> heroes it synergizes with
        self._synergies: Dict[str, List[str]] = {}
        # hero -> {win_rate, pick_rate, ban_rate, role, lane, speciality}
        self._stats: Dict[str, dict] = {}

        self._load_counter_data(data_dir)
        self._load_synergy_data(data_dir)
        self._load_stats(data_dir)

        # Build hero set from all loaded data
        all_heroes = set(self._counters.keys()) | set(self._countered_by.keys()) | set(self._stats.keys())
        self.heroes = all_heroes

    def _load_counter_data(self, data_dir: Path):
        """Load counter data from openmlbb, meta, and openmlbb_heroes sources."""
        # Source 1: hero_counters_openmlbb.json
        # "counters" = heroes this hero BEATS
        counters_path = data_dir / "hero_counters_openmlbb.json"
        if counters_path.exists():
            with open(counters_path) as f:
                data = json.load(f)
            for hero_name, info in data.items():
                hero_beats = [c["name"] for c in info.get("counters", [])]
                self._counters[hero_name] = hero_beats

        # Source 2: hero_meta.json
        # "counters" = heroes that BEAT this hero
        meta_path = data_dir / "hero_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                data = json.load(f)
            for hero in data.get("data", []):
                name = hero.get("hero_name", "")
                countered_by = [c.get("heroname") for c in hero.get("counters", []) if c.get("heroname")]
                if name:
                    self._countered_by[name] = countered_by

        # Source 3: openmlbb_heroes.json
        # weak = heroes that beat this hero
        # strong = heroes this hero beats
        # assist = synergies
        openmlbb_path = data_dir / "openmlbb_heroes.json"
        if openmlbb_path.exists():
            with open(openmlbb_path) as f:
                data = json.load(f)

            id_to_name = {}
            for rec in data.get("data", {}).get("records", []):
                d = rec.get("data", {})
                hero_id = d.get("hero_id")
                name = d.get("hero", {}).get("data", {}).get("name", "")
                if hero_id and name:
                    id_to_name[hero_id] = name

            for rec in data.get("data", {}).get("records", []):
                d = rec.get("data", {})
                name = d.get("hero", {}).get("data", {}).get("name", "")
                if not name:
                    continue

                rel = d.get("relation", {})

                # weak -> countered_by
                weak_ids = rel.get("weak", {}).get("target_hero_id", [])
                weak_names = [id_to_name[i] for i in weak_ids if i != 0 and i in id_to_name]
                if weak_names:
                    existing = self._countered_by.get(name, [])
                    merged = list(dict.fromkeys(existing + weak_names))
                    self._countered_by[name] = merged

                # strong -> counters
                strong_ids = rel.get("strong", {}).get("target_hero_id", [])
                strong_names = [id_to_name[i] for i in strong_ids if i != 0 and i in id_to_name]
                if strong_names:
                    existing = self._counters.get(name, [])
                    merged = list(dict.fromkeys(existing + strong_names))
                    self._counters[name] = merged

                # assist -> synergies
                assist_ids = rel.get("assist", {}).get("target_hero_id", [])
                assist_names = [id_to_name[i] for i in assist_ids if i != 0 and i in id_to_name]
                if assist_names:
                    existing = self._synergies.get(name, [])
                    merged = list(dict.fromkeys(existing + assist_names))
                    self._synergies[name] = merged

    def _load_synergy_data(self, data_dir: Path):
        """Load synergy data from mlbb.io overviews and hero_meta.json."""
        # Source 1: mlbb_io_overviews.json
        overviews_path = data_dir / "mlbb_io_overviews.json"
        if overviews_path.exists():
            with open(overviews_path) as f:
                data = json.load(f)
            for hero_name, info in data.items():
                syns = info.get("synergies", [])
                if syns:
                    existing = self._synergies.get(hero_name, [])
                    merged = list(dict.fromkeys(existing + syns))
                    self._synergies[hero_name] = merged

        # Source 2: hero_meta.json synergies
        meta_path = data_dir / "hero_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                data = json.load(f)
            for hero in data.get("data", []):
                name = hero.get("hero_name", "")
                syns = [s.get("heroname") for s in hero.get("synergies", []) if s.get("heroname")]
                if name and syns:
                    existing = self._synergies.get(name, [])
                    merged = list(dict.fromkeys(existing + syns))
                    self._synergies[name] = merged

    def _load_stats(self, data_dir: Path):
        """Load hero stats from mlbb.io stats."""
        stats_path = data_dir / "mlbb_io_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                data = json.load(f)
            for hero_name, info in data.items():
                self._stats[hero_name] = {
                    "win_rate": info.get("win_rate", 50.0) / 100.0,
                    "pick_rate": info.get("pick_rate", 0.0),
                    "ban_rate": info.get("ban_rate", 0.0),
                    "role": info.get("role", []),
                    "lane": info.get("lane", []),
                    "speciality": info.get("speciality", []),
                }

    def get_counter_confidence(self, hero_name: str, enemy_name: str) -> Tuple[float, str]:
        """Check if hero_name counters enemy_name, return confidence level.

        Counter data:
            - hero_counters_openmlbb.json: "counters" = heroes this hero beats
                Check: enemy_name in hero_name.counters
            - hero_meta.json: "counters" = heroes that beat this hero
                Check: hero_name in enemy_name.countered_by

        Both checks answer the same question: "Does hero_name counter enemy_name?"

        Returns:
            (1.0, "HIGH") = both sources agree
            (0.67, "MEDIUM") = only openmlbb agrees (has win_rate stats)
            (0.33, "LOW") = only meta agrees (no stats)
            (0.0, "NONE") = no source says counter
        """
        in_openmlbb = enemy_name in self._counters.get(hero_name, [])
        in_meta = hero_name in self._countered_by.get(enemy_name, [])

        if in_openmlbb and in_meta:
            return 1.0, "HIGH"
        elif in_openmlbb:
            return 0.67, "MEDIUM"
        elif in_meta:
            return 0.33, "LOW"
        else:
            return 0.0, "NONE"

    def get_heroes_that_beat(self, hero_name: str) -> List[str]:
        """Get heroes that counter (beat) the given hero."""
        return self._countered_by.get(hero_name, [])

    def get_heroes_that_hero_beats(self, hero_name: str) -> List[str]:
        """Get heroes that the given hero counters (beats)."""
        return self._counters.get(hero_name, [])

    def get_synergies(self, hero_name: str) -> List[str]:
        """Get heroes that synergize with the given hero."""
        return self._synergies.get(hero_name, [])

    def get_hero_stats(self, hero_name: str) -> dict:
        """Get stats for a hero."""
        return self._stats.get(hero_name, {
            "win_rate": 0.5,
            "pick_rate": 0.0,
            "ban_rate": 0.0,
            "role": [],
            "lane": [],
            "speciality": [],
        })

    def get_hero_lanes(self, hero_name: str) -> List[str]:
        """Get lanes a hero can play."""
        stats = self.get_hero_stats(hero_name)
        return stats.get("lane", [])

    def get_hero_roles(self, hero_name: str) -> List[str]:
        """Get roles a hero can play."""
        stats = self.get_hero_stats(hero_name)
        return stats.get("role", [])
