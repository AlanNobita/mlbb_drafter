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
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

_logger = logging.getLogger(__name__)


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
        # archetype -> [hero names]
        self._archetypes: Dict[str, List[str]] = {}
        # archetype -> {"good_against": [...], "bad_against": [...]}
        self._archetype_matchups: Dict[str, dict] = {}
        # hero -> [trait strings] (e.g., ["high_sustain"], ["anti_heal"])
        self._hero_traits: Dict[str, List[str]] = {}
        # hero -> pre-computed set(traits) for hot-path lookups. Built by
        # _apply_auto_traits() so the scoring loop never allocates a new set.
        self._hero_traits_set: Dict[str, Set[str]] = {}
        # Heroes we've already warned about for missing trait data (rate limit).
        self._warned_heroes: Set[str] = set()

        self._load_counter_data(data_dir)
        self._load_synergy_data(data_dir)
        self._load_stats(data_dir)
        self._load_archetypes()

        # Build hero set from all loaded data
        all_heroes = set(self._counters.keys()) | set(self._countered_by.keys()) | set(self._stats.keys())
        self.heroes = all_heroes

        # Apply auto-derived traits now that the full hero set is known.
        self._apply_auto_traits()

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

    def _load_archetypes(self):
        """Load hero archetype data from bundled JSON file.

        Source: server/recommendation/data/hero_archetypes.json
        Structure:
            - Archetype name -> [hero names] (roster)
            - matchup_rules: archetype -> {good_against: [...], bad_against: [...]}
            - traits: trait_name -> [hero names] (curated)
        """
        archetypes_path = Path(__file__).parent / "data" / "hero_archetypes.json"
        if not archetypes_path.exists():
            return

        try:
            with open(archetypes_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        for key, value in data.items():
            if key == "matchup_rules":
                self._archetype_matchups = value if isinstance(value, dict) else {}
            elif key == "traits":
                # New shape: trait_name -> [hero_names].
                # Invert to hero -> [trait_names] for O(1) lookup.
                if isinstance(value, dict):
                    self._hero_traits = {}
                    for trait_name, heroes in value.items():
                        if not isinstance(heroes, list):
                            continue
                        for hero in heroes:
                            self._hero_traits.setdefault(hero, []).append(trait_name)
            elif isinstance(value, list):
                self._archetypes[key] = value

    # Auto-derived trait rules. Each entry: (derived_trait, predicate).
    # Predicates receive (hero_archetypes_set, hero_traits_set, hero_stats_dict)
    # and return True if the derived trait applies to this hero.
    # Auto-derived traits are merged into _hero_traits[hero] at load time.
    # The scoring loop in scoring.py NEVER sees these predicates — it reads
    # the flat trait list via get_hero_traits() / get_hero_traits_set().
    _AUTO_TRAIT_RULES = [
        (
            "backline_threat",
            lambda archs, traits, stats: bool(
                archs & {"Assassin - Oneshotter", "Assassin - Prey Hunter"}
            ),
        ),
        (
            "dive",
            # any() short-circuits on first match; no interim set allocation.
            lambda archs, traits, stats: (
                any(a.startswith("Assassin -") for a in archs)
                or "high_mobility" in traits
            ),
        ),
        (
            "anti_cc",
            lambda archs, traits, stats: "disengage" in traits,
        ),
        (
            "high_armor",
            lambda archs, traits, stats: bool(
                archs & {
                    "Tank - Vanguard", "Tank - Stone Wall",
                    "Fighter - Juggernaut", "Fighter - Berserker",
                }
            ),
        ),
        (
            "high_burst",
            lambda archs, traits, stats: bool(
                archs & {"Assassin - Oneshotter", "Mage - Burst Mage"}
            ),
        ),
    ]

    def _apply_auto_traits(self) -> None:
        """Run _AUTO_TRAIT_RULES against every hero in the roster. Idempotent.

        Auto-derived traits are merged into self._hero_traits[hero] alongside
        user-curated traits from hero_archetypes.json. Then a parallel
        _hero_traits_set dict is built once so the scoring loop can do
        pure hash lookups without re-allocating a set per call.
        """
        for hero in self.heroes:
            traits_list = self._hero_traits.setdefault(hero, [])
            existing = set(traits_list)
            archs = set(self.get_hero_archetypes(hero))
            for derived_trait, predicate in self._AUTO_TRAIT_RULES:
                if derived_trait not in existing and predicate(
                    archs, existing, self._stats.get(hero, {})
                ):
                    traits_list.append(derived_trait)
                    existing.add(derived_trait)
        # Pre-compile the set view for hot-path lookups.
        self._hero_traits_set = {
            hero: set(traits) for hero, traits in self._hero_traits.items()
        }

    # Cold-start fallback: map upstream base role tags to a generic archetype vector.
    # Used when a hero is missing from the custom JSON database (e.g., newly released
    # or freshly re-worked heroes like Arlott at the time of data collection).
    # The mapping is intentionally generic — the goal is baseline matchup coverage,
    # not perfect classification.
    _ROLE_TO_ARCHETYPE = {
        "Fighter": "Fighter - Juggernaut",
        "Assassin": "Assassin - Prey Hunter",
        "Mage": "Mage - Burst Mage",
        "Marksman": "Marksman - Crit based",
        "Tank": "Tank - Stone Wall",
        "Support": "Support - Enchanter",
    }

    def get_hero_archetypes(self, hero_name: str) -> List[str]:
        """Get all archetypes a hero belongs to.

        Strategy:
            1. First, look up in the custom JSON roster (preferred).
            2. If empty, fall back to inferring from upstream `role` field
               (e.g., "Fighter" -> "Fighter - Juggernaut").
            3. If still empty, return [] gracefully.

        Returns:
            List of archetype names (e.g., ["Tank - Vanguard", "Fighter - Juggernaut"])
            Empty list if hero has no archetype or role data.
        """
        result = []
        for archetype, heroes in self._archetypes.items():
            if hero_name in heroes:
                result.append(archetype)

        if not result:
            result = self._infer_archetypes_from_role(hero_name)

        return result

    def _infer_archetypes_from_role(self, hero_name: str) -> List[str]:
        """Cold-start inference: map upstream `role` field to generic archetype vectors.

        Used when a hero is missing from the custom JSON database. The mapping is
        generic by design — we want baseline matchup coverage for new/reworked
        heroes, not perfect classification.

        Example:
            "Fighter/Assassin" -> ["Fighter - Juggernaut", "Assassin - Prey Hunter"]
            "Mage" -> ["Mage - Burst Mage"]
            [] -> []
        """
        stats = self._stats.get(hero_name, {})
        roles = stats.get("role", []) or []
        archetypes = []
        for role in roles:
            mapped = self._ROLE_TO_ARCHETYPE.get(role)
            if mapped and mapped not in archetypes:
                archetypes.append(mapped)
        return archetypes

    def get_archetype_matchups(self, archetype: str) -> dict:
        """Get matchup rules for an archetype.

        Returns:
            {"good_against": [...], "bad_against": [...]}
            Safe defaults if archetype is unknown.
        """
        return self._archetype_matchups.get(
            archetype, {"good_against": [], "bad_against": []}
        )

    def get_hero_traits(self, hero_name: str) -> List[str]:
        """Get traits for a hero (e.g., ["high_sustain"], ["anti_heal"]).

        Logs a one-time warning per unknown hero. Returns an empty list
        for heroes with no trait data (never raises).

        Returns:
            List of trait strings. Empty list if hero has no traits.
        """
        if hero_name not in self._hero_traits and hero_name not in self._warned_heroes:
            self._warned_heroes.add(hero_name)
            _logger.warning(f"No trait data for hero {hero_name!r}; returning []")
        return self._hero_traits.get(hero_name, [])

    def get_hero_traits_set(self, hero_name: str) -> Set[str]:
        """Pre-computed set view of a hero's traits. Use in hot loops.

        The set is built once at load time by _apply_auto_traits(). Returns
        an empty set for unknown heroes (no allocation, no warning spam
        per call — the warning is owned by get_hero_traits()).
        """
        return self._hero_traits_set.get(hero_name, set())
