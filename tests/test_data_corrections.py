"""Tests for the data corrections applied to hero_archetypes.json.

Validates:
- Dyrroth has armor_shred (not true_damage)
- Masha is NOT in Tank - Stone Wall
- Mage - Summoner key does NOT exist (renamed to Utility - Summoner)
- Kagura is NOT in Utility - Summoner
- Zilong is NOT in Fighter - Stunlocker
- Tank - Glorious Setter has bad_against populated
"""
import pytest
from server.recommendation.scoring_data import ScoringData


class TestDataCorrections:
    def test_dyrroth_has_armor_shred(self):
        """Dyrroth should have armor_shred, not true_damage."""
        d = ScoringData()
        traits = d.get_hero_traits("Dyrroth")
        assert "armor_shred" in traits
        assert "true_damage" not in traits

    def test_masha_not_in_stone_wall(self):
        """Masha should be removed from Tank - Stone Wall after rework."""
        d = ScoringData()
        masha_archetypes = d.get_hero_archetypes("Masha")
        assert "Tank - Stone Wall" not in masha_archetypes, (
            f"Masha should not be in Stone Wall: {masha_archetypes}"
        )
        # Masha should still be in Berserker
        assert "Fighter - Berserker" in masha_archetypes

    def test_mage_summoner_key_renamed(self):
        """Mage - Summoner key should not exist; Utility - Summoner should."""
        d = ScoringData()
        assert "Mage - Summoner" not in d._archetypes, (
            "Old Mage - Summoner key should be removed"
        )
        assert "Utility - Summoner" in d._archetypes

    def test_kagura_not_in_utility_summoner(self):
        """Kagura (projectile, not summon) should be removed from Utility - Summoner."""
        d = ScoringData()
        utility_summoners = d._archetypes.get("Utility - Summoner", [])
        assert "Kagura" not in utility_summoners, (
            f"Kagura should not be a summoner: {utility_summoners}"
        )

    def test_zilong_not_in_stunlocker(self):
        """Zilong should be removed from Fighter - Stunlocker."""
        d = ScoringData()
        stunlockers = d._archetypes.get("Fighter - Stunlocker", [])
        assert "Zilong" not in stunlockers

    def test_glorious_setter_has_no_bad_against(self):
        """Tank - Glorious Setter's bad_against is intentionally empty.

        User explicitly decided Glorious Setter (Gatotkaca / Khufra style
        initiator tanks) has no clean hard counters among MLBB archetypes.
        The engine relies on the new dynamic trait system
        (high_mobility / hard_engage / crowd_control counter tags) instead
        of a static bad_against list for these tanks.
        """
        d = ScoringData()
        rules = d._archetype_matchups["Tank - Glorious Setter"]
        assert rules["bad_against"] == [], (
            f"Glorious Setter bad_against should remain empty by spec: {rules}"
        )


class TestPythonCodeUpdated:
    def test_magic_archetypes_excludes_utility_summoner(self):
        """scoring.py's _MAGIC_ARCHETYPES should not include Utility - Summoner
        (it contains non-mage heroes like Sun, Popol and Kupa)."""
        from server.recommendation.scoring import HeroScorer
        from server.recommendation.scoring_data import ScoringData
        s = HeroScorer(ScoringData())
        assert "Utility - Summoner" not in s._MAGIC_ARCHETYPES, (
            "Utility - Summoner contains non-mages; should not count as magic"
        )
        assert "Mage - Summoner" not in s._MAGIC_ARCHETYPES, (
            "Old Mage - Summoner key no longer exists"
        )

    def test_intent_summoner_maps_to_utility_summoner(self):
        """scoring.py's _INTENT_ARCHETYPES should include 'summoner' -> Utility - Summoner."""
        from server.recommendation.scoring import HeroScorer
        from server.recommendation.scoring_data import ScoringData
        s = HeroScorer(ScoringData())
        assert "summoner" in s._INTENT_ARCHETYPES
        assert "Utility - Summoner" in s._INTENT_ARCHETYPES["summoner"]

    def test_intent_mage_no_longer_includes_summoner(self):
        """scoring.py's 'mage' intent should not include Utility - Summoner
        (since it's a mix of mages and non-mages)."""
        from server.recommendation.scoring import HeroScorer
        from server.recommendation.scoring_data import ScoringData
        s = HeroScorer(ScoringData())
        mage_intent = s._INTENT_ARCHETYPES.get("mage", set())
        assert "Utility - Summoner" not in mage_intent
        assert "Mage - Summoner" not in mage_intent
