"""Tests for the dynamic trait counter system (Gap 4 refactor).

Covers:
    - Per-enemy trait cross-reference (linearly-scaled bonus)
    - MAX_TRAIT_BONUS cap on stacking threat
    - Cross-threat matches (one candidate vs an enemy with multiple threats)
    - Auto-derived traits (backline_threat / dive / anti_cc / high_armor / high_burst)
    - Pre-computed trait set API (get_hero_traits_set)
    - Safe defaults for unknown heroes
    - Gord is properly placed in Mage - Burst Mage and has poke_kite
    - Glorious Setter's bad_against is intentionally empty (no static penalty)
    - enemy_has_trait() method has been removed
"""
import pytest
from server.recommendation.scoring import HeroScorer
from server.recommendation.scoring_data import ScoringData


@pytest.fixture
def data() -> ScoringData:
    return ScoringData()


@pytest.fixture
def scorer() -> HeroScorer:
    return HeroScorer(ScoringData())


class TestTraitCounterBasic:
    """Dynamic per-enemy cross-reference scoring."""

    def test_no_threat_no_bonus(self, scorer):
        """Empty enemy roster: zero trait bonus regardless of candidate traits."""
        # X.Borg has true_damage; with no enemies there's nothing to counter.
        score = scorer.score_hero("X.Borg", ally_picks=[], enemy_picks=[])
        assert score > 0  # still gets baseline

    def test_single_threat_match(self, scorer):
        """One enemy with high_sustain + candidate with anti_heal -> +0.15."""
        # Baxia has anti_heal; Uranus has high_sustain.
        s_with = scorer.score_hero("Baxia", ally_picks=[], enemy_picks=["Uranus"])
        s_without = scorer.score_hero("Baxia", ally_picks=[], enemy_picks=["Layla"])
        # Anti-heal vs sustain should boost by 0.15.
        assert s_with - s_without >= 0.14, (
            f"anti_heal vs high_sustain should add +0.15, got {s_with - s_without:.3f}"
        )

    def test_multi_threat_match(self, scorer):
        """Two sustain enemies + anti_heal candidate -> 2 * 0.15 = +0.30."""
        s_one = scorer.score_hero("Baxia", ally_picks=[], enemy_picks=["Uranus"])
        s_two = scorer.score_hero(
            "Baxia", ally_picks=[], enemy_picks=["Uranus", "Estes"]
        )
        # Adding Estes (also high_sustain) should add another +0.15.
        assert s_two - s_one >= 0.14, (
            f"second sustain enemy should add +0.15, got {s_two - s_one:.3f}"
        )

    def test_caps_at_max_trait_bonus(self, scorer):
        """4+ sustain enemies vs anti_heal candidate -> capped at MAX_TRAIT_BONUS."""
        # Force the bonus past the cap.
        s_four = scorer.score_hero(
            "Baxia",
            ally_picks=[],
            enemy_picks=["Uranus", "Estes", "Ruby", "Alpha"],
        )
        s_zero = scorer.score_hero("Baxia", ally_picks=[], enemy_picks=["Layla"])
        # MAX_TRAIT_BONUS = 0.45 (3 matches worth, 4th saturates).
        delta = s_four - s_zero
        # The full score_hero path has other variables, so the trait delta alone
        # should be no greater than the cap, allowing for a tiny numerical floor.
        assert delta <= 0.46, f"trait bonus exceeded MAX_TRAIT_BONUS: {delta:.3f}"
        assert delta >= 0.44, f"trait bonus did not reach cap: {delta:.3f}"

    def test_cross_threat_match(self, scorer):
        """Enemy with TWO threats: high_sustain AND high_armor.
        Candidate with one counter (true_damage) that bridges both -> +0.30.
        """
        # Uranus has high_sustain (curated) and high_armor (auto-derived: in
        # Tank - Stone Wall + Fighter - Berserker).
        s_uranus = scorer.score_hero(
            "X.Borg", ally_picks=[], enemy_picks=["Uranus"]
        )
        s_neutral = scorer.score_hero(
            "X.Borg", ally_picks=[], enemy_picks=["Layla"]
        )
        # X.Borg has true_damage; Uranus has BOTH threats; expect +0.30.
        assert s_uranus - s_neutral >= 0.29, (
            f"true_damage vs (sustain+armor) enemy should add +0.30, "
            f"got {s_uranus - s_neutral:.3f}"
        )

    def test_unknown_hero_safe(self, scorer):
        """Unknown heroes in either side must not raise."""
        score = scorer.score_hero(
            "Baxia", ally_picks=[], enemy_picks=["Nonexistent12345"]
        )
        assert isinstance(score, (int, float))
        # An unknown enemy should contribute zero to the trait bonus.
        score_unknown_enemy = scorer.score_hero(
            "Baxia", ally_picks=[], enemy_picks=["Nonexistent12345"]
        )
        score_no_enemy = scorer.score_hero("Baxia", ally_picks=[], enemy_picks=[])
        assert abs(score_unknown_enemy - score_no_enemy) < 0.01


class TestGordPlacement:
    """Regression: Gord must be a Burst Mage and carry the poke_kite trait."""

    def test_gord_in_burst_mage_archetype(self, data):
        """Gord belongs in Mage - Burst Mage (was missing in old 12-hero list)."""
        assert "Mage - Burst Mage" in data.get_hero_archetypes("Gord")

    def test_gord_has_poke_kite_trait(self, data):
        """Gord should carry poke_kite from the user-curated trait list."""
        traits = data.get_hero_traits("Gord")
        assert "poke_kite" in traits, f"Gord poke_kite missing: {traits}"

    def test_gord_has_high_burst_auto_derived(self, data):
        """Gord is in Mage - Burst Mage -> high_burst auto-derived."""
        traits = data.get_hero_traits("Gord")
        assert "high_burst" in traits, f"Gord high_burst missing: {traits}"


class TestAutoDerivedTraits:
    """Auto-derivation rules in scoring_data._AUTO_TRAIT_RULES."""

    def test_assassin_oneshotter_gets_backline_threat(self, data):
        """Hayabusa (Assassin - Oneshotter) gets backline_threat auto-derived."""
        traits = data.get_hero_traits("Hayabusa")
        assert "backline_threat" in traits, (
            f"Hayabusa should be auto-tagged backline_threat: {traits}"
        )

    def test_any_assassin_archetype_gets_dive(self, data):
        """Any hero in any Assassin archetype (Nimble Speedster, Oneshotter,
        Prey Hunter) gets dive auto-derived."""
        for hero in ["Lancelot", "Hayabusa", "Fanny"]:
            traits = data.get_hero_traits(hero)
            assert "dive" in traits, (
                f"{hero} (Assassin archetype) should be auto-tagged dive: {traits}"
            )

    def test_tank_vanguard_gets_high_armor(self, data):
        """Tigreal (Tank - Vanguard) -> high_armor auto-derived."""
        traits = data.get_hero_traits("Tigreal")
        assert "high_armor" in traits, (
            f"Tigreal should be auto-tagged high_armor: {traits}"
        )

    def test_disengage_heroes_get_anti_cc(self, data):
        """Akai has disengage -> anti_cc auto-derived."""
        traits = data.get_hero_traits("Akai")
        assert "disengage" in traits
        assert "anti_cc" in traits, (
            f"Akai (disengage) should be auto-tagged anti_cc: {traits}"
        )


class TestGloriousSetterSpec:
    """Glorious Setter's bad_against is intentionally empty by spec."""

    def test_glorious_setter_no_bad_against(self, data):
        """Static bad_against for Tank - Glorious Setter is [] by design."""
        rules = data._archetype_matchups["Tank - Glorious Setter"]
        assert rules["bad_against"] == []


class TestTraitAPI:
    """Pre-computed set API and graceful warnings."""

    def test_get_hero_traits_set_returns_set(self, data):
        """get_hero_traits_set returns a set (not list) for fast lookup."""
        s = data.get_hero_traits_set("Baxia")
        assert isinstance(s, set)
        assert "anti_heal" in s

    def test_get_hero_traits_set_unknown_returns_empty(self, data):
        """Unknown hero: empty set, no warning spam per call."""
        s = data.get_hero_traits_set("Nonexistent12345")
        assert s == set()

    def test_enemy_has_trait_method_removed(self, data):
        """enemy_has_trait is gone; replaced by per-hero set lookup."""
        assert not hasattr(data, "enemy_has_trait"), (
            "enemy_has_trait should be removed; use get_hero_traits_set()"
        )


class TestMaxTraitBonusEnforcement:
    """MAX_TRAIT_BONUS must hold at the score_hero boundary, not just inside
    _scan_enemy_traits."""

    def test_max_trait_bonus_enforced_in_score_hero(self, scorer):
        """5+ sustain enemies vs Baxia (anti_heal) -> raw trait bonus capped."""
        s_five = scorer.score_hero(
            "Baxia",
            ally_picks=[],
            enemy_picks=["Uranus", "Estes", "Ruby", "Alpha", "Balmond"],
        )
        s_zero = scorer.score_hero("Baxia", ally_picks=[], enemy_picks=[])
        delta = s_five - s_zero
        # Cap should hold even at 5 sustain enemies; allow 0.01 numerical floor.
        assert delta <= 0.46, f"trait bonus exceeded MAX_TRAIT_BONUS: {delta:.3f}"
