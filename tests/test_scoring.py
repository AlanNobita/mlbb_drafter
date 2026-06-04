"""Tests for hero scoring engine."""
import pytest
from server.recommendation.scoring import HeroScorer
from server.recommendation.scoring_data import ScoringData


class TestHeroScorer:
    def setup_method(self):
        self.data = ScoringData()
        self.scorer = HeroScorer(self.data)

    def test_baseline_score(self):
        """Empty draft: hero gets baseline (0.5) + win_rate bonus only."""
        score = self.scorer.score_hero("Suyou", [], [])
        assert 0.0 <= score <= 1.0
        # Should be > 0.5 because Suyou has win_rate > 50%
        assert score > 0.5

    def test_counter_boost_high_confidence(self):
        """Harith beats Suyou (HIGH confidence) -> Harith should rank high vs Suyou."""
        score_with_enemy = self.scorer.score_hero("Harith", [], ["Suyou"])
        score_without = self.scorer.score_hero("Harith", [], [])
        assert score_with_enemy > score_without

    def test_counter_boost_medium_confidence(self):
        """Suyou beats Sun (MEDIUM confidence) -> Suyou should boost vs Sun."""
        score_with_enemy = self.scorer.score_hero("Suyou", [], ["Sun"])
        score_without = self.scorer.score_hero("Suyou", [], [])
        assert score_with_enemy > score_without

    def test_counter_boost_low_confidence(self):
        """Cyclops beats Harith (LOW confidence) -> Cyclops gets small boost."""
        score_with_enemy = self.scorer.score_hero("Cyclops", [], ["Harith"])
        score_without = self.scorer.score_hero("Cyclops", [], [])
        assert score_with_enemy > score_without

    def test_synergy_boost(self):
        """Suyou synergizes with allies -> Suyou should boost with synergy partners."""
        # Pick an ally that Suyou is known to synergize with
        ally = self.data.get_synergies("Suyou")[0] if self.data.get_synergies("Suyou") else "Freya"
        score_with_ally = self.scorer.score_hero("Suyou", [ally], [])
        score_without = self.scorer.score_hero("Suyou", [], [])
        assert score_with_ally > score_without

    def test_combination_bonus(self):
        """Hero that counters enemy + synergizes with ally gets combo bonus."""
        # Find a hero that both counters someone and synergizes with someone
        # Use a simple case: find a hero that counters Harith
        for hero in self.data.heroes:
            conf_harith, _ = self.data.get_counter_confidence(hero, "Harith")
            if conf_harith > 0:
                syns = self.data.get_synergies(hero)
                if syns:
                    ally = syns[0]
                    score_combo = self.scorer.score_hero(hero, [ally], ["Harith"])
                    score_counter_only = self.scorer.score_hero(hero, [], ["Harith"])
                    assert score_combo > score_counter_only
                    return
        pytest.skip("Could not find a hero that counters Harith and has synergies")

    def test_lane_bonus(self):
        """Filling needed lane gives +0.20 bonus."""
        # Suyou can play Jungle
        score_jungle = self.scorer.score_hero("Suyou", [], [], needed_lane="Jungle")
        # Pick a lane Suyou definitely cannot play
        score_wrong = self.scorer.score_hero("Suyou", [], [], needed_lane="Gold")
        # Filling needed lane should be higher
        assert score_jungle > score_wrong

    def test_banned_hero_score_zero(self):
        """Banned hero should get score 0."""
        score = self.scorer.score_hero("Suyou", [], [], enemy_bans=["Suyou"])
        assert score == 0.0

    def test_score_range(self):
        """Score should always be between 0 and 1."""
        for hero in list(self.data.heroes)[:20]:
            score = self.scorer.score_hero(hero, ["Angela"], ["Sun"])
            assert 0.0 <= score <= 1.0, f"{hero} score {score} out of range"

    def test_rank_heroes(self):
        """rank_heroes should return sorted list."""
        results = self.scorer.rank_heroes(
            available_heroes=["Suyou", "Sun", "Harith", "Layla"],
            ally_picks=["Angela"],
            enemy_picks=["Yve", "Claude"],
            needed_lane="Jungle",
            top_k=4
        )
        assert len(results) == 4
        # Sorted descending
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_rank_excludes_picks(self):
        """Already picked heroes should be excluded."""
        results = self.scorer.rank_heroes(
            available_heroes=["Suyou", "Sun", "Harith"],
            ally_picks=["Suyou"],
            enemy_picks=[],
            top_k=3
        )
        names = [n for n, _ in results]
        assert "Suyou" not in names

    def test_rank_excludes_bans(self):
        """Banned heroes should be excluded."""
        results = self.scorer.rank_heroes(
            available_heroes=["Suyou", "Sun"],
            ally_picks=[],
            enemy_picks=[],
            enemy_bans=["Suyou"],
            top_k=2
        )
        names = [n for n, _ in results]
        assert "Suyou" not in names
