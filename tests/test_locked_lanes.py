"""Tests for the locked-lane fix.

Bug: scoring.py's `needed_lane_taken` check used to assume that any ally
with the needed lane in their lane list was "taking" it. That misclassified
flex heroes (Freya on EXP, Harith on Gold) as also occupying Jungle, and
applied a -0.5 penalty to every jungle candidate.

Fix: when the user passes explicit lane assignments (locked_ally_lanes),
the scorer uses that set directly and ignores the ally's lane list.

These tests assert that:
  1. Without locked lanes, flex allies in the lane list still poison
     the score (legacy behavior preserved as fallback).
  2. With locked lanes, the same draft recovers the proper score.
  3. score_hero with the same ally list but no locked lane produces a
     different (lower) score than with the correct locked lane.
"""
import pytest
from server.recommendation.scoring import HeroScorer
from server.recommendation.scoring_data import ScoringData


@pytest.fixture
def scorer():
    return HeroScorer(ScoringData())


class TestLockedLanes:
    """Regression: per-pick lane info must flow into the lane-taken check."""

    def test_freya_on_exp_does_not_poison_jungle_score(self, scorer):
        """Freya is dual-lane (EXP, Jungle). If she's playing EXP, the
        scorer should not treat Jungle as taken."""
        # Aamon is a pure Jungle assassin who counters Claude
        score_no_locks = scorer.score_hero(
            "Aamon",
            ally_picks=["Marcel", "Yve", "Freya", "Harith"],
            enemy_picks=["Valentina", "Leomord", "Kalea", "Claude", "Guinevere"],
            needed_lane="Jungle",
            enemy_bans=[],
        )
        score_with_locks = scorer.score_hero(
            "Aamon",
            ally_picks=["Marcel", "Yve", "Freya", "Harith"],
            enemy_picks=["Valentina", "Leomord", "Kalea", "Claude", "Guinevere"],
            needed_lane="Jungle",
            enemy_bans=[],
            locked_ally_lanes={"roam", "mid", "exp", "gold"},
        )
        # Locking Freya=exp + Harith=gold frees Jungle, so the score must
        # be at least 0.4 higher (the LANE_PENALTY_TAKEN is -0.5).
        assert score_with_locks - score_no_locks >= 0.4, (
            f"Expected locked score to recover >0.4 vs unlocked "
            f"(no_locks={score_no_locks:.3f}, with_locks={score_with_locks:.3f})"
        )

    def test_freya_on_jungle_does_poison_score(self, scorer):
        """Sanity check: if Freya IS on jungle, the penalty should apply."""
        score_freya_jungle = scorer.score_hero(
            "Aamon",
            ally_picks=["Marcel", "Yve", "Freya"],
            enemy_picks=["Claude"],
            needed_lane="Jungle",
            enemy_bans=[],
            locked_ally_lanes={"roam", "mid", "jungle"},
        )
        score_freya_exp = scorer.score_hero(
            "Aamon",
            ally_picks=["Marcel", "Yve", "Freya"],
            enemy_picks=["Claude"],
            needed_lane="Jungle",
            enemy_bans=[],
            locked_ally_lanes={"roam", "mid", "exp"},
        )
        assert score_freya_exp > score_freya_jungle, (
            f"Freya on EXP should not be penalized "
            f"(jungle={score_freya_jungle:.3f}, exp={score_freya_exp:.3f})"
        )

    def test_generate_recommendations_respects_locked_lanes(self, scorer):
        """Top-K pipeline should also forward locked_ally_lanes.

        Post-refactor note: the trait counter system now correctly rewards
        Saber over Aamon here — Saber has armor_shred (counters Leomord's
        high_armor + Kalea's high_armor) AND counters Claude's squishy_burst
        (3 matches → 0.45 trait bonus, capped). Aamon only matches Claude
        (1 match → 0.15 trait bonus). The principle tested here is still
        valid: locked_lanes unlocks the lane-taken penalty, allowing
        high-scoring jungle candidates to surface.
        """
        available = [h for h in [
            "Aamon", "Sun", "Saber", "Harley", "Hayabusa", "Granger",
            "Lancelot", "Fanny",
        ]]
        # Recs without locks: -0.5 penalty on all jungle picks
        recs_no_locks = scorer.generate_recommendations(
            available_heroes=available,
            ally_picks=["Marcel", "Yve", "Freya", "Harith"],
            enemy_picks=["Valentina", "Leomord", "Kalea", "Claude", "Guinevere"],
            needed_lane="Jungle",
            top_k=3,
        )
        # Same draft, with locks: penalty removed
        recs_with_locks = scorer.generate_recommendations(
            available_heroes=available,
            ally_picks=["Marcel", "Yve", "Freya", "Harith"],
            enemy_picks=["Valentina", "Leomord", "Kalea", "Claude", "Guinevere"],
            needed_lane="Jungle",
            top_k=3,
            locked_ally_lanes={"roam", "mid", "exp", "gold"},
        )
        # Aamon must appear in top-K with locks AND score much higher than
        # without locks (the lock effect is the principle being tested).
        locked_names = [name for name, _ in recs_with_locks]
        assert "Aamon" in locked_names, (
            f"Aamon should be in top-K with locks, got {locked_names}"
        )
        aamon_locked = next(s for n, s in recs_with_locks if n == "Aamon")
        # Find Aamon's score without locks (may not be in top-K at all).
        unlocked_names = [name for name, _ in recs_no_locks]
        aamon_unlocked = next(
            (s for n, s in recs_no_locks if n == "Aamon"),
            scorer.score_hero(
                "Aamon",
                ally_picks=["Marcel", "Yve", "Freya", "Harith"],
                enemy_picks=["Valentina", "Leomord", "Kalea", "Claude", "Guinevere"],
                needed_lane="Jungle",
                locked_ally_lanes=None,
            ),
        )
        # The lock effect must boost Aamon by at least 0.4 (LANE_PENALTY_TAKEN).
        assert aamon_locked - aamon_unlocked >= 0.4, (
            f"Lock effect on Aamon: locked={aamon_locked:.3f}, "
            f"unlocked={aamon_unlocked:.3f}, diff={aamon_locked - aamon_unlocked:.3f}"
        )

    def test_empty_locked_lanes_means_nothing_taken(self, scorer):
        """An explicit empty locked set means "user has assigned no lanes"
        which is different from the legacy fallback. The explicit-empty
        path is the correct one: Jungle is NOT taken, so Aamon gets
        LANE_BONUS_FULFILL."""
        score_no_param = scorer.score_hero(
            "Aamon", ally_picks=["Freya"], enemy_picks=["Claude"],
            needed_lane="Jungle", enemy_bans=[],
        )
        score_empty = scorer.score_hero(
            "Aamon", ally_picks=["Freya"], enemy_picks=["Claude"],
            needed_lane="Jungle", enemy_bans=[],
            locked_ally_lanes=set(),
        )
        # No-param falls back to legacy "Freya CAN play jungle" → poison
        # Empty set explicitly says "nothing is locked" → no poison
        assert score_empty > score_no_param
