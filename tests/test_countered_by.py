"""Tests for the 'Countered by' field in the Recommendation payload.

Backend exposes which currently-picked enemy heroes counter the candidate.
This is the inverse of the existing 'counters' field.
"""
import pytest
from server.recommendation.scoring_data import ScoringData


@pytest.fixture
def data() -> ScoringData:
    return ScoringData()


class TestGetHeroesThatBeat:
    def test_returns_list_from_meta(self, data):
        """get_heroes_that_beat returns _countered_by entries."""
        # X.Borg is beaten by Zilong per hero_meta.json.
        result = data.get_heroes_that_beat("X.Borg")
        assert "Zilong" in result, f"X.Borg should be countered by Zilong: {result}"

    def test_unknown_hero_returns_empty(self, data):
        """Unknown hero returns an empty list, never raises."""
        assert data.get_heroes_that_beat("Nonexistent12345") == []


class TestBuildRecommendationDict:
    """Test the per-recommendation dict construction helper.

    Extracted so we can unit-test the countered_by filter logic without
    standing up the full WebSocket handler.
    """

    def test_countered_by_filters_enemy_roster(self, data):
        """countered_by contains only enemies currently picked that beat the candidate."""
        from server.main import MLdrafterServer
        server = MLdrafterServer.__new__(MLdrafterServer)
        server.scoring_data = data
        candidate = "X.Borg"
        # X.Borg is beaten by Zilong; use Zilong + an irrelevant hero.
        enemy_set = {"Zilong", "Layla"}
        result = server._build_recommendation_dict(
            hero=candidate,
            score=0.75,
            breakdown={},
            ally_set=set(),
            enemy_set=enemy_set,
        )
        assert "countered_by" in result
        # Zilong beats X.Borg; Layla doesn't.
        assert result["countered_by"] == ["Zilong"]

    def test_countered_by_empty_when_no_enemies(self, data):
        """countered_by is [] when enemy_set is empty."""
        from server.main import MLdrafterServer
        server = MLdrafterServer.__new__(MLdrafterServer)
        server.scoring_data = data
        result = server._build_recommendation_dict(
            hero="X.Borg",
            score=0.5,
            breakdown={},
            ally_set=set(),
            enemy_set=set(),
        )
        assert result["countered_by"] == []

    def test_countered_by_empty_when_no_match(self, data):
        """countered_by is [] when no current enemy beats the candidate."""
        from server.main import MLdrafterServer
        server = MLdrafterServer.__new__(MLdrafterServer)
        server.scoring_data = data
        # Layla has no counter relationships in the data.
        result = server._build_recommendation_dict(
            hero="Layla",
            score=0.5,
            breakdown={},
            ally_set=set(),
            enemy_set={"Roger", "Karrie"},
        )
        assert result["countered_by"] == []

    def test_preserves_existing_counters_and_synergies(self, data):
        """Existing counters and synergies fields still appear alongside countered_by."""
        from server.main import MLdrafterServer
        server = MLdrafterServer.__new__(MLdrafterServer)
        server.scoring_data = data
        result = server._build_recommendation_dict(
            hero="Baxia",
            score=0.6,
            breakdown={},
            ally_set={"Angela"},
            enemy_set={"Uranus"},
        )
        assert "counters" in result
        assert "synergies" in result
        assert "countered_by" in result
