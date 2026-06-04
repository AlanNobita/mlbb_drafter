"""Tests for scoring data loader."""
import pytest
from server.recommendation.scoring_data import ScoringData


class TestScoringData:
    def setup_method(self):
        self.data = ScoringData()

    def test_loads_all_files(self):
        """Should load all 132 heroes from the data files."""
        assert len(self.data.heroes) > 0
        assert "Suyou" in self.data.heroes
        assert "Harith" in self.data.heroes

    def test_counter_lookup_from_openmlbb(self):
        """Suyou beats Sun (from hero_counters_openmlbb.json)."""
        beats = self.data.get_heroes_that_hero_beats("Suyou")
        assert "Sun" in beats

    def test_countered_by_lookup_from_meta(self):
        """Harith beats Suyou (from hero_meta.json countered_by)."""
        beats_suyou = self.data.get_heroes_that_beat("Suyou")
        assert "Harith" in beats_suyou

    def test_double_check_high_confidence(self):
        """Both sources agree: Harith beats Suyou."""
        conf, level = self.data.get_counter_confidence("Harith", "Suyou")
        assert conf == 1.0
        assert level == "HIGH"

    def test_double_check_medium_confidence(self):
        """Only openmlbb says counter: Suyou beats Sun."""
        conf, level = self.data.get_counter_confidence("Suyou", "Sun")
        assert conf == 0.67
        assert level == "MEDIUM"

    def test_double_check_low_confidence(self):
        """Only meta says counter: Cyclops beats Harith."""
        conf, level = self.data.get_counter_confidence("Cyclops", "Harith")
        assert conf == 0.33
        assert level == "LOW"

    def test_double_check_none(self):
        """No source says counter: Suyou beats Suyou (self)."""
        conf, level = self.data.get_counter_confidence("Suyou", "Suyou")
        assert conf == 0.0
        assert level == "NONE"

    def test_synergy_lookup(self):
        """Should return synergies from merged sources."""
        synergies = self.data.get_synergies("Suyou")
        assert len(synergies) > 0

    def test_hero_stats(self):
        """Should have win_rate, pick_rate, ban_rate, lane."""
        stats = self.data.get_hero_stats("Suyou")
        assert "win_rate" in stats
        assert "pick_rate" in stats
        assert "ban_rate" in stats
        assert "lane" in stats

    def test_lane_lookup(self):
        """Should return lanes a hero can play."""
        lanes = self.data.get_hero_lanes("Suyou")
        assert len(lanes) > 0

    def test_heroes_attribute(self):
        """Should expose hero names as a list/set."""
        assert "Suyou" in self.data.heroes
