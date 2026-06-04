"""Hero scoring engine for draft recommendation.

Scoring formula:
    baseline = 0.5
    win_rate_bonus = (win_rate - 0.5) * 0.3
    counter_bonus = sum(confidence * 0.15) for each countered enemy
    synergy_bonus = min(count * 0.10, 0.30)
    combo_bonus = 0.20 if (counter > 0 and synergy > 0)
    ban_rate_bonus = 0.10 if ban_rate > 5.0
    lane_bonus = +0.20 if fills needed, -0.30 if wrong
    stability_bonus = team composition check
"""
from typing import Dict, List, Optional, Set, Tuple
from .scoring_data import ScoringData


class HeroScorer:
    """Scores heroes based on counter, synergy, win rate, ban rate, and lane fit."""

    def __init__(self, data: ScoringData):
        self.data = data

    def score_hero(
        self,
        hero_name: str,
        ally_picks: List[str],
        enemy_picks: List[str],
        needed_lane: Optional[str] = None,
        enemy_bans: Optional[List[str]] = None,
    ) -> float:
        """Score a hero for the current draft state.

        Args:
            hero_name: Hero to score
            ally_picks: Heroes already picked by our team
            enemy_picks: Heroes already picked by enemy team
            needed_lane: Lane that needs to be filled (e.g., "Jungle", "EXP")
            enemy_bans: Heroes banned by enemy team

        Returns:
            Score between 0.0 and 1.0
        """
        if enemy_bans and hero_name in enemy_bans:
            return 0.0

        # 1. Baseline
        score = 0.5

        # 2. Win rate bonus
        stats = self.data.get_hero_stats(hero_name)
        win_rate = stats.get("win_rate", 0.5)
        score += (win_rate - 0.5) * 0.3

        # 3. Counter bonus: heroes that hero_name beats in enemy team
        countered_enemies = set()
        for enemy in enemy_picks:
            conf, _ = self.data.get_counter_confidence(hero_name, enemy)
            if conf > 0:
                countered_enemies.add(enemy)

        counter_bonus = 0.0
        for enemy in countered_enemies:
            conf, _ = self.data.get_counter_confidence(hero_name, enemy)
            counter_bonus += conf * 0.15
        counter_bonus = min(counter_bonus, 0.45)
        score += counter_bonus

        # 4. Synergy bonus: heroes that synergize with ally team
        synergy_with_allies = set()
        hero_synergies = set(self.data.get_synergies(hero_name))
        for ally in ally_picks:
            if ally in hero_synergies:
                synergy_with_allies.add(ally)
            ally_synergies = set(self.data.get_synergies(ally))
            if hero_name in ally_synergies:
                synergy_with_allies.add(ally)

        synergy_bonus = min(len(synergy_with_allies) * 0.10, 0.30)
        score += synergy_bonus

        # 5. Combination bonus: counter enemy + synergy with ally
        if countered_enemies and synergy_with_allies:
            score += 0.20

        # 6. Ban rate bonus: high ban rate heroes are strong picks
        ban_rate = stats.get("ban_rate", 0.0)
        if ban_rate > 5.0:
            score += 0.10

        # 7. Lane check
        if needed_lane:
            hero_lanes_raw = self.data.get_hero_lanes(hero_name)
            hero_lanes = [l.replace(" Lane", "") for l in hero_lanes_raw]
            primary_lane = hero_lanes[0] if hero_lanes else ""
            primary_lane = primary_lane.replace(" Lane", "")

            if needed_lane in hero_lanes or needed_lane == primary_lane:
                score += 0.20
            elif hero_lanes and hero_lanes[0] != needed_lane:
                score -= 0.30

        # 8. Team stability
        stability = self.check_team_stability(ally_picks, hero_name, needed_lane)
        score += stability

        return max(0.0, min(1.0, score))

    def check_team_stability(
        self,
        current_picks: List[str],
        candidate_hero: str,
        needed_lane: Optional[str] = None,
    ) -> float:
        """Check if adding this hero makes the team composition stable.

        Returns:
            +0.20: Fills a needed role perfectly
            -0.30: Creates role overlap
            0.0: Neutral
        """
        if not needed_lane:
            return 0.0

        candidate_lanes = [
            l.replace(" Lane", "")
            for l in self.data.get_hero_lanes(candidate_hero)
        ]

        # Fills needed lane
        if needed_lane in candidate_lanes:
            return 0.20

        # Creates overlap with existing picks
        for pick in current_picks:
            pick_lanes = [
                l.replace(" Lane", "")
                for l in self.data.get_hero_lanes(pick)
            ]
            if pick_lanes and candidate_lanes:
                overlap = set(pick_lanes) & set(candidate_lanes)
                if overlap:
                    return -0.30

        return 0.0

    def rank_heroes(
        self,
        available_heroes: List[str],
        ally_picks: List[str],
        enemy_picks: List[str],
        needed_lane: Optional[str] = None,
        enemy_bans: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Rank available heroes for the current draft.

        Returns:
            List of (hero_name, score) tuples sorted by score descending.
        """
        scores = []
        for hero in available_heroes:
            if enemy_bans and hero in enemy_bans:
                continue
            if hero in ally_picks or hero in enemy_picks:
                continue
            score = self.score_hero(
                hero, ally_picks, enemy_picks, needed_lane, enemy_bans
            )
            scores.append((hero, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
