"""Hero scoring engine for draft recommendation.

Scoring formula (refactored - high-resolution raw values, no pre-sort clamping):

    Components (additive):
        1. Baseline:              0.20 flat
        2. Win rate:              (win_rate - 0.5) * 0.3  (Range: ±0.15)
        3. Counter bonus:         sum(confidence * 0.10) per countered enemy
                                  + archetype good_against / bad_against (±0.05 per matchup)
                                  Max contribution +0.25
        4. Synergy bonus:         min(count * 0.05, 0.20)  (Max +0.20)
        5. Combo bonus:           +0.10 if hero counters enemy AND synergizes with ally
        6. Ban rate bonus:        (ban_rate / 100 * 0.20) capped at +0.10  (continuous, not binary)
        7. Lane bonus:            +0.30 fulfill unpicked needed lane, -0.50 fill taken lane
        8. Team stability:        +0.10 introduces needed archetype, -0.20 redundancy
                                  Plus -0.08 per shared archetype with ally
        9. Trait counter:         +0.15 per (enemy_threat, candidate_counter) match
                                  Linear scaling, capped at +0.45 (MAX_TRAIT_BONUS)
        10. Magic/Frontline gap:  +0.15 each when team lacks magic damage / frontline
        11. User intent bump:     +0.30 if candidate archetype matches user_intent

Pipeline:
    1. score_hero() returns raw unclamped value
    2. rank_heroes() sorts all candidates by raw score (no early clamp)
    3. rank_heroes() applies Sub-Role Diversity Decay during top-K extraction
    4. Only the final top 5 selections are clamped to [0.0, 1.0] for the API contract
"""
from typing import Dict, List, Optional, Set, Tuple
from .scoring_data import ScoringData


class HeroScorer:
    """Scores heroes based on counter, synergy, archetype matchups, win rate, ban rate, and lane fit.

    Scores are returned as raw, high-resolution decimals. Clamping to [0.0, 1.0]
    happens only at the final API payload stage (top 5 output), not during
    intermediate ranking. This prevents the "5/5 ceiling saturation" problem
    where top meta heroes all hit 1.000 and sorting falls back to JSON load order.
    """

    # Component caps (raw values, before any final clamp)
    BASELINE = 0.20
    WIN_RATE_RANGE = 0.15  # ±0.15
    COUNTER_BONUS_MAX = 0.25
    SYNERGY_BONUS_MAX = 0.20
    COMBO_BONUS = 0.10
    BAN_RATE_BONUS_MAX = 0.10
    LANE_BONUS_FULFILL = 0.30
    LANE_PENALTY_TAKEN = -0.50
    STABILITY_BONUS_MAX = 0.10
    STABILITY_PENALTY_MAX = -0.20
    ARCHETYPE_REDUNDANCY_PENALTY = -0.08
    ARCHETYPE_DIVERSITY_BONUS = 0.04
    ARCHETYPE_MATCHUP_BONUS = 0.05
    ARCHETYPE_MATCHUP_PENALTY = -0.05

    # Per-counter contribution: confidence * multiplier, max contributions add up to COUNTER_BONUS_MAX
    COUNTER_PER_ENEMY = 0.10  # HIGH (1.0) * 0.10 = 0.10 per enemy

    # Trait counter system: dynamic per-enemy cross-reference.
    # Per successful (threat, counter) match, the candidate gets +0.15.
    # Linear scaling per match. Capped at MAX_TRAIT_BONUS to prevent inflation
    # when the enemy team stacks a singular threat (e.g. 3 sustain heroes).
    TRAIT_BONUS_PER_MATCH = 0.15
    MAX_TRAIT_BONUS = 0.45

    # Counter trait matrix: threat trait (left) -> set of counter-traits (right).
    # Both sides are flat strings. The scoring loop reads trait strings and
    # intersects with this map. No archetype lookups, no string-prefix checks.
    # "backline_threat" is the auto-derived tag for Assassin - Oneshotter /
    # Assassin - Prey Hunter (renamed from "assassin" to avoid role-name collision).
    _COUNTER_TRAIT_MAP = {
        "high_sustain":  {"anti_heal", "true_damage"},
        "high_armor":    {"armor_shred", "true_damage"},
        "squishy_burst": {"high_burst", "backline_threat", "dive"},
        "high_mobility": {"crowd_control", "anti_dash", "suppress"},
        "hard_engage":   {"disengage", "anti_cc", "poke_kite"},
        "heavy_shields": {"shield_breaker", "true_damage"},
        "poke_kite":     {"dive", "high_mobility", "backline_threat"},
    }

    # Diversity decay: penalty per duplicate sub-role in top picks
    DIVERSITY_DECAY = -0.05

    # Gaps 1 & 2: Global ally state bonuses
    NEEDS_MAGIC_BONUS = 0.15          # Mage candidates when team has no magic damage
    ALLY_VULNERABILITY_BONUS = 0.15   # Vanguard/Stone Wall when team is squishy

    # Gap 3: User intent strategy bump (flat baseline bump for archetype matches)
    USER_INTENT_BONUS = 0.30

    # Mapping: user_intent string -> set of archetype names that satisfy it.
    # Sub-archetype overrides (vanguard, stone_wall, dps_mage, etc.) are intentionally
    # narrow so the +0.30 strategic bump processes correctly and doesn't pollute
    # the search with off-archetype matches.
    _INTENT_ARCHETYPES = {
        # Core mappings
        "enchanter": {"Support - Enchanter"},
        "initiator": {"Support - Initiator", "Tank - Glorious Setter"},
        "tank": {"Tank - Vanguard", "Tank - Stone Wall"},
        "mage": {"Mage - Burst Mage", "Mage - Battlemage", "Mage - Control Mage",
                 "Mage - DPS Mage"},
        "summoner": {"Utility - Summoner"},
        "assassin": {"Assassin - Nimble Speedster", "Assassin - Prey Hunter"},
        "marksman": {"Marksman - Crit based", "Marksman - Skill based"},
        "fighter": {"Fighter - Juggernaut", "Fighter - Berserker", "Fighter - Stunlocker"},
        # Sub-archetype overrides
        "vanguard": {"Tank - Vanguard"},
        "stone_wall": {"Tank - Stone Wall"},
        "dps_mage": {"Mage - DPS Mage"},
        "prey_hunter": {"Assassin - Prey Hunter"},
        "nimble_speedster": {"Assassin - Nimble Speedster"},
        "control_mage": {"Mage - Control Mage"},
    }

    # Heroes explicitly allowed to flex to Roam even though their primary role
    # is not Tank/Support. This is a narrow whitelist — only heroes that genuinely
    # work as roamers in real MLBB play.
    # Note: Cyclops, Nana, Lunox, Harley, Yve, Alice, Lylia are EXPLICITLY excluded
    # from the Roam pool even though some players run them as roam off-meta.
    _ROAM_FLEX: Set[str] = set()  # Empty by default; populate if specific flexes needed

    # Archetypes classified as "magic damage" sources.
    # Note: "Utility - Summoner" is EXCLUDED because it contains non-mage heroes
    # (Sun, Popol and Kupa) — including it would corrupt team magic damage ratios.
    _MAGIC_ARCHETYPES = {
        "Mage - Burst Mage", "Mage - Battlemage", "Mage - Control Mage",
        "Mage - DPS Mage",
    }
    # Archetypes classified as "frontline / protector"
    _PROTECTOR_ARCHETYPES = {
        "Tank - Vanguard", "Tank - Stone Wall", "Tank - Glorious Setter",
    }
    # Archetypes classified as "glass cannon" (squishy high-damage)
    _GLASS_CANNON_ARCHETYPES = {
        "Marksman - Crit based", "Mage - Burst Mage", "Assassin - Prey Hunter",
    }

    def __init__(self, data: ScoringData):
        self.data = data

    def score_hero(
        self,
        hero_name: str,
        ally_picks: List[str],
        enemy_picks: List[str],
        needed_lane: Optional[str] = None,
        enemy_bans: Optional[List[str]] = None,
        user_intent: Optional[str] = None,
        locked_ally_lanes: Optional[set] = None,
        locked_enemy_lanes: Optional[set] = None,
    ) -> float:
        """Score a hero for the current draft state.

        Returns RAW, UNCLAMPED score. Clamping happens at the API boundary only.

        Args:
            hero_name: Hero to score
            ally_picks: Heroes already picked by our team
            enemy_picks: Heroes already picked by enemy team
            needed_lane: Lane that needs to be filled (e.g., "Jungle", "EXP")
            enemy_bans: Heroes banned by enemy team
            user_intent: Optional strategy preference (e.g., "enchanter", "initiator")
            locked_ally_lanes: Set of lane names already filled by ally picks
                (e.g., {"roam", "exp"}). When the user has explicitly told us
                which lane an ally is playing, we use that to decide if a
                lane is "taken" — otherwise we fall back to the heuristic
                of "any ally with this lane in their lane list counts."
            locked_enemy_lanes: Same, for the enemy team.

        Returns:
            Raw score (can exceed 1.0). Clamp at API boundary.
        """
        if enemy_bans and hero_name in enemy_bans:
            return 0.0

        # 1. Baseline
        score = self.BASELINE

        # 2. Win rate bonus
        stats = self.data.get_hero_stats(hero_name)
        win_rate = stats.get("win_rate", 0.5)
        score += (win_rate - 0.5) * 0.3

        # Gaps 1 & 2: Global ally state bonuses (computed per call)
        # The state vector captures strategic gaps the per-candidate loop cannot see:
        # - needs_magic: when team has zero magic damage
        # - ally_vulnerability: when team is heavy on squishies + no frontline
        ally_state = self.compute_ally_state(ally_picks)
        candidate_archetypes = set(self.data.get_hero_archetypes(hero_name))

        # Gap 1: Magic damage balance
        if ally_state["needs_magic"] and candidate_archetypes & self._MAGIC_ARCHETYPES:
            score += self.NEEDS_MAGIC_BONUS

        # Gap 2: Squishy ally protection — boost protector sub-roles
        if ally_state["ally_vulnerability"] == "HIGH":
            if candidate_archetypes & self._PROTECTOR_ARCHETYPES:
                score += self.ALLY_VULNERABILITY_BONUS

        # 3. Counter bonus (data + archetype rules)
        countered_enemies: Set[str] = set()
        for enemy in enemy_picks:
            conf, _ = self.data.get_counter_confidence(hero_name, enemy)
            if conf > 0:
                countered_enemies.add(enemy)

        # Data-driven counter contribution
        counter_bonus = 0.0
        for enemy in countered_enemies:
            conf, _ = self.data.get_counter_confidence(hero_name, enemy)
            counter_bonus += conf * self.COUNTER_PER_ENEMY
        counter_bonus = min(counter_bonus, self.COUNTER_BONUS_MAX)

        # Archetype-driven counter contribution
        # If candidate archetype is good_against an enemy archetype, add bonus
        # If candidate archetype is bad_against an enemy archetype, subtract
        archetype_counter = self._archetype_matchup_score(
            hero_name, enemy_picks, ally_picks, perspective="enemy"
        )
        # Archetype counter can push total above COUNTER_BONUS_MAX, but we cap total
        total_counter = min(counter_bonus + archetype_counter, self.COUNTER_BONUS_MAX)
        # But bad_against should not be clipped away by good_against gains
        # Apply bad_against penalty directly (no cap on penalty, but total capped)
        if archetype_counter < 0:
            total_counter = counter_bonus + archetype_counter  # allow it to go negative
        score += total_counter

        # 4. Synergy bonus
        synergy_with_allies: Set[str] = set()
        hero_synergies = set(self.data.get_synergies(hero_name))
        for ally in ally_picks:
            if ally in hero_synergies:
                synergy_with_allies.add(ally)
            ally_synergies = set(self.data.get_synergies(ally))
            if hero_name in ally_synergies:
                synergy_with_allies.add(ally)

        synergy_bonus = min(len(synergy_with_allies) * 0.05, self.SYNERGY_BONUS_MAX)
        score += synergy_bonus

        # 5. Combination bonus: counter enemy + synergy with ally
        if countered_enemies and synergy_with_allies:
            score += self.COMBO_BONUS

        # Gap 3: User intent strategy bump
        # If a user_intent is set and the candidate's archetype matches, apply +0.30.
        # This preserves counter sorting within the intent-narrowed candidate set.
        if user_intent:
            intent_archetypes = self._INTENT_ARCHETYPES.get(user_intent)
            if intent_archetypes and candidate_archetypes & intent_archetypes:
                score += self.USER_INTENT_BONUS

        # 6. Ban rate bonus: continuous linear, capped at +0.10
        ban_rate = stats.get("ban_rate", 0.0)
        ban_rate_bonus = min(ban_rate * 0.002, self.BAN_RATE_BONUS_MAX)  # ban_rate is in %
        score += ban_rate_bonus

        # 7. Lane check
        if needed_lane:
            hero_lanes_raw = self.data.get_hero_lanes(hero_name)
            hero_lanes = [l.replace(" Lane", "").upper() for l in hero_lanes_raw]
            needed_lane_upper = needed_lane.upper()

            # Decide if the needed lane is "already taken" by allies.
            #
            # Preferred path: the user has told us which lane each ally is
            # filling (locked_ally_lanes). We trust that signal directly.
            #
            # Fallback: any ally whose lane list contains this lane is
            # considered to be taking it. This is wrong for flex heroes
            # (Freya can be EXP or Jungle) when they're actually on EXP,
            # so it's only used when no explicit lane info is available.
            needed_lane_taken = False
            if locked_ally_lanes is not None:
                needed_lane_taken = needed_lane_upper in {
                    l.upper() for l in locked_ally_lanes
                }
            else:
                for ally in ally_picks:
                    ally_lanes = [
                        l.replace(" Lane", "").upper()
                        for l in self.data.get_hero_lanes(ally)
                    ]
                    if needed_lane_upper in ally_lanes:
                        needed_lane_taken = True
                        break

            if needed_lane_upper in hero_lanes and not needed_lane_taken:
                score += self.LANE_BONUS_FULFILL
            elif needed_lane_upper in hero_lanes and needed_lane_taken:
                score += self.LANE_PENALTY_TAKEN  # fills already-taken lane
            elif hero_lanes:
                score += self.LANE_PENALTY_TAKEN  # wrong lane (effectively fills wrong role)

        # 8. Team stability with archetype redundancy penalty
        stability = self._calculate_stability(hero_name, ally_picks, needed_lane)
        score += stability

        # 9. Dynamic trait counter: scan enemy composition for threats the
        # candidate can counter. Per-match +0.15, linear scaling, capped at
        # MAX_TRAIT_BONUS. Pure hash lookups via pre-computed trait sets.
        trait_bonus = self._scan_enemy_traits(hero_name, enemy_picks)
        score += min(trait_bonus, self.MAX_TRAIT_BONUS)

        return score  # RAW, unclamped

    def compute_ally_state(self, ally_picks: List[str]) -> dict:
        """Compute a global ally state vector from the locked ally team.

        Evaluates the team as a unified entity rather than per-hero, capturing
        strategic gaps that individual scoring cannot detect:
            - needs_magic: True if NO ally deals magic damage
            - ally_vulnerability: HIGH if multiple glass cannons + no frontline
                                 MEDIUM if some frontline but multiple squishies
                                 LOW otherwise
            - ally_archetypes: Set of all archetypes present in ally team

        This is computed ONCE before the scoring loop, not per candidate,
        so the cost is O(ally_picks * archetypes) total.
        """
        if not ally_picks:
            return {
                "needs_magic": True,
                "ally_vulnerability": "MEDIUM",
                "ally_archetypes": set(),
            }

        # Collect all archetypes present in ally team
        ally_archetypes: Set[str] = set()
        for ally in ally_picks:
            ally_archetypes.update(self.data.get_hero_archetypes(ally))

        # Gap 1: Damage type tracking
        has_magic = bool(ally_archetypes & self._MAGIC_ARCHETYPES)
        needs_magic = not has_magic

        # Gap 2: Squishiness index
        glass_cannon_count = len(ally_archetypes & self._GLASS_CANNON_ARCHETYPES)
        has_frontline = bool(ally_archetypes & self._PROTECTOR_ARCHETYPES)

        if glass_cannon_count >= 2 and not has_frontline:
            ally_vulnerability = "HIGH"
        elif glass_cannon_count >= 2 and has_frontline:
            ally_vulnerability = "MEDIUM"
        else:
            ally_vulnerability = "LOW"

        return {
            "needs_magic": needs_magic,
            "ally_vulnerability": ally_vulnerability,
            "ally_archetypes": ally_archetypes,
        }

    def _scan_enemy_traits(self, hero_name: str, enemy_picks: List[str]) -> float:
        """Per-enemy trait cross-reference.

        For each enemy hero, look at the enemy's traits. For each enemy trait
        that appears in _COUNTER_TRAIT_MAP as a key, check if the candidate
        has any of the listed counter-traits. Add TRAIT_BONUS_PER_MATCH per
        successful match. The caller caps at MAX_TRAIT_BONUS.

        Hot-path: uses pre-computed trait sets from get_hero_traits_set() —
        no set() allocations inside the loop.

        Args:
            hero_name: Candidate hero being scored.
            enemy_picks: Currently picked enemy heroes.

        Returns:
            Raw trait bonus (unclamped). 0.0 if no matchups.
        """
        candidate_traits = self.data.get_hero_traits_set(hero_name)
        if not candidate_traits or not enemy_picks:
            return 0.0
        bonus = 0.0
        for enemy in enemy_picks:
            enemy_traits = self.data.get_hero_traits_set(enemy)
            if not enemy_traits:
                continue
            for threat, counters in self._COUNTER_TRAIT_MAP.items():
                if threat in enemy_traits and candidate_traits & counters:
                    bonus += self.TRAIT_BONUS_PER_MATCH
        return bonus

    def _archetype_matchup_score(
        self,
        hero_name: str,
        enemy_picks: List[str],
        ally_picks: List[str],
        perspective: str,
    ) -> float:
        """Compute archetype matchup contribution.

        For perspective="enemy":
            - If hero archetype is good_against an enemy archetype, +0.05 per matching enemy
            - If hero archetype is bad_against an enemy archetype, -0.05 per matching enemy

        For perspective="ally" (synergy proxy):
            - Currently not used in scoring but available for future extension
        """
        if perspective != "enemy":
            return 0.0

        hero_archetypes = self.data.get_hero_archetypes(hero_name)
        if not hero_archetypes:
            return 0.0

        matchup_total = 0.0
        for enemy in enemy_picks:
            enemy_archetypes = self.data.get_hero_archetypes(enemy)
            if not enemy_archetypes:
                continue
            for archetype in hero_archetypes:
                rules = self.data.get_archetype_matchups(archetype)
                # good_against: +0.05 per matching enemy archetype
                for ea in enemy_archetypes:
                    if ea in rules.get("good_against", []):
                        matchup_total += self.ARCHETYPE_MATCHUP_BONUS
                    if ea in rules.get("bad_against", []):
                        matchup_total += self.ARCHETYPE_MATCHUP_PENALTY

        return matchup_total

    def _calculate_stability(
        self,
        hero_name: str,
        ally_picks: List[str],
        needed_lane: Optional[str],
    ) -> float:
        """Calculate team stability bonus/penalty.

        Rules:
            - Shared archetype with ally: -0.08 per overlap (max -0.20)
            - Introduces needed archetype (not in any ally): +0.04 (max +0.10)
            - Wrong lane: included in main score already, not duplicated here
        """
        if not ally_picks:
            return 0.0

        candidate_archetypes = set(self.data.get_hero_archetypes(hero_name))
        if not candidate_archetypes:
            return 0.0

        # Collect all archetypes present in ally team
        ally_archetypes: Set[str] = set()
        for ally in ally_picks:
            ally_archetypes.update(self.data.get_hero_archetypes(ally))

        # Redundancy penalty: shared archetype with existing ally
        shared = candidate_archetypes & ally_archetypes
        redundancy_penalty = len(shared) * self.ARCHETYPE_REDUNDANCY_PENALTY
        redundancy_penalty = max(redundancy_penalty, self.STABILITY_PENALTY_MAX)

        # Diversity bonus: introduces archetype no ally has
        unique_to_candidate = candidate_archetypes - ally_archetypes
        if unique_to_candidate:
            diversity_bonus = self.ARCHETYPE_DIVERSITY_BONUS
        else:
            diversity_bonus = 0.0

        total = diversity_bonus + redundancy_penalty  # penalty is negative
        return max(self.STABILITY_PENALTY_MAX, min(self.STABILITY_BONUS_MAX, total))

    def _hero_can_play_lane(self, hero_name: str, needed_lane: str) -> bool:
        """Hard lane filter: returns True if the hero can play the requested lane.

        Rules:
            - Roam: primary role is Tank, primary role is Support, OR hero in _ROAM_FLEX
            - Mid:  primary role is Mage
            - Gold: primary role is Marksman
            - EXP / Jungle: requested lane must be in the hero's lane list

        The previous soft -0.50 lane penalty was insufficient because counter bonuses
        could overcome it, causing Mages to pollute Roam recommendations. This hard
        filter guarantees lane integrity.

        Args:
            hero_name: Hero to check
            needed_lane: Lane the team needs filled (e.g., "Roam", "Mid", "Gold", "EXP", "Jungle")

        Returns:
            True if the hero can legitimately play that lane
        """
        if not needed_lane:
            return True

        # Normalize: accept both "Mid" and "Mid Lane" forms
        lane_upper = needed_lane.replace(" Lane", "").replace(" lane", "").upper()
        roles = self.data.get_hero_roles(hero_name)
        primary_role = roles[0] if roles else ""

        if lane_upper == "ROAM":
            return (
                primary_role in ("Tank", "Support")
                or hero_name in self._ROAM_FLEX
            )
        if lane_upper == "MID":
            return primary_role == "Mage"
        if lane_upper == "GOLD":
            return primary_role == "Marksman"
        # EXP / Jungle: hero must list the lane in their lane list
        hero_lanes = [
            l.replace(" Lane", "").upper()
            for l in self.data.get_hero_lanes(hero_name)
        ]
        return lane_upper in hero_lanes

    def generate_recommendations(
        self,
        available_heroes: List[str],
        ally_picks: List[str],
        enemy_picks: List[str],
        needed_lane: Optional[str] = None,
        enemy_bans: Optional[List[str]] = None,
        top_k: int = 5,
        user_intent: Optional[str] = None,
        locked_ally_lanes: Optional[set] = None,
        locked_enemy_lanes: Optional[set] = None,
    ) -> List[Tuple[str, float]]:
        """Generate ranked hero recommendations with hard lane filter.

        Pipeline:
            1. Hard lane filter: exclude heroes that cannot play needed_lane
            2. Optional user_intent narrowing: filter to heroes matching the intent
            3. Safe fallback: if intent+lane yields < 3 candidates, drop intent
            4. Score remaining candidates with raw (unclamped) values
            5. Sort by raw score descending
            6. Apply Sub-Role Diversity Decay during top-K extraction
            7. Clamp final top-K to [0.0, 1.0] for the API contract

        Args:
            available_heroes: Pool of heroes to consider (typically all heroes)
            ally_picks: Heroes already picked by our team
            enemy_picks: Heroes already picked by enemy team
            needed_lane: Lane that needs to be filled
            enemy_bans: Heroes banned by enemy team
            top_k: Number of recommendations to return
            user_intent: Optional strategy preference (e.g., "enchanter", "dps_mage")

        Returns:
            List of (hero_name, score) tuples sorted by score descending, clamped to [0.0, 1.0]
        """
        # Step 1: Hard lane filter
        lane_candidates = [
            h for h in available_heroes
            if self._hero_can_play_lane(h, needed_lane or "")
        ]

        if not lane_candidates:
            return []

        # Step 2 & 3: Optional user_intent narrowing with safe fallback
        candidates = lane_candidates
        if user_intent:
            intent_archetypes = self._INTENT_ARCHETYPES.get(user_intent)
            if intent_archetypes:
                intent_candidates = [
                    h for h in lane_candidates
                    if set(self.data.get_hero_archetypes(h)) & intent_archetypes
                ]
                # Safe fallback: only narrow if we still have enough candidates
                if len(intent_candidates) >= 3:
                    candidates = intent_candidates
                # else: keep lane_candidates (drop intent, keep lane boundary)

        # Step 4: Score all candidates (raw)
        scores: List[Tuple[str, float]] = []
        for hero in candidates:
            if enemy_bans and hero in enemy_bans:
                continue
            if hero in ally_picks or hero in enemy_picks:
                continue
            raw = self.score_hero(
                hero, ally_picks, enemy_picks,
                needed_lane, enemy_bans, user_intent,
                locked_ally_lanes=locked_ally_lanes,
                locked_enemy_lanes=locked_enemy_lanes,
            )
            scores.append((hero, raw))

        # Step 5: Sort by raw score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Step 6: Apply diversity decay during top-K extraction
        top_picks: List[Tuple[str, float]] = []
        archetype_counts: Dict[str, int] = {}

        for hero, raw_score in scores:
            if len(top_picks) >= top_k:
                break

            # Compute decay based on how many of this hero's archetypes are already in top_picks
            hero_archetypes = self.data.get_hero_archetypes(hero)
            decay = 0.0
            for archetype in hero_archetypes:
                count = archetype_counts.get(archetype, 0)
                if count > 0:
                    decay += self.DIVERSITY_DECAY * count

            adjusted = raw_score + decay
            top_picks.append((hero, adjusted))

            # Update archetype counts
            for archetype in hero_archetypes:
                archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1

        # Re-sort top picks by adjusted score (decay may have changed order)
        top_picks.sort(key=lambda x: x[1], reverse=True)

        # Step 7: Clamp final top-K to [0.0, 1.0] for API contract
        result = [(name, max(0.0, min(1.0, score))) for name, score in top_picks[:top_k]]
        return result

    def rank_heroes(
        self,
        available_heroes: List[str],
        ally_picks: List[str],
        enemy_picks: List[str],
        needed_lane: Optional[str] = None,
        enemy_bans: Optional[List[str]] = None,
        top_k: int = 5,
        user_intent: Optional[str] = None,
        locked_ally_lanes: Optional[set] = None,
        locked_enemy_lanes: Optional[set] = None,
    ) -> List[Tuple[str, float]]:
        """Rank available heroes for the current draft.

        Thin wrapper around generate_recommendations() that applies the hard
        lane filter and full recommendation pipeline. Kept for backward
        compatibility with code that previously called rank_heroes directly.

        Returns:
            List of (hero_name, score) tuples sorted by score descending.
        """
        return self.generate_recommendations(
            available_heroes=available_heroes,
            ally_picks=ally_picks,
            enemy_picks=enemy_picks,
            needed_lane=needed_lane,
            enemy_bans=enemy_bans,
            top_k=top_k,
            user_intent=user_intent,
            locked_ally_lanes=locked_ally_lanes,
            locked_enemy_lanes=locked_enemy_lanes,
        )
