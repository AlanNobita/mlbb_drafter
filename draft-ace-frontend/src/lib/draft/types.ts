export type Lane = "gold" | "mid" | "exp" | "jungle" | "roam";
export type Team = "blue" | "red";
export type ConnectionStatus = "connected" | "reconnecting" | "disconnected" | "idle";
export type DataSourceMode = "live" | "synthetic" | "manual";

/**
 * A pick slot is fixed to a single lane. The slot IS the lane assignment —
 * no separate lane picker is needed. MLBB has a fixed lane system, so the
 * user picks the hero for that lane and the scoring engine gets the lane
 * context for free.
 */
export interface PickSlot {
  lane: Lane;
  heroName: string | null;
}

export interface BanSlot {
  index: number;
  heroName: string | null;
  team: Team;
}

export interface ScoreBreakdown {
  synergy: number;
  counter: number;
  laneMatch: number;
}

export interface Recommendation {
  heroId: string;
  heroName: string;
  composite_score: number;
  breakdown: ScoreBreakdown;
  fallback?: boolean;
  /** Enemy heroes (currently picked) that this hero counters. */
  counters: string[];
  /** Enemy heroes (currently picked) that counter this hero. */
  countered_by: string[];
  /** Ally heroes (currently picked) that this hero synergizes with. */
  synergies: string[];
}

export interface LaneRecommendations {
  heroes: Recommendation[];
  fallbackActive: boolean;
}

export type Recommendations = Record<Lane, LaneRecommendations>;

export interface GameStatePayload {
  blue: PickSlot[];
  red: PickSlot[];
  bans: BanSlot[];
  recommendations: Recommendations;
  phase: string;
  intents: Record<Lane, string>;
}
