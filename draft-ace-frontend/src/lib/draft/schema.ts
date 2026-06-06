import { z } from "zod";

const laneSchema = z.enum(["gold", "mid", "exp", "jungle", "roam"]);

const pickSlotSchema = z.object({
  lane: laneSchema,
  heroName: z.string().nullable(),
});

const banSchema = z.object({
  index: z.number().int(),
  heroName: z.string().nullable(),
  team: z.enum(["blue", "red"]),
});

const recSchema = z.object({
  heroId: z.string(),
  heroName: z.string(),
  // Accept composite_score; gracefully fall back from legacy win_rate if needed.
  composite_score: z.number(),
  breakdown: z.object({
    synergy: z.number(),
    counter: z.number(),
    laneMatch: z.number(),
  }),
  fallback: z.boolean().optional(),
  counters: z.array(z.string()).default([]),
  countered_by: z.array(z.string()).default([]),
  synergies: z.array(z.string()).default([]),
});

const laneRecsSchema = z.object({
  heroes: z.array(recSchema),
  fallbackActive: z.boolean().default(false),
});

export const gameStateSchema = z.object({
  blue: z.array(pickSlotSchema).length(5),
  red: z.array(pickSlotSchema).length(5),
  bans: z.array(banSchema),
  recommendations: z.object({
    gold: laneRecsSchema,
    mid: laneRecsSchema,
    exp: laneRecsSchema,
    jungle: laneRecsSchema,
    roam: laneRecsSchema,
  }),
  phase: z.string().default("Draft"),
  intents: z.object({
    gold: z.string(),
    mid: z.string(),
    exp: z.string(),
    jungle: z.string(),
    roam: z.string(),
  }),
});

export const intentDispatchSchema = z.object({
  lane: laneSchema,
  intent: z.string().min(1).max(64),
  gameStateId: z.string().optional(),
});
