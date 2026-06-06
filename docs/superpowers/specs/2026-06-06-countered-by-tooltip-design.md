# Countered By Tooltip Section — Design

Date: 2026-06-06
Status: Approved

## Purpose

Help users avoid picking a hero that gets countered by the current enemy
roster. Mirrors the existing "Counters" section but for the inverse
direction.

## Backend

**`server/recommendation/scoring_data.py`** — no change. Already exposes
`_countered_by[hero_name]` from `hero_meta.json` (heroes that beat this
hero, double-checked against `openmlbb.weak`).

**`server/main.py`** — in the per-recommendation dict (inside the
recommendation loop that already builds `counters` and `synergies`):

```python
countered_by = [
    e for e in enemy_picks
    if e in data._countered_by.get(hero_name, [])
]
```

This is a pure data-driven filter. No trait math (mirrors `counters`).

## Wire Format

Add to `Recommendation` payload:

```python
"countered_by": list[str]  # filtered by current enemy roster
```

## Frontend

**`draft-ace-frontend/src/lib/draft/types.ts`** — extend `Recommendation`:

```ts
countered_by: string[];
```

**`draft-ace-frontend/src/lib/draft/schema.ts`** — extend `recSchema`:

```ts
countered_by: z.array(z.string()).default([]),
```

**`draft-ace-frontend/src/components/draft/RecommendationRow.tsx`** —
add new section between "Counters" and "Synergizes with":

- Header: amber ShieldAlert icon + "Countered by (n)"
- n > 0: render amber-tinted `HeroPill` list (reuse `HeroPill`, add
  `tone="warning"` variant — amber background, white text)
- n == 0: render single line: "✓ No enemy counters" (gray text)

## Tests

New test in `tests/test_trait_scoring.py` (or new
`tests/test_countered_by.py`):

- `test_countered_by_filters_by_enemy_roster`: candidate whose
  `_countered_by` includes 2 of 5 enemy heroes → `countered_by` has
  those 2 only
- `test_countered_by_empty_when_no_match`: candidate with no enemy
  in `_countered_by` → `countered_by == []`
- `test_countered_by_empty_when_no_enemies`: empty enemy roster →
  `countered_by == []`
- `test_countered_by_uses_correct_direction`: data sanity — `X.Borg`
  has Roger in `_countered_by` (X.Borg is weak against Roger)

Tests call the per-recommendation dict builder directly (need to
extract or test the helper). If the builder is inline in `main.py`,
refactor to a private helper for testability.

## Commits

Single commit:

```
feat(tooltip): add 'Countered by' section with amber warning pills
```

## Out of Scope

- Trait-based countered-by (e.g., an enemy with `armor_shred` doesn't
  necessarily counter all `high_armor` heroes in the data sense)
- Cross-referencing with the dynamic trait system
- Any change to the existing Counters / Synergizes sections
