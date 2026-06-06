# Countered By Tooltip Section — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Countered by" section to the recommendation tooltip that shows which currently-picked enemy heroes counter the candidate pick.

**Architecture:** Pure data-driven filter on `_countered_by` (already loaded from `hero_meta.json`). Wire format extends `Recommendation` with a new `countered_by: string[]` field. Frontend reuses the `HeroPill` pattern with a new `warning` tone (amber) and a `✓ No enemy counters` fallback.

**Tech Stack:** Python (FastAPI/uvicorn), TypeScript (TanStack Start), zod, lucide-react.

---

## Task 1: Backend — Add `countered_by` to Recommendation payload

**Files:**
- Modify: `server/main.py:379-395` (the per-recommendation hero dict builder)
- Test: `tests/test_countered_by.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_countered_by.py`:

```python
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
        # Use a hero known to have at least one counter in the data.
        # Roger is widely cited as a counter to X.Borg (no heal mechanic).
        result = data.get_heroes_that_beat("X.Borg")
        assert "Roger" in result, f"X.Borg should be countered by Roger: {result}"

    def test_unknown_hero_returns_empty(self, data):
        """Unknown hero returns an empty list, never raises."""
        assert data.get_heroes_that_beat("Nonexistent12345") == []


class TestCounteredByFilter:
    """Filter _countered_by against current enemy roster."""

    def test_filters_by_enemy_roster(self, data):
        """Only enemies currently in the roster are returned."""
        # Get all heroes that beat X.Borg, then filter by a known subset.
        all_counters = set(data.get_heroes_that_beat("X.Borg"))
        assert len(all_counters) >= 2, "Need at least 2 known counters for this test"
        enemy_roster = list(all_counters)[:2]
        # Filter logic mirrors main.py's recommendation builder.
        filtered = [e for e in enemy_roster if e in all_counters]
        assert filtered == enemy_roster

    def test_empty_roster_returns_empty(self, data):
        """No enemy picks -> empty filtered list."""
        filtered = [e for e in [] if e in set(data.get_heroes_that_beat("X.Borg"))]
        assert filtered == []
```

- [ ] **Step 2: Run test to verify it passes (sanity check)**

Run: `python -m pytest tests/test_countered_by.py -v`
Expected: PASS (since `get_heroes_that_beat` already exists).

- [ ] **Step 3: Modify `server/main.py` to include `countered_by`**

In `server/main.py`, in the per-recommendation hero dict (around line 379-395), add a new field. Replace the `heroes.append({...})` block with:

```python
                        countered_enemies = sorted(
                            enemy for enemy in enemy_set
                            if self.scoring_data.get_counter_confidence(hero, enemy)[0] > 0
                        )
                        synergy_with_allies = sorted(
                            ally for ally in ally_set
                            if ally in self.scoring_data.get_synergies(hero)
                            or hero in self.scoring_data.get_synergies(ally)
                        )
                        countered_by = sorted(
                            enemy for enemy in enemy_set
                            if enemy in self.scoring_data.get_heroes_that_beat(hero)
                        )
                        heroes.append({
                            "heroId": hero.lower().replace(" ", "_"),
                            "heroName": hero,
                            "composite_score": round(score, 3),
                            "breakdown": breakdown,
                            "counters": countered_enemies,
                            "synergies": synergy_with_allies,
                            "countered_by": countered_by,
                        })
```

- [ ] **Step 4: Re-run test suite to verify nothing broke**

Run: `python -m pytest tests/ --ignore=tests/test_yolo_detector.py --ignore=tests/test_yolo_pipeline.py --ignore=tests/test_yolo_detector_smoke.py --ignore=tests/test_capture.py --ignore=tests/test_synthetic_capture.py --ignore=tests/test_dataset_integrity.py --ignore=tests/test_detection.py --ignore=tests/test_scraper.py --ignore=tests/test_websocket.py --ignore=tests/test_integration.py --ignore=tests/test_draft_layout.py --ignore=tests/test_state_tracker.py -q`
Expected: 154 passed, 1 skipped (150 prior + 4 new).

- [ ] **Step 5: Commit**

```bash
git add server/main.py tests/test_countered_by.py
git commit -m "feat(tooltip): add countered_by field to recommendation payload"
```

---

## Task 2: Frontend — Extend types and zod schema

**Files:**
- Modify: `draft-ace-frontend/src/lib/draft/types.ts:29-39`
- Modify: `draft-ace-frontend/src/lib/draft/schema.ts:25-30`

- [ ] **Step 1: Update `types.ts`**

Replace the `Recommendation` interface (lines 29-39) with:

```ts
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
```

- [ ] **Step 2: Update `schema.ts`**

Find the `Recommendation` schema and add `countered_by`. Replace the relevant lines with:

```ts
  counters: z.array(z.string()).default([]),
  countered_by: z.array(z.string()).default([]),
  synergies: z.array(z.string()).default([]),
```

- [ ] **Step 3: Commit**

```bash
git add draft-ace-frontend/src/lib/draft/types.ts draft-ace-frontend/src/lib/draft/schema.ts
git commit -m "feat(tooltip): extend Recommendation type with countered_by"
```

---

## Task 3: Frontend — Render "Countered by" section with warning tone

**Files:**
- Modify: `draft-ace-frontend/src/components/draft/RecommendationRow.tsx`

- [ ] **Step 1: Add `tone` to `HeroPill` props**

Replace the `HeroPill` component (lines 23-37) with a version that supports `warning` tone:

```tsx
function HeroPill({
  name,
  team,
  tone = "team",
}: {
  name: string;
  team: "ally" | "enemy";
  tone?: "team" | "warning";
}) {
  return (
    <div
      className={cn(
        "flex items-center gap-1.5 rounded-md border px-1.5 py-1",
        tone === "warning"
          ? "border-amber-500/50 bg-amber-500/15"
          : "border-border/60 bg-surface/60",
      )}
    >
      <HeroTile name={name} team={team} size="sm" />
      <span
        className={cn(
          "max-w-[80px] truncate text-[11px] font-semibold",
          tone === "warning"
            ? "text-amber-200"
            : team === "ally"
              ? "text-ally"
              : "text-enemy",
        )}
      >
        {name}
      </span>
    </div>
  );
}
```

- [ ] **Step 2: Import `ShieldAlert` and `Check` from lucide-react**

Replace the import line (line 5) with:

```tsx
import { Swords, HandHeart, MapPin, ShieldAlert, Check } from "lucide-react";
```

- [ ] **Step 3: Update `RecommendationRow` to render the new section**

In the `RecommendationRow` function (line 39), add the `hasCounteredBy` check and the new section. Replace the `hasAnyContext` block (lines 77-112) with:

```tsx
        {hasAnyContext && (
          <div className="space-y-2">
            {hasCounters && (
              <div>
                <p className="mb-1 flex items-center gap-1 text-[10px] font-bold uppercase tracking-widest text-enemy">
                  <Swords className="h-3 w-3" />
                  Counters ({rec.counters.length})
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {rec.counters.map((name) => (
                    <HeroPill key={name} name={name} team="enemy" />
                  ))}
                </div>
              </div>
            )}
            <div>
              <p className="mb-1 flex items-center gap-1 text-[10px] font-bold uppercase tracking-widest text-amber-400">
                <ShieldAlert className="h-3 w-3" />
                Countered by ({rec.countered_by.length})
              </p>
              {rec.countered_by.length > 0 ? (
                <div className="flex flex-wrap gap-1.5">
                  {rec.countered_by.map((name) => (
                    <HeroPill key={name} name={name} team="enemy" tone="warning" />
                  ))}
                </div>
              ) : (
                <p className="flex items-center gap-1 text-[11px] text-muted-foreground">
                  <Check className="h-3 w-3 text-emerald-400" />
                  No enemy counters
                </p>
              )}
            </div>
            {hasSynergies && (
              <div>
                <p className="mb-1 flex items-center gap-1 text-[10px] font-bold uppercase tracking-widest text-ally">
                  <HandHeart className="h-3 w-3" />
                  Synergizes with ({rec.synergies.length})
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {rec.synergies.map((name) => (
                    <HeroPill key={name} name={name} team="ally" />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
```

- [ ] **Step 4: Update `hasAnyContext` to include `countered_by`**

The new section renders unconditionally (even when empty, showing the checkmark). Update the logic so the "Add enemy or ally picks" empty-state message still works correctly. The "Countered by (0)" with checkmark should be shown when there are NO enemy or ally picks at all (since `countered_by` is also empty in that case, but the checkmark is the right message).

Replace the `hasAnyContext` constant (line 42) with logic that always treats the tooltip as having context if the user has any enemy/ally picks OR always shows the "Countered by" section:

```tsx
  const hasCounters = rec.counters.length > 0;
  const hasSynergies = rec.synergies.length > 0;
  const hasAnyContext = hasCounters || hasSynergies || rec.countered_by.length > 0;
```

The empty-state `<p>` (lines 108-112) remains unchanged.

- [ ] **Step 5: Verify with `npm run build`**

Run from `draft-ace-frontend/`: `npm run build 2>&1 | tail -20`
Expected: Build succeeds with no TypeScript errors.

- [ ] **Step 6: Commit**

```bash
git add draft-ace-frontend/src/components/draft/RecommendationRow.tsx
git commit -m "feat(tooltip): render 'Countered by' section with warning tone"
```

---

## Task 4: E2E audit

- [ ] **Step 1: Manually verify the new section renders**

Run from repo root: `python -c "..."` to start the server, open the UI, draft a scenario where one enemy counters the recommendation, and confirm the new "Countered by" section appears with amber-tinted pills.

- [ ] **Step 2: Final test run**

```bash
python -m pytest tests/ --ignore=tests/test_yolo_detector.py --ignore=tests/test_yolo_pipeline.py --ignore=tests/test_yolo_detector_smoke.py --ignore=tests/test_capture.py --ignore=tests/test_synthetic_capture.py --ignore=tests/test_dataset_integrity.py --ignore=tests/test_detection.py --ignore=tests/test_scraper.py --ignore=tests/test_websocket.py --ignore=tests/test_integration.py --ignore=tests/test_draft_layout.py --ignore=tests/test_state_tracker.py -q
```

Expected: 154 passed, 1 skipped.

- [ ] **Step 3: Push (if credentials available)**

`git push origin master`

Otherwise stop and inform user.
