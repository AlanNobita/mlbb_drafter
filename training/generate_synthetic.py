#!/usr/bin/env python3
"""Generate synthetic MLBB draft training data."""

import json
import csv
import random
import os
from pathlib import Path

NUM_SAMPLES = 10000
FRIENDLY_PICKS = 5
ENEMY_PICKS = 5
NUM_BANS = 6

OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "synthetic_drafts.csv"
HERO_DATA = Path(__file__).parent.parent / "shared" / "hero_meta.json"


def load_heroes():
    with open(HERO_DATA) as f:
        data = json.load(f)
    heroes = data["heroes"]
    for h in heroes:
        h["syn_win_rate"] = random.uniform(0.48, 0.54)
        h["syn_pick_rate"] = random.uniform(1.0, 20.0)
    return heroes


def calc_role_diversity(picks, hero_map):
    roles = set(hero_map[hid]["role"] for hid in picks if hid in hero_map)
    return len(roles) / 6.0


def calc_pick_rate_factor(picks, hero_map):
    rates = [hero_map[hid].get("syn_pick_rate", 5.0) for hid in picks if hid in hero_map]
    avg = sum(rates) / len(rates) if rates else 5.0
    return (avg / 20.0) * 0.1


def generate_draft(heroes, hero_map, hero_ids):
    available = hero_ids[:]
    random.shuffle(available)

    friendly = available[:FRIENDLY_PICKS]
    remaining = available[FRIENDLY_PICKS:]
    enemy = remaining[:ENEMY_PICKS]
    remaining = remaining[ENEMY_PICKS:]
    bans = remaining[:NUM_BANS]

    diversity = calc_role_diversity(friendly, hero_map)
    pick_factor = calc_pick_rate_factor(friendly, hero_map)
    noise = random.uniform(-0.1, 0.1)

    win_rate = 0.5 + (diversity * 0.1) + pick_factor + noise
    win_rate = max(0.1, min(0.9, win_rate))

    return friendly, enemy, bans, round(win_rate, 4)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    heroes = load_heroes()
    hero_map = {h["id"]: h for h in heroes}
    hero_ids = [h["id"] for h in heroes]

    print(f"Loaded {len(heroes)} heroes")
    print(f"Generating {NUM_SAMPLES} synthetic drafts...")

    rows = []
    for i in range(NUM_SAMPLES):
        friendly, enemy, bans, win_rate = generate_draft(heroes, hero_map, hero_ids)
        rows.append({
            "match_id": i + 1,
            "friendly_picks": json.dumps(friendly),
            "enemy_picks": json.dumps(enemy),
            "bans": json.dumps(bans),
            "win_rate": win_rate,
        })

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["match_id", "friendly_picks", "enemy_picks", "bans", "win_rate"])
        writer.writeheader()
        writer.writerows(rows)

    win_rates = [r["win_rate"] for r in rows]
    print(f"\nGenerated {len(rows)} drafts")
    print(f"Output: {OUTPUT_FILE}")
    print(f"\nWin rate stats:")
    print(f"  Mean: {sum(win_rates)/len(win_rates):.4f}")
    print(f"  Min: {min(win_rates):.4f}")
    print(f"  Max: {max(win_rates):.4f}")

    roles = set(h["role"] for h in heroes)
    print(f"\nRoles: {', '.join(sorted(roles))}")
    print(f"Hero count: {len(heroes)}")


if __name__ == "__main__":
    main()
