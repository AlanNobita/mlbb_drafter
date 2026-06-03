#!/usr/bin/env python3
"""
Fetch ranked match statistics from mlbb.io API (via Parse).

Provides:
- Hero win/pick/ban rates by rank (All, Mythic, Legend, Epic)
- Hero tier lists
- Hero counters, synergies, weak against
- Hero overview with statistics across ranks

Requires: Parse API key (free at parse.bot)
Usage:
    python training/fetch_mlbb_io.py --api-key YOUR_PARSE_API_KEY
    PARSE_API_KEY=xxx python training/fetch_mlbb_io.py
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("training/data")
API_DIR = DATA_DIR / "api_data"
API_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.parse.bot/scraper/574cbcf8-811b-4ed8-8128-c5e9a39efcc8"

RANKS = {1: "All", 2: "Mythic", 3: "Legend", 4: "Epic"}
TIMEFRAMES = {1: "1day", 2: "3days", 3: "7days"}


def fetch_parse(endpoint, params=None, api_key=None):
    """Fetch from Parse API."""
    url = f"{BASE_URL}/{endpoint}"
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url += f"?{query}"

    headers = {
        "User-Agent": "MLBBDrafter/1.0",
        "X-API-Key": api_key,
    }

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def fetch_hero_statistics(api_key):
    """Fetch win/pick/ban rates for all heroes across all ranks."""
    print("\n[1/5] Fetching hero statistics by rank...")
    all_stats = {}

    for rank_id, rank_name in RANKS.items():
        print(f"  Rank: {rank_name}...")
        for tf_id, tf_name in TIMEFRAMES.items():
            try:
                data = fetch_parse("get_hero_statistics", {
                    "rank_id": rank_id,
                    "timeframe_id": tf_id,
                }, api_key)

                heroes = data.get("data", {}).get("heroes", [])
                for hero in heroes:
                    name = hero.get("hero_name", "")
                    if name not in all_stats:
                        all_stats[name] = {"hero_id": hero.get("hero_id"), "name": name}

                    key = f"rank_{rank_name.lower()}_tf_{tf_name}"
                    all_stats[name][key] = {
                        "win_rate": hero.get("win_rate", 0),
                        "pick_rate": hero.get("pick_rate", 0),
                        "ban_rate": hero.get("ban_rate", 0),
                    }

                time.sleep(0.3)
            except Exception as e:
                print(f"    Error {rank_name}/{tf_name}: {e}")

    print(f"  Got stats for {len(all_stats)} heroes")
    return all_stats


def fetch_hero_tier_list(api_key):
    """Fetch hero tier list."""
    print("\n[2/5] Fetching hero tier list...")
    try:
        data = fetch_parse("get_hero_tier_list", {}, api_key)
        heroes = data.get("data", {}).get("heroes", [])
        last_updated = data.get("data", {}).get("lastUpdated", "")

        tier_data = {}
        for h in heroes:
            name = h.get("hero_name", "")
            tier_data[name] = {
                "hero_id": h.get("hero_id"),
                "tier": h.get("tier", ""),
                "score": h.get("score", 0),
                "role": h.get("role", []),
                "lane": h.get("lane", []),
                "speciality": h.get("speciality", []),
            }

        print(f"  Got tier data for {len(tier_data)} heroes (updated: {last_updated})")
        return tier_data
    except Exception as e:
        print(f"  Error: {e}")
        return {}


def fetch_hero_overviews(api_key, hero_names):
    """Fetch detailed overview for each hero (counters, synergies, stats)."""
    print(f"\n[3/5] Fetching hero overviews for {len(hero_names)} heroes...")
    overviews = {}
    fetched = 0

    for name in hero_names:
        try:
            data = fetch_parse("get_hero_overview", {"hero_name": name}, api_key)
            hero_data = data.get("data", {})

            overviews[name] = {
                "counters": hero_data.get("counters", []),
                "weak_against": hero_data.get("weakAgainst", []),
                "synergies": hero_data.get("synergies", []),
                "statistics": hero_data.get("statistics", []),
                "tier": hero_data.get("tier", ""),
                "score": hero_data.get("score", 0),
            }

            fetched += 1
            if fetched % 20 == 0:
                print(f"  Fetched {fetched}/{len(hero_names)}...")
            time.sleep(0.3)

        except Exception as e:
            if "429" in str(e):
                print(f"  Rate limited at {fetched}, waiting 60s...")
                time.sleep(60)
            else:
                print(f"  Error fetching {name}: {e}")

    print(f"  Got overviews for {len(overviews)} heroes")
    return overviews


def save_data(name, data):
    """Save data to JSON file."""
    filepath = API_DIR / f"{name}.json"

    if filepath.exists():
        with open(filepath) as f:
            existing = json.load(f)
        if isinstance(existing, dict) and isinstance(data, dict):
            existing.update(data)
            data = existing

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    size = filepath.stat().st_size / 1024
    print(f"  Saved {name}.json ({size:.1f}KB)")


def main():
    parser = argparse.ArgumentParser(description="Fetch MLBB ranked stats from mlbb.io API")
    parser.add_argument("--api-key", type=str, default=os.environ.get("PARSE_API_KEY"),
                        help="Parse API key (or set PARSE_API_KEY env var)")
    parser.add_argument("--skip-overviews", action="store_true",
                        help="Skip hero overviews (saves credits)")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: Parse API key required.")
        print("  Get free key at: https://parse.bot")
        print("  Then run: python training/fetch_mlbb_io.py --api-key YOUR_KEY")
        print("  Or set: export PARSE_API_KEY=YOUR_KEY")
        return

    print("=" * 60)
    print("MLBB.IO RANKED STATS FETCHER")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Fetch all data
    stats = fetch_hero_statistics(args.api_key)
    tiers = fetch_hero_tier_list(args.api_key)

    if not args.skip_overviews:
        hero_names = list(stats.keys())
        overviews = fetch_hero_overviews(args.api_key, hero_names)
    else:
        overviews = {}

    # Save data
    print("\n[4/5] Saving data...")
    save_data("mlbb_io_stats", stats)
    save_data("mlbb_io_tiers", tiers)
    save_data("mlbb_io_overviews", overviews)

    # Create combined ranked stats file
    print("\n[5/5] Creating combined ranked stats...")
    combined = {}
    for name, stat_data in stats.items():
        combined[name] = {
            "name": name,
            "hero_id": stat_data.get("hero_id"),
        }
        # Add tier info
        if name in tiers:
            combined[name]["tier"] = tiers[name]["tier"]
            combined[name]["score"] = tiers[name]["score"]
            combined[name]["role"] = tiers[name]["role"]
            combined[name]["lane"] = tiers[name]["lane"]

        # Add current stats (Mythic, 7 days)
        mythic_7d = stat_data.get("rank_mythic_tf_7days", {})
        if mythic_7d:
            combined[name]["mythic_win_rate"] = mythic_7d.get("win_rate", 0)
            combined[name]["mythic_pick_rate"] = mythic_7d.get("pick_rate", 0)
            combined[name]["mythic_ban_rate"] = mythic_7d.get("ban_rate", 0)

        # Add overview data
        if name in overviews:
            ov = overviews[name]
            combined[name]["counters"] = ov.get("counters", [])
            combined[name]["weak_against"] = ov.get("weak_against", [])
            combined[name]["synergies"] = ov.get("synergies", [])

    save_data("mlbb_io_ranked_combined", combined)

    # Summary
    print("\n" + "=" * 60)
    print("FETCH COMPLETE")
    print("=" * 60)
    print(f"  Hero statistics: {len(stats)} heroes × {len(RANKS)} ranks × {len(TIMEFRAMES)} timeframes")
    print(f"  Tier list: {len(tiers)} heroes")
    print(f"  Hero overviews: {len(overviews)} heroes")
    print(f"\nData saved to: {API_DIR}")


if __name__ == "__main__":
    main()
