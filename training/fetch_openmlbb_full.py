#!/usr/bin/env python3
"""
Comprehensive OpenMLBB API fetcher.

Fetches ALL available data from mlbb.rone.dev:
- Hero list (132 heroes)
- Hero rank statistics (win/pick/ban rates)
- Hero counters (who beats whom)
- Hero relations (synergies, counters)
- Hero compatibility (team compositions)
- Hero skill combos
- Hero trends (performance over time)
- Hero positions (lane distribution)
- Academy data (heroes, roles, equipment, spells, emblems)

Usage:
    python training/fetch_openmlbb_full.py
"""

import json
import time
import urllib.request
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("training/data")
API_DIR = DATA_DIR / "api_data"
API_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://mlbb.rone.dev/api"


def fetch_json(url, timeout=30):
    """Fetch JSON from URL."""
    req = urllib.request.Request(url, headers={"User-Agent": "MLBBDrafter/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_paginated(url_template, page_size=20):
    """Fetch all pages of a paginated endpoint."""
    all_records = []
    page = 1
    while True:
        url = url_template.format(size=page_size, index=page)
        data = fetch_json(url)
        records = data.get("data", {}).get("records", [])
        if not records:
            break
        all_records.extend(records)
        page += 1
        time.sleep(0.5)
    return all_records


def fetch_hero_list():
    """Fetch complete hero list."""
    print("\n[1/10] Fetching hero list...")
    records = fetch_paginated(f"{BASE_URL}/heroes?size={{size}}&index={{index}}&order=desc&lang=en")
    heroes = {}
    for rec in records:
        d = rec.get("data", {})
        hero_id = d.get("hero_id")
        name = d.get("hero", {}).get("data", {}).get("name", "")
        if hero_id and name:
            heroes[hero_id] = {
                "hero_id": hero_id,
                "name": name,
                "icon": d.get("hero", {}).get("data", {}).get("head", ""),
            }
    print(f"  Got {len(heroes)} heroes")
    return heroes


def fetch_hero_rank_stats():
    """Fetch hero rank statistics (win/pick/ban rates)."""
    print("\n[2/10] Fetching hero rank statistics...")
    records = fetch_paginated(f"{BASE_URL}/heroes/rank?lang=en&size={{size}}&index={{index}}&order=desc")
    stats = {}
    for rec in records:
        d = rec.get("data", {})
        hero_id = d.get("main_heroid")
        name = d.get("main_hero", {}).get("data", {}).get("name", "")
        if hero_id and name:
            stats[hero_id] = {
                "name": name,
                "win_rate": d.get("main_hero_win_rate", 0),
                "pick_rate": d.get("main_hero_appearance_rate", 0),
                "ban_rate": d.get("main_hero_ban_rate", 0),
                "synergies": [
                    {
                        "hero_id": s.get("heroid"),
                        "increase_win_rate": s.get("increase_win_rate", 0),
                    }
                    for s in d.get("sub_hero", []) if s.get("heroid")
                ],
            }
    print(f"  Got stats for {len(stats)} heroes")
    return stats


def fetch_hero_counters(hero_ids):
    """Fetch counter data for all heroes."""
    print(f"\n[3/10] Fetching counters for {len(hero_ids)} heroes...")
    counters = {}
    fetched = 0

    for hero_id in hero_ids:
        try:
            data = fetch_json(f"{BASE_URL}/heroes/{hero_id}/counters?lang=en")
            records = data.get("data", {}).get("records", [])

            if records:
                for rec in records:
                    rec_data = rec.get("data", {})
                    main_name = rec_data.get("main_hero", {}).get("data", {}).get("name", "")
                    counters_list = []
                    for sub in rec_data.get("sub_hero", []):
                        counters_list.append({
                            "hero_id": sub.get("heroid"),
                            "win_rate": sub.get("hero_win_rate", 0),
                            "appearance_rate": sub.get("hero_appearance_rate", 0),
                            "increase_win_rate": sub.get("increase_win_rate", 0),
                        })
                    if main_name:
                        counters[main_name] = counters_list

            fetched += 1
            if fetched % 20 == 0:
                print(f"  Fetched {fetched}/{len(hero_ids)}...")
            time.sleep(0.5)

        except Exception as e:
            print(f"  Error fetching counters for {hero_id}: {e}")

    print(f"  Got counters for {len(counters)} heroes")
    return counters


def fetch_hero_relations(hero_ids):
    """Fetch hero relations (synergies, counters) for all heroes."""
    print(f"\n[4/10] Fetching relations for {len(hero_ids)} heroes...")
    relations = {}
    fetched = 0

    for hero_id in hero_ids:
        try:
            data = fetch_json(f"{BASE_URL}/heroes/{hero_id}/relations?lang=en")
            records = data.get("data", {}).get("records", [])

            if records:
                for rec in records:
                    rec_data = rec.get("data", {})
                    main_name = rec_data.get("main_hero", {}).get("data", {}).get("name", "")
                    if main_name:
                        relations[main_name] = {
                            "best_with": [],
                            "strong_against": [],
                            "weak_against": [],
                        }
                        for sub in rec_data.get("sub_hero", []):
                            hero_name = sub.get("hero", {}).get("data", {}).get("name", "")
                            relation_type = sub.get("relation_type", "")
                            if hero_name and relation_type:
                                if relation_type == "best_with":
                                    relations[main_name]["best_with"].append(hero_name)
                                elif relation_type == "strong_against":
                                    relations[main_name]["strong_against"].append(hero_name)
                                elif relation_type == "weak_against":
                                    relations[main_name]["weak_against"].append(hero_name)

            fetched += 1
            if fetched % 20 == 0:
                print(f"  Fetched {fetched}/{len(hero_ids)}...")
            time.sleep(0.5)

        except Exception as e:
            print(f"  Error fetching relations for {hero_id}: {e}")

    print(f"  Got relations for {len(relations)} heroes")
    return relations


def fetch_hero_compatibility(hero_ids):
    """Fetch hero compatibility data."""
    print(f"\n[5/10] Fetching compatibility for {len(hero_ids)} heroes...")
    compatibility = {}
    fetched = 0

    for hero_id in hero_ids:
        try:
            data = fetch_json(f"{BASE_URL}/heroes/{hero_id}/compatibility?lang=en")
            records = data.get("data", {}).get("records", [])

            if records:
                for rec in records:
                    rec_data = rec.get("data", {})
                    main_name = rec_data.get("main_hero", {}).get("data", {}).get("name", "")
                    if main_name:
                        compatibility[main_name] = []
                        for sub in rec_data.get("sub_hero", []):
                            hero_name = sub.get("hero", {}).get("data", {}).get("name", "")
                            score = sub.get("compatibility_score", 0)
                            if hero_name:
                                compatibility[main_name].append({
                                    "name": hero_name,
                                    "score": score,
                                })

            fetched += 1
            if fetched % 20 == 0:
                print(f"  Fetched {fetched}/{len(hero_ids)}...")
            time.sleep(0.5)

        except Exception as e:
            print(f"  Error fetching compatibility for {hero_id}: {e}")

    print(f"  Got compatibility for {len(compatibility)} heroes")
    return compatibility


def fetch_hero_trends(hero_ids):
    """Fetch hero performance trends."""
    print(f"\n[6/10] Fetching trends for {len(hero_ids)} heroes...")
    trends = {}
    fetched = 0

    for hero_id in hero_ids:
        try:
            data = fetch_json(f"{BASE_URL}/heroes/{hero_id}/trends?lang=en")
            records = data.get("data", {}).get("records", [])

            if records:
                for rec in records:
                    rec_data = rec.get("data", {})
                    main_name = rec_data.get("main_hero", {}).get("data", {}).get("name", "")
                    if main_name:
                        trends[main_name] = {
                            "win_rate_trend": rec_data.get("win_rate_trend", []),
                            "pick_rate_trend": rec_data.get("pick_rate_trend", []),
                            "ban_rate_trend": rec_data.get("ban_rate_trend", []),
                        }

            fetched += 1
            if fetched % 20 == 0:
                print(f"  Fetched {fetched}/{len(hero_ids)}...")
            time.sleep(0.5)

        except Exception as e:
            print(f"  Error fetching trends for {hero_id}: {e}")

    print(f"  Got trends for {len(trends)} heroes")
    return trends


def fetch_hero_positions():
    """Fetch hero position/lane data."""
    print("\n[7/10] Fetching hero positions...")
    try:
        data = fetch_json(f"{BASE_URL}/heroes/positions?lang=en")
        records = data.get("data", {}).get("records", [])
        positions = {}
        for rec in records:
            d = rec.get("data", {})
            hero_id = d.get("hero_id")
            name = d.get("hero", {}).get("data", {}).get("name", "")
            if hero_id and name:
                positions[name] = {
                    "hero_id": hero_id,
                    "positions": d.get("positions", []),
                }
        print(f"  Got positions for {len(positions)} heroes")
        return positions
    except Exception as e:
        print(f"  Error: {e}")
        return {}


def fetch_hero_skill_combos(hero_ids):
    """Fetch hero skill combos."""
    print(f"\n[8/10] Fetching skill combos for {len(hero_ids)} heroes...")
    combos = {}
    fetched = 0

    for hero_id in hero_ids:
        try:
            data = fetch_json(f"{BASE_URL}/heroes/{hero_id}/skill-combos?lang=en")
            records = data.get("data", {}).get("records", [])

            if records:
                for rec in records:
                    rec_data = rec.get("data", {})
                    main_name = rec_data.get("main_hero", {}).get("data", {}).get("name", "")
                    if main_name:
                        combos[main_name] = []
                        for combo in rec_data.get("combos", []):
                            combos[main_name].append({
                                "sequence": combo.get("sequence", ""),
                                "description": combo.get("description", ""),
                            })

            fetched += 1
            if fetched % 20 == 0:
                print(f"  Fetched {fetched}/{len(hero_ids)}...")
            time.sleep(0.5)

        except Exception as e:
            print(f"  Error fetching combos for {hero_id}: {e}")

    print(f"  Got combos for {len(combos)} heroes")
    return combos


def fetch_academy_data():
    """Fetch academy data (heroes, roles, equipment, spells, emblems)."""
    print("\n[9/10] Fetching academy data...")
    academy = {}

    endpoints = [
        ("heroes", "academy/heroes"),
        ("roles", "academy/roles"),
        ("equipment", "academy/equipment"),
        ("spells", "academy/spells"),
        ("emblems", "academy/emblems"),
    ]

    for name, endpoint in endpoints:
        try:
            data = fetch_json(f"{BASE_URL}/{endpoint}?lang=en")
            academy[name] = data.get("data", {})
            time.sleep(0.5)
        except Exception as e:
            print(f"  Error fetching academy/{name}: {e}")

    print(f"  Got academy data: {list(academy.keys())}")
    return academy


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
    print("=" * 60)
    print("COMPREHENSIVE OPENMLBB FETCHER")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Fetch all data
    heroes = fetch_hero_list()
    hero_ids = list(heroes.keys())

    rank_stats = fetch_hero_rank_stats()
    counters = fetch_hero_counters(hero_ids)
    relations = fetch_hero_relations(hero_ids)
    compatibility = fetch_hero_compatibility(hero_ids)
    trends = fetch_hero_trends(hero_ids)
    positions = fetch_hero_positions()
    combos = fetch_hero_skill_combos(hero_ids)
    academy = fetch_academy_data()

    # Save data
    print("\n[10/10] Saving data...")
    save_data("openmlbb_heroes", heroes)
    save_data("openmlbb_rank_stats", rank_stats)
    save_data("openmlbb_counters", counters)
    save_data("openmlbb_relations", relations)
    save_data("openmlbb_compatibility", compatibility)
    save_data("openmlbb_trends", trends)
    save_data("openmlbb_positions", positions)
    save_data("openmlbb_combos", combos)

    for name, data in academy.items():
        save_data(f"openmlbb_academy_{name}", data)

    # Summary
    print("\n" + "=" * 60)
    print("FETCH COMPLETE")
    print("=" * 60)
    print(f"  Heroes: {len(heroes)}")
    print(f"  Rank stats: {len(rank_stats)}")
    print(f"  Counters: {len(counters)}")
    print(f"  Relations: {len(relations)}")
    print(f"  Compatibility: {len(compatibility)}")
    print(f"  Trends: {len(trends)}")
    print(f"  Positions: {len(positions)}")
    print(f"  Combos: {len(combos)}")
    print(f"\nData saved to: {API_DIR}")


if __name__ == "__main__":
    main()
