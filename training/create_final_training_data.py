#!/usr/bin/env python3
"""
Create final consolidated training file combining ALL data sources.
This is the single file needed for model training.
"""
import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "training" / "data"
API_DIR = DATA_DIR / "api_data"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    print("Creating final consolidated training data...")
    
    # Load all data sources
    hero_stats_temporal = {}
    with open(DATA_DIR / "hero_stats_temporal.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hero_stats_temporal[row["hero_name"]] = row
    
    hero_stats_comprehensive = {}
    with open(DATA_DIR / "hero_stats_comprehensive.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hero_stats_comprehensive[row["hero_name"]] = row
    
    tournament_drafts = load_json(DATA_DIR / "tournament_drafts.json")
    patch_timeline = load_json(DATA_DIR / "patch_timeline.json")
    era_stats = load_json(DATA_DIR / "era_hero_stats.json")
    synergy_matrix = load_json(DATA_DIR / "synergy_matrix_tournament.json")
    adjacency = load_json(DATA_DIR / "adjacency_real.json")
    hero_counters = load_json(API_DIR / "hero_counters_all.json")
    
    # Build final consolidated structure
    consolidated = {
        "metadata": {
            "version": "1.0.0",
            "generated": "2026-06-03",
            "description": "MLBB Drafter training data - all sources consolidated",
            "sources": [
                "drive_data: 13,171 tournament matches (M1-M5, MPL)",
                "openmlbb_api: Hero stats, counters, synergies",
                "p3hndrx: Hero metadata, items, emblems",
                "mlbb_winrate: Current win/ban/pick rates",
            ],
            "temporal_coverage": "2017-2023 (54 patches)",
            "hero_count": len(hero_stats_temporal),
            "match_count": len(tournament_drafts),
        },
        
        "heroes": {},
        
        "tournament_drafts": tournament_drafts,
        
        "patch_timeline": patch_timeline,
        
        "era_stats": era_stats,
        
        "synergy_matrix": synergy_matrix,
        
        "adjacency_matrix": adjacency,
    }
    
    # Merge hero data from all sources
    all_heroes = set()
    all_heroes.update(hero_stats_temporal.keys())
    all_heroes.update(hero_stats_comprehensive.keys())
    all_heroes.update(hero_counters.keys())
    
    for hero in sorted(all_heroes):
        temporal = hero_stats_temporal.get(hero, {})
        comprehensive = hero_stats_comprehensive.get(hero, {})
        counters = hero_counters.get(hero, {})
        
        # Extract counter names from API data
        counter_records = counters.get("data", {}).get("records", [])
        counter_names = []
        if counter_records:
            sub_heroes = counter_records[0].get("data", {}).get("sub_hero", [])
            # Get weak against heroes
            for sh in sub_heroes[:5]:
                name = sh.get("hero", {}).get("data", {}).get("name", "")
                if name:
                    counter_names.append(name)
        
        consolidated["heroes"][hero] = {
            # Basic info
            "role": temporal.get("role", comprehensive.get("role", "")),
            "lane": temporal.get("lane", comprehensive.get("lane", "")),
            
            # Temporal features (time-aware)
            "recency_score": float(temporal.get("recency_score", 0)),
            "meta_score": float(temporal.get("meta_score", 0)),
            "trend": float(temporal.get("trend", 0)),
            "total_tournament_games": int(temporal.get("total_tournament_games", 0)),
            "recent_tournament_games": int(temporal.get("recent_tournament_games", 0)),
            
            # Time-decayed stats
            "weighted_win_rate": float(temporal.get("weighted_win_rate", 0)),
            "weighted_pick_rate": float(temporal.get("weighted_pick_rate", 0)),
            "weighted_ban_rate": float(temporal.get("weighted_ban_rate", 0)),
            
            # Current stats (latest patch)
            "current_win_rate": float(temporal.get("current_win_rate", 0)),
            "current_pick_rate": float(temporal.get("current_pick_rate", 0)),
            "current_ban_rate": float(temporal.get("current_ban_rate", 0)),
            
            # Tournament stats
            "tournament_win_rate": float(comprehensive.get("tournament_win_rate", 0)),
            "tournament_pick_rate": float(comprehensive.get("tournament_pick_rate", 0)),
            "tournament_ban_rate": float(comprehensive.get("tournament_ban_rate", 0)),
            "tournament_games": int(comprehensive.get("tournament_games", 0)),
            "tournament_wins": int(comprehensive.get("tournament_wins", 0)),
            
            # Relationships
            "counters": comprehensive.get("counters", "").split("|") if comprehensive.get("counters") else [],
            "strong_against": comprehensive.get("strong_against", "").split("|") if comprehensive.get("strong_against") else [],
            "top_synergies": comprehensive.get("top_synergies", "").split("|") if comprehensive.get("top_synergies") else [],
            "api_counters": counter_names,
        }
    
    # Save
    output_path = DATA_DIR / "MLBB_TRAINING_DATA_FINAL.json"
    with open(output_path, "w") as f:
        json.dump(consolidated, f, indent=2)
    
    size = output_path.stat().st_size
    print(f"\nSaved: {output_path}")
    print(f"Size: {size/1024/1024:.1f}MB")
    print(f"Heroes: {len(consolidated['heroes'])}")
    print(f"Drafts: {len(consolidated['tournament_drafts'])}")
    print(f"Patches: {len(consolidated['patch_timeline'])}")
    print(f"Eras: {len(consolidated['era_stats'])}")


if __name__ == "__main__":
    main()
