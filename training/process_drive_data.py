#!/usr/bin/env python3
"""
Process ALL MLBB data (API + Drive) into training-ready format for GCN model.
Combines:
- API data (winrates, counters, synergies, relations)
- Drive data (13K+ tournament matches, hero info, tournament metadata)
"""
import json
import csv
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
DRIVE_DIR = BASE_DIR / "drive_data" / "data"
API_DIR = BASE_DIR / "training" / "data" / "api_data"
OUTPUT_DIR = BASE_DIR / "training" / "data"


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_hero_list(s):
    """Parse hero pick/ban tuple string like "('Hero1', 'Hero2')" into list."""
    if not s or s == "()":
        return []
    s = s.strip("()")
    return [h.strip().strip("'\"") for h in s.split(",") if h.strip()]


def extract_hero_info():
    """Extract hero metadata from drive_data/hero_info.csv."""
    rows = load_csv(DRIVE_DIR / "hero_info.csv")
    heroes = {}
    for row in rows:
        name = row.get("Name", "").strip()
        if not name:
            continue
        roles = row.get("Role(s)", "").strip()
        specialty = row.get("Specialty(ies)", "").strip()
        lane = row.get("Lane Recommendation(s)", "").strip()
        release = row.get("Release year", "").strip()
        heroes[name] = {
            "role": roles,
            "specialty": specialty,
            "lane": lane,
            "release_year": release,
        }
    return heroes


def safe_float(val, default=0.0):
    """Safely convert to float."""
    if val is None or val == "" or val == " ":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def extract_tournament_matches():
    """Extract all tournament draft data from consolidated_game_data.csv."""
    rows = load_csv(DRIVE_DIR / "consolidated_game_data.csv")
    matches = []
    for row in rows:
        t1_picks = parse_hero_list(row.get("t1_picks", ""))
        t1_bans = parse_hero_list(row.get("t1_bans", ""))
        t2_picks = parse_hero_list(row.get("t2_picks", ""))
        t2_bans = parse_hero_list(row.get("t2_bans", ""))

        if not t1_picks or not t2_picks:
            continue

        matches.append({
            "date": row.get("date", ""),
            "t1_name": row.get("t1_name", ""),
            "t1_side": row.get("t1_side", ""),
            "t1_picks": t1_picks,
            "t1_bans": t1_bans,
            "t1_result": safe_float(row.get("t1_result")),
            "t2_name": row.get("t2_name", ""),
            "t2_side": row.get("t2_side", ""),
            "t2_picks": t2_picks,
            "t2_bans": t2_bans,
            "t2_result": safe_float(row.get("t2_result")),
            "game_time_sec": safe_float(row.get("game_time_sec")),
            "format": row.get("match_format_BON", ""),
            "stage": row.get("tournament_stage", ""),
        })
    return matches


def extract_tournament_list():
    """Extract tournament metadata from tournament_data.csv."""
    rows = load_csv(DRIVE_DIR / "tournament_data.csv")
    tournaments = []
    for row in rows:
        tournaments.append({
            "code": row.get("tournament_code", ""),
            "name": row.get("tournament_name", ""),
            "tier": row.get("tier", ""),
            "start_date": row.get("start_date", ""),
            "end_date": row.get("end_date", ""),
            "patch": row.get("patch_code", ""),
            "url": row.get("url", ""),
        })
    return tournaments


def extract_api_winrates():
    """Extract winrate data from API."""
    data = load_json(API_DIR / "hero_winrate.json")
    return {
        h["name"]: {
            "win_rate": float(h["winrate"].replace("%", "")) / 100,
            "ban_rate": float(h["banrate"].replace("%", "")) / 100,
            "pick_rate": float(h["pickrate"].replace("%", "")) / 100,
        }
        for h in data
    }


def extract_api_relations():
    """Extract hero relations from OpenMLBB."""
    data = load_json(API_DIR / "openmlbb_heroes.json")
    id_map = {}
    relations = {}
    for rec in data.get("data", {}).get("records", []):
        hd = rec.get("data", {})
        hero_id = hd.get("hero_id")
        hero_name = hd.get("hero", {}).get("data", {}).get("name", "")
        rel = hd.get("relation", {})
        if hero_name:
            id_map[hero_id] = hero_name
            relations[hero_name] = {
                "hero_id": hero_id,
                "assist": rel.get("assist", {}).get("target_hero_id", []),
                "strong": rel.get("strong", {}).get("target_hero_id", []),
                "weak": rel.get("weak", {}).get("target_hero_id", []),
            }
    return relations, id_map


def extract_api_hero_meta():
    """Extract hero metadata from p3hndrx."""
    data = load_json(API_DIR / "hero_meta.json")
    heroes_list = data.get("data", [])
    meta = {}
    for h in heroes_list:
        name = h.get("hero_name", "")
        if name and name != "None":
            meta[name] = {
                "mlid": h.get("mlid", ""),
                "uid": h.get("uid", ""),
                "portrait": h.get("portrait", ""),
                "skills": h.get("skills", []),
            }
    return meta


def compute_tournament_stats(matches):
    """Compute hero statistics from tournament matches."""
    hero_wins = defaultdict(int)
    hero_picks = defaultdict(int)
    hero_bans = defaultdict(int)
    hero_pair_wins = defaultdict(lambda: defaultdict(int))
    hero_pair_picks = defaultdict(lambda: defaultdict(int))

    for m in matches:
        # Team 1
        t1_heroes = set(m["t1_picks"])
        t2_heroes = set(m["t2_picks"])
        all_bans = set(m["t1_bans"] + m["t2_bans"])

        # Picks
        for h in t1_heroes:
            hero_picks[h] += 1
        for h in t2_heroes:
            hero_picks[h] += 1

        # Bans
        for h in all_bans:
            hero_bans[h] += 1

        # Wins
        if m["t1_result"] > 0:
            for h in t1_heroes:
                hero_wins[h] += 1
        if m["t2_result"] > 0:
            for h in t2_heroes:
                hero_wins[h] += 1

        # Synergy tracking (heroes on same team)
        for h1 in t1_heroes:
            for h2 in t1_heroes:
                if h1 != h2:
                    hero_pair_picks[h1][h2] += 1
                    if m["t1_result"] > 0:
                        hero_pair_wins[h1][h2] += 1
        for h1 in t2_heroes:
            for h2 in t2_heroes:
                if h1 != h2:
                    hero_pair_picks[h1][h2] += 1
                    if m["t2_result"] > 0:
                        hero_pair_wins[h1][h2] += 1

    return hero_wins, hero_picks, hero_bans, hero_pair_wins, hero_pair_picks


def build_training_csv(hero_info, api_winrates, hero_wins, hero_picks, hero_bans,
                       relations, id_map, hero_pair_wins, hero_pair_picks):
    """Build comprehensive training CSV combining all data sources."""
    all_heroes = set()
    all_heroes.update(hero_info.keys())
    all_heroes.update(api_winrates.keys())
    all_heroes.update(hero_wins.keys())
    all_heroes.update(hero_picks.keys())

    total_matches = sum(1 for _ in range(1))  # placeholder

    rows = []
    for name in sorted(all_heroes):
        picks = hero_picks.get(name, 0)
        wins = hero_wins.get(name, 0)
        bans = hero_bans.get(name, 0)

        # Tournament stats
        tournament_win_rate = wins / picks if picks > 0 else 0
        tournament_ban_rate = bans / max(sum(hero_bans.values()), 1)
        tournament_pick_rate = picks / max(sum(hero_picks.values()), 1)

        # API stats (if available)
        api = api_winrates.get(name, {})
        api_win = api.get("win_rate", 0)
        api_ban = api.get("ban_rate", 0)
        api_pick = api.get("pick_rate", 0)

        # Hero meta
        meta = hero_info.get(name, {})
        role = meta.get("role", "")
        lane = meta.get("lane", "")
        specialty = meta.get("specialty", "")

        # Synergy data - find top synergies
        synergies = hero_pair_picks.get(name, {})
        top_synergies = sorted(synergies.items(), key=lambda x: x[1], reverse=True)[:5]
        synergy_names = [s[0] for s in top_synergies]

        # Counter data from API
        rel = relations.get(name, {})
        id_to_name = {v: k for k, v in id_map.items()}
        counter_ids = rel.get("weak", [])
        counter_names = [id_to_name.get(cid, str(cid)) for cid in counter_ids if cid in id_to_name]
        strong_ids = rel.get("strong", [])
        strong_names = [id_to_name.get(cid, str(cid)) for cid in strong_ids if cid in id_to_name]

        rows.append({
            "hero_name": name,
            "role": role,
            "lane": lane,
            "specialty": specialty,
            # Tournament data (REAL drafts)
            "tournament_win_rate": round(tournament_win_rate, 4),
            "tournament_pick_rate": round(tournament_pick_rate, 6),
            "tournament_ban_rate": round(tournament_ban_rate, 6),
            "tournament_games": picks,
            "tournament_wins": wins,
            # API data
            "api_win_rate": round(api_win, 4),
            "api_ban_rate": round(api_ban, 4),
            "api_pick_rate": round(api_pick, 4),
            # Relationships
            "counters": "|".join(counter_names[:5]),
            "strong_against": "|".join(strong_names[:5]),
            "top_synergies": "|".join(synergy_names),
        })

    return rows


def build_draft_training_data(matches):
    """Build draft sequence training data for the GCN model."""
    drafts = []
    for m in matches:
        # Represent draft as sequence of picks/bans
        draft = {
            "date": m["date"],
            "stage": m["stage"],
            "format": m["format"],
            "game_time_sec": m["game_time_sec"],
            "winner": "t1" if m["t1_result"] > 0 else "t2",
            "blue_picks": m["t1_picks"] if m["t1_side"] == "blue" else m["t2_picks"],
            "blue_bans": m["t1_bans"] if m["t1_side"] == "blue" else m["t2_bans"],
            "red_picks": m["t2_picks"] if m["t1_side"] == "blue" else m["t1_picks"],
            "red_bans": m["t2_bans"] if m["t1_side"] == "blue" else m["t1_bans"],
            "blue_team": m["t1_name"] if m["t1_side"] == "blue" else m["t2_name"],
            "red_team": m["t2_name"] if m["t1_side"] == "blue" else m["t1_name"],
        }
        drafts.append(draft)
    return drafts


def build_synergy_matrix(hero_pair_wins, hero_pair_picks, all_heroes):
    """Build hero synergy matrix from tournament data."""
    n = len(all_heroes)
    name_to_idx = {name: i for i, name in enumerate(sorted(all_heroes))}
    matrix = [[0.0] * n for _ in range(n)]

    for h1 in hero_pair_picks:
        if h1 not in name_to_idx:
            continue
        i = name_to_idx[h1]
        for h2, picks in hero_pair_picks[h1].items():
            if h2 not in name_to_idx:
                continue
            j = name_to_idx[h2]
            wins = hero_pair_wins[h1].get(h2, 0)
            wr = wins / picks if picks > 0 else 0.5
            # Convert to synergy score: >0.5 = positive synergy, <0.5 = negative
            matrix[i][j] = (wr - 0.5) * 2  # Range: -1 to 1

    return sorted(all_heroes), matrix


def main():
    print("=" * 60)
    print("MLBB COMPREHENSIVE DATA PROCESSOR")
    print("=" * 60)

    # 1. Load all data
    print("\n[1/7] Loading hero info...")
    hero_info = extract_hero_info()
    print(f"  {len(hero_info)} heroes with metadata")

    print("\n[2/7] Loading tournament matches...")
    matches = extract_tournament_matches()
    print(f"  {len(matches)} tournament matches loaded")

    print("\n[3/7] Loading tournament list...")
    tournaments = extract_tournament_list()
    print(f"  {len(tournaments)} tournaments")

    print("\n[4/7] Loading API winrates...")
    api_winrates = extract_api_winrates()
    print(f"  {len(api_winrates)} heroes with API stats")

    print("\n[5/7] Loading API relations...")
    relations, id_map = extract_api_relations()
    print(f"  {len(relations)} heroes with relations")

    print("\n[6/7] Computing tournament statistics...")
    hero_wins, hero_picks, hero_bans, hero_pair_wins, hero_pair_picks = compute_tournament_stats(matches)
    print(f"  {len(hero_picks)} unique heroes picked in tournaments")
    print(f"  {sum(hero_picks.values())} total picks")
    print(f"  {sum(hero_bans.values())} total bans")

    # 2. Build training CSV
    print("\n[7/7] Building training files...")
    rows = build_training_csv(
        hero_info, api_winrates, hero_wins, hero_picks, hero_bans,
        relations, id_map, hero_pair_wins, hero_pair_picks
    )
    csv_path = OUTPUT_DIR / "hero_stats_comprehensive.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} heroes to {csv_path.name}")

    # 3. Build draft training data
    drafts = build_draft_training_data(matches)
    drafts_path = OUTPUT_DIR / "tournament_drafts.json"
    with open(drafts_path, "w") as f:
        json.dump(drafts, f, indent=2)
    print(f"  Saved {len(drafts)} draft records to {drafts_path.name}")

    # 4. Build synergy matrix
    hero_names, synergy_matrix = build_synergy_matrix(hero_pair_wins, hero_pair_picks,
                                                        set(hero_info.keys()) | set(api_winrates.keys()))
    synergy_path = OUTPUT_DIR / "synergy_matrix_tournament.json"
    with open(synergy_path, "w") as f:
        json.dump({"hero_names": hero_names, "synergy_matrix": synergy_matrix}, f)
    print(f"  Saved {len(hero_names)}x{len(hero_names)} synergy matrix to {synergy_path.name}")

    # 5. Build hero pair co-occurrence
    cooccurrence = {}
    for h1 in hero_pair_picks:
        cooccurrence[h1] = {}
        for h2 in hero_pair_picks[h1]:
            cooccurrence[h1][h2] = {
                "games_together": hero_pair_picks[h1][h2],
                "wins_together": hero_pair_wins[h1][h2],
                "win_rate": hero_pair_wins[h1][h2] / hero_pair_picks[h1][h2] if hero_pair_picks[h1][h2] > 0 else 0,
            }
    cooc_path = OUTPUT_DIR / "hero_cooccurrence_tournament.json"
    with open(cooc_path, "w") as f:
        json.dump(cooccurrence, f)
    print(f"  Saved co-occurrence data to {cooc_path.name}")

    # 6. Summary
    print("\n" + "=" * 60)
    print("TRAINING DATA SUMMARY")
    print("=" * 60)
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print(f"\n{'File':<45} {'Size':>10} {'Description'}")
    print("-" * 80)

    files = [
        ("hero_stats_comprehensive.csv", "Hero stats (tournament + API)"),
        ("tournament_drafts.json", "Tournament draft sequences"),
        ("synergy_matrix_tournament.json", "Hero synergy matrix"),
        ("hero_cooccurrence_tournament.json", "Hero pair co-occurrence"),
        ("hero_stats_real.csv", "Hero stats (API only)"),
        ("adjacency_real.json", "Adjacency matrix (API)"),
        ("mlbb_data_consolidated.json", "Consolidated API data"),
    ]

    for fname, desc in files:
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            size = fpath.stat().st_size
            print(f"  {fname:<43} {size/1024:>8.1f}KB  {desc}")

    print(f"\nTournament data:")
    print(f"  {len(matches)} matches from {len(tournaments)} tournaments")
    print(f"  Date range: {min(m['date'] for m in matches)} - {max(m['date'] for m in matches)}")
    print(f"  {len(hero_picks)} unique heroes in tournaments")

    # Tournament breakdown
    stage_counts = defaultdict(int)
    for m in matches:
        stage_counts[m["stage"]] += 1
    print(f"\nMatches by stage:")
    for stage, count in sorted(stage_counts.items(), key=lambda x: -x[1]):
        print(f"  {stage}: {count}")


if __name__ == "__main__":
    main()
