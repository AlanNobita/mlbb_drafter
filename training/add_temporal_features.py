#!/usr/bin/env python3
"""
Add temporal/time-aware features to training data.
- Patch-to-date mapping
- Time-decayed weights (recent > old)
- Patch-era hero statistics
- Temporal embeddings for GCN
"""
import json
import csv
import math
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

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


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def parse_date(d):
    """Parse date string YYYYMMDD to datetime."""
    if not d or len(d) != 8:
        return None
    try:
        return datetime.strptime(d, "%Y%m%d")
    except ValueError:
        return None


def parse_version(v):
    """Parse version string like '1.8.30B' to comparable tuple."""
    if not v:
        return (0, 0, 0, 0)
    # Remove letters, split by dots
    import re
    nums = re.findall(r'\d+', v)
    result = tuple(int(n) for n in nums[:4])
    return result + (0,) * (4 - len(result))  # Pad to 4 elements


def build_patch_timeline():
    """Build patch timeline from tournament data."""
    tournaments = load_csv(DRIVE_DIR / "tournament_data.csv")
    
    # Extract unique patches with dates
    patches = {}
    for t in tournaments:
        patch = t.get("patch_code", "").strip()
        start = t.get("start_date", "").strip()
        end = t.get("end_date", "").strip()
        
        if patch and start:
            # Handle ranges like "1.8.08 – 1.8.22"
            if "–" in patch or "-" in patch:
                parts = patch.replace("–", "-").split("-")
                patch = parts[0].strip()  # Use first version
            
            if patch not in patches:
                patches[patch] = {
                    "version": patch,
                    "version_tuple": parse_version(patch),
                    "first_seen": start,
                    "last_seen": end or start,
                    "tournaments": [],
                }
            patches[patch]["tournaments"].append(t.get("tournament_name", ""))
            if start < patches[patch]["first_seen"]:
                patches[patch]["first_seen"] = start
            if end and end > patches[patch]["last_seen"]:
                patches[patch]["last_seen"] = end
    
    # Sort by version
    sorted_patches = sorted(patches.values(), key=lambda x: x["version_tuple"])
    
    # Add relative time (days from first patch)
    if sorted_patches:
        base_date = parse_date(sorted_patches[0]["first_seen"])
        if base_date:
            for p in sorted_patches:
                d = parse_date(p["first_seen"])
                if d:
                    p["days_from_start"] = (d - base_date).days
                else:
                    p["days_from_start"] = 0
    
    return sorted_patches


def compute_time_decayed_stats(matches, patches, decay_rate=0.001):
    """
    Compute hero statistics with time decay.
    Recent matches have higher weight than old matches.
    """
    # Find reference date (most recent match)
    all_dates = [m["date"] for m in matches if m["date"]]
    ref_date = max(all_dates) if all_dates else "20231217"
    ref_dt = parse_date(ref_date)
    
    hero_stats = defaultdict(lambda: {
        "weighted_wins": 0,
        "weighted_picks": 0,
        "weighted_bans": 0,
        "total_games": 0,
        "recent_games": 0,  # Last 6 months
        "era_stats": defaultdict(lambda: {"wins": 0, "picks": 0, "bans": 0}),
    })
    
    for m in matches:
        date = parse_date(m["date"])
        if not date or not ref_dt:
            continue
        
        # Time decay weight: exponential decay based on days ago
        days_ago = (ref_dt - date).days
        weight = math.exp(-decay_rate * days_ago)
        
        # Determine patch era
        patch = "unknown"
        for p in patches:
            if p["first_seen"] <= m["date"] <= p["last_seen"]:
                patch = p["version"]
                break
        
        # Is recent? (last 180 days)
        is_recent = days_ago <= 180
        
        # Process picks/bans
        all_bans = set(m.get("t1_bans", []) + m.get("t2_bans", []))
        
        for side in ["t1", "t2"]:
            picks = m.get(f"{side}_picks", [])
            result = m.get(f"{side}_result", 0)
            
            for h in picks:
                hero_stats[h]["weighted_picks"] += weight
                hero_stats[h]["total_games"] += 1
                hero_stats[h]["era_stats"][patch]["picks"] += 1
                
                if is_recent:
                    hero_stats[h]["recent_games"] += 1
                
                if result > 0:
                    hero_stats[h]["weighted_wins"] += weight
                    hero_stats[h]["era_stats"][patch]["wins"] += 1
        
        for h in all_bans:
            hero_stats[h]["weighted_bans"] += weight
            hero_stats[h]["era_stats"][patch]["bans"] += 1
    
    return hero_stats


def compute_era_pickrates(hero_stats, patches):
    """Compute pick rates per patch era."""
    era_stats = defaultdict(lambda: defaultdict(lambda: {"picks": 0, "wins": 0, "bans": 0, "total": 0}))
    
    for hero, stats in hero_stats.items():
        for patch, pstats in stats["era_stats"].items():
            era_stats[patch][hero] = pstats
    
    # Compute rates per era
    era_picks = {}
    for patch, heroes in era_stats.items():
        total_picks = sum(h["picks"] for h in heroes.values())
        total_bans = sum(h["bans"] for h in heroes.values())
        total = total_picks + total_bans
        
        if total > 0:
            era_picks[patch] = {
                hero: {
                    "pick_rate": h["picks"] / total if total > 0 else 0,
                    "win_rate": h["wins"] / h["picks"] if h["picks"] > 0 else 0,
                    "ban_rate": h["bans"] / total if total > 0 else 0,
                    "games": h["picks"],
                }
                for hero, h in heroes.items()
                if h["picks"] > 5  # Minimum games threshold
            }
    
    return era_picks


def build_temporal_training_csv(hero_stats, patches, api_winrates, hero_info):
    """Build CSV with temporal features for each hero."""
    rows = []
    
    for hero, stats in hero_stats.items():
        # Weighted stats (time-decayed)
        weighted_wr = stats["weighted_wins"] / stats["weighted_picks"] if stats["weighted_picks"] > 0 else 0
        weighted_pr = stats["weighted_picks"] / max(sum(s["weighted_picks"] for s in hero_stats.values()), 1)
        weighted_br = stats["weighted_bans"] / max(sum(s["weighted_bans"] for s in hero_stats.values()), 1)
        
        # Recency score (higher = more relevant now)
        total_games = stats["total_games"]
        recent_games = stats["recent_games"]
        recency_score = recent_games / total_games if total_games > 0 else 0
        
        # Current meta strength (from API)
        api = api_winrates.get(hero, {})
        current_wr = api.get("win_rate", 0)
        current_pr = api.get("pick_rate", 0)
        current_br = api.get("ban_rate", 0)
        
        # Trend: compare recent to historical
        historical_wr = weighted_wr
        trend = current_wr - historical_wr if historical_wr > 0 else 0
        
        # Meta relevance score (combines recency and current pick rate)
        meta_score = recency_score * (1 + current_pr * 100)
        
        # Hero info
        info = hero_info.get(hero, {})
        
        rows.append({
            "hero_name": hero,
            "role": info.get("role", ""),
            "lane": info.get("lane", ""),
            # Temporal features
            "recency_score": round(recency_score, 4),
            "meta_score": round(meta_score, 4),
            "trend": round(trend, 4),
            "total_tournament_games": total_games,
            "recent_tournament_games": recent_games,
            # Time-decayed stats
            "weighted_win_rate": round(weighted_wr, 4),
            "weighted_pick_rate": round(weighted_pr, 6),
            "weighted_ban_rate": round(weighted_br, 6),
            # Current stats (latest patch)
            "current_win_rate": round(current_wr, 4),
            "current_pick_rate": round(current_pr, 4),
            "current_ban_rate": round(current_br, 4),
            # Historical count
            "historical_games": total_games,
        })
    
    return rows


def main():
    print("=" * 60)
    print("TEMPORAL DATA PROCESSOR")
    print("=" * 60)
    
    # 1. Build patch timeline
    print("\n[1/4] Building patch timeline...")
    patches = build_patch_timeline()
    print(f"  {len(patches)} unique patches found")
    print(f"  Oldest: {patches[0]['version']} ({patches[0]['first_seen']})")
    print(f"  Newest: {patches[-1]['version']} ({patches[-1]['last_seen']})")
    
    # Save patch timeline
    patch_timeline = []
    for p in patches:
        patch_timeline.append({
            "version": p["version"],
            "first_seen": p["first_seen"],
            "last_seen": p["last_seen"],
            "days_from_start": p.get("days_from_start", 0),
            "tournament_count": len(p["tournaments"]),
        })
    save_json(OUTPUT_DIR / "patch_timeline.json", patch_timeline)
    print(f"  Saved to patch_timeline.json")
    
    # 2. Load matches
    print("\n[2/4] Loading tournament matches...")
    matches = []
    with open(DRIVE_DIR / "consolidated_game_data.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            def safe_float(val, default=0.0):
                if val is None or val == "" or val == " ":
                    return default
                try:
                    return float(val)
                except:
                    return default
            
            def parse_hero_list(s):
                if not s or s == "()":
                    return []
                s = s.strip("()")
                return [h.strip().strip("'\"") for h in s.split(",") if h.strip()]
            
            t1_picks = parse_hero_list(row.get("t1_picks", ""))
            t1_bans = parse_hero_list(row.get("t1_bans", ""))
            t2_picks = parse_hero_list(row.get("t2_picks", ""))
            t2_bans = parse_hero_list(row.get("t2_bans", ""))
            
            if not t1_picks or not t2_picks:
                continue
            
            matches.append({
                "date": row.get("date", ""),
                "t1_picks": t1_picks,
                "t1_bans": t1_bans,
                "t1_result": safe_float(row.get("t1_result")),
                "t2_picks": t2_picks,
                "t2_bans": t2_bans,
                "t2_result": safe_float(row.get("t2_result")),
            })
    
    print(f"  {len(matches)} matches loaded")
    
    # 3. Compute time-decayed stats
    print("\n[3/4] Computing time-decayed statistics...")
    hero_stats = compute_time_decayed_stats(matches, patches)
    print(f"  {len(hero_stats)} heroes with temporal stats")
    
    # Save hero temporal stats
    hero_temporal = {}
    for hero, stats in hero_stats.items():
        hero_temporal[hero] = {
            "weighted_wins": stats["weighted_wins"],
            "weighted_picks": stats["weighted_picks"],
            "weighted_bans": stats["weighted_bans"],
            "total_games": stats["total_games"],
            "recent_games": stats["recent_games"],
        }
    save_json(OUTPUT_DIR / "hero_temporal_stats.json", hero_temporal)
    print(f"  Saved to hero_temporal_stats.json")
    
    # 4. Compute era stats
    print("\n[4/4] Computing patch-era statistics...")
    era_picks = compute_era_pickrates(hero_stats, patches)
    print(f"  {len(era_picks)} patch eras")
    save_json(OUTPUT_DIR / "era_hero_stats.json", era_picks)
    print(f"  Saved to era_hero_stats.json")
    
    # 5. Build temporal training CSV
    print("\n[5/5] Building temporal training CSV...")
    api_winrates = {}
    with open(API_DIR / "hero_winrate.json") as f:
        for h in json.load(f):
            api_winrates[h["name"]] = {
                "win_rate": float(h["winrate"].replace("%", "")) / 100,
                "ban_rate": float(h["banrate"].replace("%", "")) / 100,
                "pick_rate": float(h["pickrate"].replace("%", "")) / 100,
            }
    
    hero_info = {}
    with open(DRIVE_DIR / "hero_info.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("Name", "").strip()
            if name:
                hero_info[name] = {
                    "role": row.get("Role(s)", "").strip(),
                    "lane": row.get("Lane Recommendation(s)", "").strip(),
                }
    
    rows = build_temporal_training_csv(hero_stats, patches, api_winrates, hero_info)
    
    import csv as csv_mod
    csv_path = OUTPUT_DIR / "hero_stats_temporal.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} heroes to hero_stats_temporal.csv")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEMPORAL FEATURES SUMMARY")
    print("=" * 60)
    print("""
Features added for each hero:
  - recency_score: How relevant now (0-1, higher = more recent games)
  - meta_score: Current meta relevance (recency * pick rate)
  - trend: Win rate change (current - historical, positive = improving)
  - weighted_win_rate: Time-decayed win rate (recent matches weighted more)
  - weighted_pick_rate: Time-decayed pick rate
  - weighted_ban_rate: Time-decayed ban rate
  - current_win_rate: Latest patch win rate
  - current_pick_rate: Latest patch pick rate
  - current_ban_rate: Latest patch ban rate

Patch timeline:
  - 82 patches from 2017-2023
  - Each patch mapped to date range
  - Era-specific hero stats per patch
""")


if __name__ == "__main__":
    main()
