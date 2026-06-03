#!/usr/bin/env python3
"""
Convert LiTianYeoh's scraped Liquipedia data to our format.
Merges with existing tournament_drafts.json.
"""

import csv
import json
import ast
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("training/data")
LIQUID_DATA = Path("/tmp/MLBB_Tournament_Analysis/scraping/tournament_game_info/output")


def parse_tuple_string(s):
    """Parse a tuple string like "('Hero1', 'Hero2')" to list."""
    try:
        # Clean up the string
        s = s.strip()
        if s.startswith("(") and s.endswith(")"):
            return list(ast.literal_eval(s))
        return []
    except:
        return []


def convert_to_our_format():
    """Convert LiTianYeoh's CSV to our JSON format."""
    # Load tournament metadata
    tournament_meta = {}
    with open(LIQUID_DATA / "tournament_data.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tournament_meta[row["tournament_code"]] = {
                "name": row["tournament_name"],
                "tier": row["tier"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "patch": row["patch_code"],
            }
    
    print(f"Loaded {len(tournament_meta)} tournament metadata entries")
    
    # Convert game data
    drafts = []
    with open(LIQUID_DATA / "consolidated_game_data_20240505.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Get tournament info
            t_code = row["tournament_code"]
            t_info = tournament_meta.get(t_code, {})
            tournament_name = t_info.get("name", f"Tournament_{t_code}")
            
            # Parse picks and bans
            t1_picks = parse_tuple_string(row["t1_picks"])
            t1_bans = parse_tuple_string(row["t1_bans"])
            t2_picks = parse_tuple_string(row["t2_picks"])
            t2_bans = parse_tuple_string(row["t2_bans"])
            
            # Skip if not enough picks
            if len(t1_picks) < 5 or len(t2_picks) < 5:
                continue
            
            # Determine sides
            t1_side = row["t1_side"].lower()
            t2_side = row["t2_side"].lower()
            
            # Blue/Red picks
            if t1_side == "blue":
                blue_picks = t1_picks
                blue_bans = t1_bans
                red_picks = t2_picks
                red_bans = t2_bans
                # Winner: t1_result is 1.0 if t1 won
                winner = "t1" if float(row["t1_result"]) == 1.0 else "t2"
            else:
                blue_picks = t2_picks
                blue_bans = t2_bans
                red_picks = t1_picks
                red_bans = t1_bans
                winner = "t1" if float(row["t2_result"]) == 1.0 else "t2"
            
            # Parse date
            date_str = row["date"]
            if "." in date_str:
                date_str = date_str.split(".")[0]
            try:
                # Format: 20231203
                datetime.strptime(date_str, "%Y%m%d")
            except:
                date_str = t_info.get("start_date", "20230101")
                if "." in date_str:
                    date_str = date_str.split(".")[0]
            
            drafts.append({
                "tournament": tournament_name,
                "date": date_str,
                "blue_picks": blue_picks,
                "red_picks": red_picks,
                "blue_bans": blue_bans,
                "red_bans": red_bans,
                "winner": winner,
            })
    
    print(f"Converted {len(drafts)} drafts")
    return drafts


def merge_with_existing(new_drafts):
    """Merge new drafts with existing tournament_drafts.json."""
    existing_path = DATA_DIR / "tournament_drafts.json"
    
    if existing_path.exists():
        with open(existing_path) as f:
            existing_drafts = json.load(f)
        
        # Create set of existing draft signatures for dedup
        existing_sigs = set()
        for d in existing_drafts:
            sig = (
                d.get("tournament", ""),
                d.get("date", ""),
                tuple(d.get("blue_picks", [])),
                tuple(d.get("red_picks", [])),
            )
            existing_sigs.add(sig)
        
        # Merge new drafts (skip duplicates)
        merged = existing_drafts.copy()
        added = 0
        for d in new_drafts:
            sig = (
                d.get("tournament", ""),
                d.get("date", ""),
                tuple(d.get("blue_picks", [])),
                tuple(d.get("red_picks", [])),
            )
            if sig not in existing_sigs:
                merged.append(d)
                existing_sigs.add(sig)
                added += 1
        
        print(f"Merged: {added} new drafts added, {len(existing_drafts)} existing")
        return merged
    else:
        print(f"No existing data, using {len(new_drafts)} new drafts")
        return new_drafts


def print_stats(drafts):
    """Print statistics about drafts."""
    if not drafts:
        print("No drafts to analyze")
        return
    
    # Count by tournament
    tournaments = {}
    for d in drafts:
        t = d.get("tournament", "Unknown")
        tournaments[t] = tournaments.get(t, 0) + 1
    
    # Count by year
    years = {}
    for d in drafts:
        date_str = d.get("date", "")
        if len(date_str) >= 4:
            year = date_str[:4]
            years[year] = years.get(year, 0) + 1
    
    print(f"\n{'=' * 60}")
    print(f"DATA STATISTICS")
    print(f"{'=' * 60}")
    print(f"Total drafts: {len(drafts)}")
    print(f"\nBy tournament (top 30):")
    for t, count in sorted(tournaments.items(), key=lambda x: -x[1])[:30]:
        print(f"  {t}: {count}")
    
    print(f"\nBy year:")
    for y, count in sorted(years.items()):
        print(f"  {y}: {count}")


def main():
    print("Converting LiTianYeoh's Liquipedia data...")
    new_drafts = convert_to_our_format()
    
    print("\nMerging with existing data...")
    merged = merge_with_existing(new_drafts)
    
    # Sort by date
    merged.sort(key=lambda x: x.get("date", ""), reverse=True)
    
    # Save
    output_path = DATA_DIR / "tournament_drafts.json"
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    
    print(f"\nSaved {len(merged)} drafts to {output_path}")
    print_stats(merged)


if __name__ == "__main__":
    main()
