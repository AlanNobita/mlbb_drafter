#!/usr/bin/env python3
"""
Convert all found MLBB datasets to our format.
Sources:
- HuggingFace MPL S14 (z4fL/mpl_s14_dataset)
- Kaggle MPL ID S10 (kishan9044/mpl-id-season10)
- LiTianYeoh Liquipedia data (already integrated)
"""

import csv
import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("training/data")


def convert_mpl_s14_huggingface():
    """Convert MPL S14 HuggingFace dataset."""
    drafts = []
    
    for file in ["/tmp/mpl_id_s14.csv", "/tmp/mpl_ph_s14.csv"]:
        try:
            with open(file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Format: blue_explaner, blue_jungler, blue_midlaner, blue_goldlaner, blue_roamer
                    blue_picks = [
                        row.get("blue_explaner", ""),
                        row.get("blue_jungler", ""),
                        row.get("blue_midlaner", ""),
                        row.get("blue_goldlaner", ""),
                        row.get("blue_roamer", ""),
                    ]
                    red_picks = [
                        row.get("red_explaner", ""),
                        row.get("red_jungler", ""),
                        row.get("red_midlaner", ""),
                        row.get("red_goldlaner", ""),
                        row.get("red_roamer", ""),
                    ]
                    
                    # Skip if any hero is empty
                    if not all(blue_picks) or not all(red_picks):
                        continue
                    
                    # Clean hero names (remove whitespace)
                    blue_picks = [p.strip() for p in blue_picks]
                    red_picks = [p.strip() for p in red_picks]
                    
                    # Winner: result column is "BLUE" or "RED"
                    winner = "t1" if row.get("result", "").upper() == "BLUE" else "t2"
                    
                    # Date format: "aug-9" -> need to add year
                    date_str = row.get("date", "")
                    # Try to parse month-day
                    try:
                        # Format like "aug-9"
                        parts = date_str.split("-")
                        if len(parts) == 2:
                            month_str, day_str = parts
                            # Add year 2024 for MPL S14
                            date_str = f"2024 {month_str} {day_str}"
                            dt = datetime.strptime(date_str, "%Y %b %d")
                            date_str = dt.strftime("%Y%m%d")
                        else:
                            date_str = "20240101"
                    except:
                        date_str = "20240101"
                    
                    # Determine tournament from filename
                    tournament = "MPL ID S14" if "mpl_id" in file else "MPL PH S14"
                    
                    drafts.append({
                        "tournament": tournament,
                        "date": date_str,
                        "blue_picks": blue_picks,
                        "red_picks": red_picks,
                        "blue_bans": [],  # No ban data in this dataset
                        "red_bans": [],
                        "winner": winner,
                    })
        except Exception as e:
            print(f"  Error with {file}: {e}")
    
    return drafts


def convert_kaggle_mpl_s10():
    """Convert Kaggle MPL ID S10 dataset (if available)."""
    # This would need the actual CSV file
    # For now, return empty
    return []


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
    print("Converting additional MLBB datasets...")
    
    all_new_drafts = []
    
    # Convert MPL S14 HuggingFace
    print("\n[1/2] MPL S14 HuggingFace dataset...")
    mpl_s14_drafts = convert_mpl_s14_huggingface()
    print(f"  Converted {len(mpl_s14_drafts)} drafts from MPL S14")
    all_new_drafts.extend(mpl_s14_drafts)
    
    # Convert Kaggle MPL S10 (placeholder)
    print("\n[2/2] Kaggle MPL ID S10 dataset...")
    mpl_s10_drafts = convert_kaggle_mpl_s10()
    print(f"  Converted {len(mpl_s10_drafts)} drafts from MPL S10")
    all_new_drafts.extend(mpl_s10_drafts)
    
    if all_new_drafts:
        print(f"\nTotal new drafts: {len(all_new_drafts)}")
        print("\nMerging with existing data...")
        merged = merge_with_existing(all_new_drafts)
        
        # Sort by date
        merged.sort(key=lambda x: x.get("date", ""), reverse=True)
        
        # Save
        output_path = DATA_DIR / "tournament_drafts.json"
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=2)
        
        print(f"\nSaved {len(merged)} drafts to {output_path}")
        print_stats(merged)
    else:
        print("\nNo new drafts to add")


if __name__ == "__main__":
    main()
