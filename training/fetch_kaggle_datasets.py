#!/usr/bin/env python3
"""
Download MLBB tournament datasets from Kaggle.

Available datasets:
- M7 World Championship 2026: double0x2/m7-world-championship-2026-mlbb-result-stats
- M5 World Knockout: bcakra/mobile-legend-m5-world-knockout-stage-results
- MLBB Draft Breakdown: gerryzani/mlbb-draft-breakdown-patch-1768
- MLBB Match Results: rizqinur/mobile-legends-match-results

Requires: kaggle CLI or manual download
Usage:
    python training/fetch_kaggle_datasets.py --auto
    python training/fetch_kaggle_datasets.py --download-all
"""

import argparse
import json
import os
import subprocess
import zipfile
import csv
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("training/data")
RAW_DIR = DATA_DIR / "kaggle_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

KAGGLE_DATASETS = [
    {
        "name": "M7 World Championship 2026",
        "slug": "double0x2/m7-world-championship-2026-mlbb-result-stats",
        "files": [
            "m7_2026_match_results_esportspass.csv",
            "m7_2026_team_stats_esportspass.csv",
            "m7_2026_teams_esportspass.csv",
        ],
        "has_drafts": False,  # Series-level only, no per-game drafts
    },
    {
        "name": "M5 World Knockout Stage",
        "slug": "bcakra/mobile-legend-m5-world-knockout-stage-results",
        "files": [],
        "has_drafts": True,
    },
    {
        "name": "MLBB Draft Breakdown Patch 1.7.68",
        "slug": "gerryzani/mlbb-draft-breakdown-patch-1768",
        "files": [],
        "has_drafts": True,
    },
    {
        "name": "MLBB Match Results",
        "slug": "rizqinur/mobile-legends-match-results",
        "files": [],
        "has_drafts": True,
    },
]


def check_kaggle_cli():
    """Check if kaggle CLI is installed."""
    try:
        result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_dataset(slug, output_dir):
    """Download a Kaggle dataset."""
    print(f"  Downloading {slug}...")
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(output_dir), "--unzip"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  Success!")
            return True
        else:
            print(f"  Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def download_manual(slug, output_dir):
    """Provide instructions for manual download."""
    print(f"\n  Manual download required for: {slug}")
    print(f"  URL: https://www.kaggle.com/datasets/{slug}")
    print(f"  Download and extract to: {output_dir}")
    return False


def parse_m5_data(filepath):
    """Parse M5 World Knockout Stage dataset."""
    drafts = []
    try:
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try to extract draft data
                blue_picks = row.get("blue_picks", row.get("team1_picks", ""))
                red_picks = row.get("red_picks", row.get("team2_picks", ""))
                winner = row.get("winner", row.get("result", ""))

                if blue_picks and red_picks:
                    # Parse JSON arrays if needed
                    if isinstance(blue_picks, str) and blue_picks.startswith("["):
                        blue_picks = json.loads(blue_picks)
                        red_picks = json.loads(red_picks)

                    drafts.append({
                        "tournament": "M5 World Championship",
                        "date": row.get("date", ""),
                        "blue_picks": blue_picks if isinstance(blue_picks, list) else [],
                        "red_picks": red_picks if isinstance(red_picks, list) else [],
                        "blue_bans": [],
                        "red_bans": [],
                        "winner": "t1" if "blue" in str(winner).lower() or "1" in str(winner) else "t2",
                    })
    except Exception as e:
        print(f"  Error parsing {filepath}: {e}")
    return drafts


def parse_draft_data(filepath):
    """Generic draft data parser."""
    drafts = []
    try:
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []

            # Detect column names
            blue_cols = [c for c in columns if "blue" in c.lower() and "pick" in c.lower()]
            red_cols = [c for c in columns if "red" in c.lower() and "pick" in c.lower()]
            winner_cols = [c for c in columns if "winner" in c.lower() or "result" in c.lower()]

            if blue_cols and red_cols:
                for row in reader:
                    blue_picks = row.get(blue_cols[0], "")
                    red_picks = row.get(red_cols[0], "")
                    winner = row.get(winner_cols[0], "") if winner_cols else ""

                    if isinstance(blue_picks, str) and blue_picks.startswith("["):
                        try:
                            blue_picks = json.loads(blue_picks)
                            red_picks = json.loads(red_picks)
                        except:
                            blue_picks = [h.strip() for h in blue_picks.strip("[]").split(",")]
                            red_picks = [h.strip() for h in red_picks.strip("[]").split(",")]

                    if isinstance(blue_picks, list) and len(blue_picks) >= 5:
                        drafts.append({
                            "tournament": "Kaggle Dataset",
                            "date": row.get("date", row.get("match_date", "")),
                            "blue_picks": blue_picks[:5],
                            "red_picks": red_picks[:5] if isinstance(red_picks, list) else [],
                            "blue_bans": [],
                            "red_bans": [],
                            "winner": "t1" if "blue" in str(winner).lower() or "1" in str(winner) else "t2",
                        })
    except Exception as e:
        print(f"  Error parsing {filepath}: {e}")
    return drafts


def merge_with_existing(new_drafts):
    """Merge new drafts with existing tournament_drafts.json."""
    existing_path = DATA_DIR / "tournament_drafts.json"

    if existing_path.exists():
        with open(existing_path) as f:
            existing_drafts = json.load(f)

        existing_sigs = set()
        for d in existing_drafts:
            sig = (
                d.get("date", ""),
                d.get("tournament", ""),
                tuple(d.get("blue_picks", [])),
                tuple(d.get("red_picks", [])),
                d.get("winner", ""),
            )
            existing_sigs.add(sig)

        merged = existing_drafts.copy()
        added = 0
        for d in new_drafts:
            sig = (
                d.get("date", ""),
                d.get("tournament", ""),
                tuple(d.get("blue_picks", [])),
                tuple(d.get("red_picks", [])),
                d.get("winner", ""),
            )
            if sig not in existing_sigs:
                merged.append(d)
                existing_sigs.add(sig)
                added += 1

        print(f"  Merged: {added} new drafts added to {len(existing_drafts)} existing")
        return merged
    else:
        print(f"  No existing data, using {len(new_drafts)} new drafts")
        return new_drafts


def main():
    parser = argparse.ArgumentParser(description="Download MLBB Kaggle datasets")
    parser.add_argument("--auto", action="store_true", help="Auto-download if kaggle CLI available")
    parser.add_argument("--download-all", action="store_true", help="Download all datasets")
    parser.add_argument("--dataset", type=str, help="Download specific dataset slug")
    args = parser.parse_args()

    print("=" * 60)
    print("KAGGLE DATASET DOWNLOADER")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    has_kaggle = check_kaggle_cli()
    print(f"Kaggle CLI: {'Available' if has_kaggle else 'Not installed'}")

    if not has_kaggle:
        print("\nTo install kaggle CLI:")
        print("  pip install kaggle")
        print("  kaggle datasets list -m MLBB")
        print("\nOr download manually from:")
        for ds in KAGGLE_DATASETS:
            print(f"  https://www.kaggle.com/datasets/{ds['slug']}")

    # Download datasets
    all_new_drafts = []

    for ds in KAGGLE_DATASETS:
        if args.dataset and args.dataset not in ds["slug"]:
            continue

        print(f"\n{'='*40}")
        print(f"Dataset: {ds['name']}")
        print(f"Slug: {ds['slug']}")
        print(f"Has drafts: {ds['has_drafts']}")

        output_dir = RAW_DIR / ds["slug"].split("/")[-1]
        output_dir.mkdir(parents=True, exist_ok=True)

        if has_kaggle and (args.auto or args.download_all):
            success = download_dataset(ds["slug"], output_dir)
        else:
            success = download_manual(ds["slug"], output_dir)

        # Parse downloaded files
        if success or output_dir.exists():
            for csv_file in output_dir.glob("*.csv"):
                print(f"  Parsing {csv_file.name}...")
                drafts = parse_draft_data(csv_file)
                if drafts:
                    print(f"    Found {len(drafts)} drafts")
                    all_new_drafts.extend(drafts)

    # Merge all new drafts
    if all_new_drafts:
        print(f"\n{'='*40}")
        print(f"Total new drafts: {len(all_new_drafts)}")
        merged = merge_with_existing(all_new_drafts)

        with open(DATA_DIR / "tournament_drafts.json", "w") as f:
            json.dump(merged, f, indent=2)
        print(f"  Saved {len(merged)} total drafts")

    # Summary
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Datasets processed: {len(KAGGLE_DATASETS)}")
    print(f"  New drafts found: {len(all_new_drafts)}")
    print(f"\nManual download URLs:")
    for ds in KAGGLE_DATASETS:
        print(f"  {ds['name']}: https://www.kaggle.com/datasets/{ds['slug']}")


if __name__ == "__main__":
    main()
