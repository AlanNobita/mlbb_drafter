#!/usr/bin/env python3
"""
Liquipedia Scraper for MLBB Tournament Drafts
Fetches MPL, M-series, MSC, MPLI tournament data
Uses datetime for time-aware analysis (no hardcoded years)
"""

import json
import re
import time
import gzip
import io
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from typing import Dict, List, Optional, Tuple
import argparse

# Paths
DATA_DIR = Path("training/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Liquipedia API config
LIQUIPEDIA_API = "https://liquipedia.net/mobilelegends/api.php"
USER_AGENT = "MLBBDataCollector/1.0 (mlbb-drafter project)"
REQUEST_DELAY = 2.1  # Liquipedia requires 1 request per 2 seconds

# Tournament categories to scrape
# Format: (category_page, tournament_type)
TOURNAMENT_CATEGORIES = [
    ("MPL/Indonesia", "MPL_ID"),
    ("MPL/Philippines", "MPL_PH"),
    ("MPL/Malaysia", "MPL_MY"),
    ("MPL/Singapore", "MPL_SG"),
    ("MSC", "MSC"),
    ("M-series", "M_Series"),
    ("MPL_Invitational", "MPLI"),
    ("ESL/Snapdragon_Pro_Series", "ESL"),
]

# S-Tier tournaments (World Championship, MSC)
S_TIER_PAGES = [
    # M-series (actual match data is on subpages)
    ("M1_World_Championship", ["M1_World_Championship"]),
    ("M2_World_Championship", ["M2_World_Championship"]),
    ("M3_World_Championship", ["M3_World_Championship"]),
    ("M4_World_Championship", [
        "M4_World_Championship/Group_Stage",
        "M4_World_Championship/Knockout_Stage",
    ]),
    ("M5_World_Championship", [
        "M5_World_Championship/Group_Stage",
        "M5_World_Championship/Knockout_Stage",
    ]),
    ("M6_World_Championship", [
        "M6_World_Championship/Group_Stage",
        "M6_World_Championship/Knockout_Stage",
    ]),
    ("M7_World_Championship", [
        "M7_World_Championship/Group_Stage",
        "M7_World_Championship/Knockout_Stage",
    ]),
    # MSC
    ("MSC/2023", ["MSC/2023"]),
    ("MSC/2024", ["MSC/2024"]),
    ("MSC/2025", [
        "MSC/2025/Group_Stage",
        "MSC/2025/Knockout_Stage",
    ]),
]


def fetch_liquipedia_page(page_title: str, retries: int = 5) -> Optional[str]:
    """Fetch a Liquipedia page via API with rate limiting.
    
    Liquipedia rate limits:
    - 1 request per 2 seconds for wikicode API
    - 429 = Too Many Requests (back off 30s+)
    - Need proper User-Agent per their ToS
    """
    params = f"action=parse&page={page_title}&prop=wikitext&format=json"
    url = f"{LIQUIPEDIA_API}?{params}"
    
    for attempt in range(retries):
        try:
            # Ensure minimum delay between requests
            time.sleep(REQUEST_DELAY)
            
            req = Request(url, headers={
                "User-Agent": USER_AGENT,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json",
            })
            
            with urlopen(req, timeout=30) as response:
                # Handle gzip
                data = response.read()
                if data[:2] == b'\x1f\x8b':  # gzip magic bytes
                    data = gzip.decompress(data)
                
                result = json.loads(data.decode("utf-8"))
                
                if "parse" in result:
                    return result["parse"]["wikitext"]["*"]
                elif "error" in result:
                    print(f"  API error: {result['error'].get('info', 'Unknown')}")
                    return None
                    
        except HTTPError as e:
            if e.code == 429:  # Rate limited
                # Exponential backoff: 30s, 60s, 120s, 240s, 480s
                wait = 30 * (2 ** attempt)
                print(f"  Rate limited (attempt {attempt+1}/{retries}), waiting {wait}s...")
                time.sleep(wait)
            elif e.code == 404:
                print(f"  Page not found: {page_title}")
                return None
            else:
                print(f"  HTTP error {e.code} for {page_title}")
                time.sleep(REQUEST_DELAY * 2)
        except (URLError, TimeoutError) as e:
            print(f"  Network error: {e}")
            time.sleep(REQUEST_DELAY * 3)
        except Exception as e:
            print(f"  Unexpected error: {e}")
            return None
    
    print(f"  Failed to fetch {page_title} after {retries} attempts")
    return None


def discover_seasons(category: str) -> List[str]:
    """Discover season pages for a tournament category."""
    # Fetch category page to find season links
    wikitext = fetch_liquipedia_page(category)
    if not wikitext:
        return []
    
    # Find season links (e.g., MPL/Indonesia/Season_14)
    season_pattern = re.compile(r'\[\[(?:MPL|MSC|M[0-9]|ESL)[^\]]*?/Season[_ ](\d+)')
    seasons = season_pattern.findall(wikitext)
    
    # Also check for direct season references
    season_pattern2 = re.compile(r'Season[_ ](\d+)')
    seasons.extend(season_pattern2.findall(wikitext))
    
    # Deduplicate and sort
    unique_seasons = sorted(set(int(s) for s in seasons if s.isdigit()))
    return [str(s) for s in unique_seasons]


def parse_wikitext_drafts(wikitext: str, tournament_name: str, date_str: str = None) -> List[Dict]:
    """Parse wikitext to extract draft data."""
    drafts = []
    
    # Find all Map sections directly in wikitext
    # Each map has: |mapN={{Map|...|team1side=X|team2side=Y|winner=Z|...t1h1=hero|...}}
    map_pattern = re.compile(
        r'\|map(\d+)\s*=\s*\{\{Map\|'
        r'[^|]*\|'  # skip vod=
        r'([^|]*)\|'  # team1side=blue| or other params
        r'([^|]*)\|'  # team2side=red| or other params
        r'([^|]*)\|',  # winner=1| or other params
        re.DOTALL
    )
    
    # Simpler approach: find all maps with their context
    # Split by |map to get each map section
    map_sections = re.split(r'(?=\|map\d+\s*=\s*\{\{Map)', wikitext)
    
    for section in map_sections:
        # Skip if not a map section
        if not section.startswith('|map'):
            continue
        
        # Extract map number
        map_num_match = re.match(r'\|map(\d+)', section)
        if not map_num_match:
            continue
        
        # Extract sides
        team1_side_match = re.search(r'\|team1side=(blue|red)', section)
        team2_side_match = re.search(r'\|team2side=(red|blue)', section)
        winner_match = re.search(r'\|winner=(\w+)', section)
        
        if not (team1_side_match and team2_side_match and winner_match):
            continue
        
        team1_side = team1_side_match.group(1)
        team2_side = team2_side_match.group(1)
        winner_raw = winner_match.group(1)
        
        # Normalize winner
        if winner_raw in ("1", "t1"):
            winner_num = 1
        elif winner_raw in ("2", "t2"):
            winner_num = 2
        elif winner_raw == "blue":
            winner_num = 1 if team1_side == "blue" else 2
        elif winner_raw == "red":
            winner_num = 1 if team1_side == "red" else 2
        else:
            try:
                winner_num = int(winner_raw)
            except ValueError:
                continue
        
        # Extract hero picks for team 1
        t1_picks = []
        for i in range(1, 6):
            pick = re.search(rf'\|t1h{i}=([^\n|]+)', section)
            if pick:
                t1_picks.append(pick.group(1).strip().title())
        
        # Extract hero bans for team 1
        t1_bans = []
        for i in range(1, 6):
            ban = re.search(rf'\|t1b{i}=([^\n|]+)', section)
            if ban:
                t1_bans.append(ban.group(1).strip().title())
        
        # Extract hero picks for team 2
        t2_picks = []
        for i in range(1, 6):
            pick = re.search(rf'\|t2h{i}=([^\n|]+)', section)
            if pick:
                t2_picks.append(pick.group(1).strip().title())
        
        # Extract hero bans for team 2
        t2_bans = []
        for i in range(1, 6):
            ban = re.search(rf'\|t2b{i}=([^\n|]+)', section)
            if ban:
                t2_bans.append(ban.group(1).strip().title())
        
        # Skip if not enough picks
        if len(t1_picks) < 5 or len(t2_picks) < 5:
            continue
        
        # Determine blue/red team picks based on side
        if team1_side == "blue":
            blue_picks = t1_picks
            blue_bans = t1_bans
            red_picks = t2_picks
            red_bans = t2_bans
            winner_str = "t1" if winner_num == 1 else "t2"
        else:
            blue_picks = t2_picks
            blue_bans = t2_bans
            red_picks = t1_picks
            red_bans = t1_bans
            winner_str = "t1" if winner_num == 2 else "t2"
        
        # Try to find date from surrounding context
        # Look backwards in wikitext for the match date
        map_pos = wikitext.find(section[:50])
        if map_pos > 0:
            # Search backwards for |date=
            preceding = wikitext[max(0, map_pos-2000):map_pos]
            date_match = re.search(r'\|date=([^\n|]+)', preceding[::-1])
            if date_match:
                date_str_found = date_match.group(1)[::-1]
            else:
                date_str_found = date_str
        else:
            date_str_found = date_str
        
        parsed_date = parse_date(date_str_found)
        
        drafts.append({
            "tournament": tournament_name,
            "date": parsed_date,
            "blue_picks": blue_picks,
            "red_picks": red_picks,
            "blue_bans": blue_bans,
            "red_bans": red_bans,
            "winner": winner_str,
        })
    
    return drafts


def parse_date(date_str: str) -> str:
    """Parse various date formats to ISO format (YYYYMMDD)."""
    if not date_str:
        return datetime.now().strftime("%Y%m%d")
    
    date_str = date_str.strip()
    
    # Try various formats
    formats = [
        "%B %d, %Y",           # October 23, 2024
        "%b %d, %Y",           # Oct 23, 2024
        "%Y-%m-%d",            # 2024-10-23
        "%d/%m/%Y",            # 23/10/2024
        "%m/%d/%Y",            # 10/23/2024
        "%Y%m%d",              # 20241023
        "%B %d, %Y - %H:%M",  # October 23, 2024 - 13:00
        "%b %d, %Y - %H:%M",  # Oct 23, 2024 - 13:00
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.split("{{")[0].strip(), fmt)
            return dt.strftime("%Y%m%d")
        except ValueError:
            continue
    
    # Try to extract just the year-month-day with regex
    match = re.search(r'(\w+)\s+(\d+),?\s+(\d{4})', date_str)
    if match:
        month_str, day_str, year_str = match.groups()
        try:
            dt = datetime.strptime(f"{month_str} {day_str} {year_str}", "%B %d %Y")
            return dt.strftime("%Y%m%d")
        except ValueError:
            pass
    
    # Fallback: use current date
    return datetime.now().strftime("%Y%m%d")


def scrape_tournament_page(page_title: str, tournament_name: str) -> List[Dict]:
    """Scrape a single tournament page for draft data."""
    print(f"  Fetching {page_title}...")
    wikitext = fetch_liquipedia_page(page_title)
    
    if not wikitext:
        return []
    
    drafts = parse_wikitext_drafts(wikitext, tournament_name)
    return drafts


def scrape_all_tournaments(
    start_year: int = None,
    end_year: int = None,
    max_requests: int = None
) -> List[Dict]:
    """
    Scrape all MLBB tournaments from Liquipedia.
    
    Args:
        start_year: Minimum year (None = auto-detect from data)
        end_year: Maximum year (None = current year)
        max_requests: Maximum API requests (None = unlimited)
    """
    # Use datetime for time-aware analysis
    now = datetime.now()
    if end_year is None:
        end_year = now.year
    
    all_drafts = []
    request_count = 0
    
    print(f"Scraping MLBB tournaments from Liquipedia")
    print(f"Year range: {start_year or 'auto'} to {end_year}")
    print("=" * 60)
    
    # Scrape S-Tier tournaments (M-series, MSC)
    print("\n[1/4] S-Tier Tournaments (M-series, MSC)...")
    for tournament_name, pages in S_TIER_PAGES:
        if max_requests and request_count >= max_requests:
            print(f"  Reached max requests ({max_requests})")
            break
        
        for page in pages:
            if max_requests and request_count >= max_requests:
                break
            
            drafts = scrape_tournament_page(page, tournament_name)
            all_drafts.extend(drafts)
            request_count += 1
            
            if drafts:
                print(f"    {page}: {len(drafts)} drafts")
            
            time.sleep(REQUEST_DELAY)
    
    # Scrape MPL seasons
    print("\n[2/4] MPL Regional Seasons...")
    for category, region in TOURNAMENT_CATEGORIES:
        if max_requests and request_count >= max_requests:
            break
        
        print(f"\n  {region}:")
        
        # Discover seasons
        seasons = discover_seasons(category)
        if not seasons:
            # Try common season numbers
            seasons = [str(s) for s in range(1, 16)]
        
        for season in seasons:
            if max_requests and request_count >= max_requests:
                break
            
            # Construct page title
            page_title = f"{category}/Season_{season}"
            tournament_name = f"{region} S{season}"
            
            drafts = scrape_tournament_page(page_title, tournament_name)
            all_drafts.extend(drafts)
            request_count += 1
            
            if drafts:
                print(f"    Season {season}: {len(drafts)} drafts")
            
            time.sleep(REQUEST_DELAY)
    
    # Scrape specific tournament pages
    print("\n[3/4] Specific Tournament Pages...")
    specific_pages = [
        ("MPL/Indonesia/Season_14/Playoffs", "MPL ID S14 Playoffs"),
        ("MPL/Indonesia/Season_15", "MPL ID S15"),
        ("MPL/Philippines/Season_14", "MPL PH S14"),
        ("MPL/Malaysia/Season_14", "MPL MY S14"),
        ("ESL/Snapdragon_Pro_Series/2025/Masters", "ESL Masters 2025"),
        ("MSC/2025", "MSC 2025"),
    ]
    
    for page_title, tournament_name in specific_pages:
        if max_requests and request_count >= max_requests:
            break
        
        drafts = scrape_tournament_page(page_title, tournament_name)
        all_drafts.extend(drafts)
        request_count += 1
        
        if drafts:
            print(f"    {tournament_name}: {len(drafts)} drafts")
        
        time.sleep(REQUEST_DELAY)
    
    # Scrape recent/active tournaments
    print("\n[4/4] Recent Tournaments...")
    recent_pages = [
        f"MPL/Indonesia/Season_{s}" for s in range(10, 16)
    ] + [
        f"MPL/Philippines/Season_{s}" for s in range(10, 15)
    ] + [
        f"MPL/Malaysia/Season_{s}" for s in range(10, 15)
    ]
    
    for page_title in recent_pages:
        if max_requests and request_count >= max_requests:
            break
        
        tournament_name = page_title.split("/")[-1].replace("_", " ")
        drafts = scrape_tournament_page(page_title, tournament_name)
        all_drafts.extend(drafts)
        request_count += 1
        
        if drafts:
            print(f"    {tournament_name}: {len(drafts)} drafts")
        
        time.sleep(REQUEST_DELAY)
    
    print(f"\n{'=' * 60}")
    print(f"Total drafts scraped: {len(all_drafts)}")
    print(f"Total API requests: {request_count}")
    
    return all_drafts


def merge_with_existing(new_drafts: List[Dict]) -> List[Dict]:
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


def save_drafts(drafts: List[Dict], output_path: str = None):
    """Save drafts to JSON file."""
    if output_path is None:
        output_path = DATA_DIR / "tournament_drafts.json"
    
    # Create backup if file exists
    if Path(output_path).exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = DATA_DIR / "backups" / f"tournament_drafts_{timestamp}.json"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy2(output_path, backup_path)
        print(f"Backup created: {backup_path}")
    
    # Sort by date
    drafts.sort(key=lambda x: x.get("date", ""), reverse=True)
    
    with open(output_path, "w") as f:
        json.dump(drafts, f, indent=2)
    
    print(f"Saved {len(drafts)} drafts to {output_path}")


def print_stats(drafts: List[Dict]):
    """Print statistics about scraped drafts."""
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
    
    # Count heroes
    hero_counts = {}
    for d in drafts:
        for hero in d.get("blue_picks", []) + d.get("red_picks", []):
            hero_counts[hero] = hero_counts.get(hero, 0) + 1
    
    print(f"\n{'=' * 60}")
    print(f"SCRAPING STATISTICS")
    print(f"{'=' * 60}")
    print(f"Total drafts: {len(drafts)}")
    print(f"\nBy tournament (top 20):")
    for t, count in sorted(tournaments.items(), key=lambda x: -x[1])[:20]:
        print(f"  {t}: {count}")
    
    print(f"\nBy year:")
    for y, count in sorted(years.items()):
        print(f"  {y}: {count}")
    
    print(f"\nTop 20 heroes:")
    for hero, count in sorted(hero_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {hero}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Liquipedia MLBB Tournament Scraper")
    parser.add_argument("--start-year", type=int, default=None,
                        help="Start year (default: auto-detect)")
    parser.add_argument("--end-year", type=int, default=None,
                        help="End year (default: current year)")
    parser.add_argument("--max-requests", type=int, default=None,
                        help="Max API requests (rate limit aware)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't save, just print stats")
    parser.add_argument("--test", action="store_true",
                        help="Test with single tournament")
    
    args = parser.parse_args()
    
    # Use datetime for time-aware decisions
    now = datetime.now()
    print(f"Liquipedia MLBB Scraper")
    print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Rate limit: 1 request per {REQUEST_DELAY}s")
    
    if args.test:
        # Test with single tournament
        print("\n[TEST MODE] Scraping MPL ID S14 Playoffs...")
        drafts = scrape_tournament_page(
            "MPL/Indonesia/Season_14/Playoffs",
            "MPL ID S14 Playoffs"
        )
        print(f"Found {len(drafts)} drafts")
        if drafts:
            print("\nSample draft:")
            print(json.dumps(drafts[0], indent=2))
    else:
        # Full scrape
        all_drafts = scrape_all_tournaments(
            start_year=args.start_year,
            end_year=args.end_year,
            max_requests=args.max_requests
        )
        
        if not args.dry_run and all_drafts:
            # Merge with existing
            merged = merge_with_existing(all_drafts)
            save_drafts(merged, args.output)
            print_stats(merged)
        elif all_drafts:
            print_stats(all_drafts)


if __name__ == "__main__":
    main()
