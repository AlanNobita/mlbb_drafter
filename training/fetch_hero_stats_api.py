#!/usr/bin/env python3
"""
Fetch comprehensive MLBB hero data from multiple APIs:
- OpenMLBB API: hero stats, counters, synergies
- Pren7/MLBB-Winrate: daily winrate data
- mlbbhub.com: counter matchups
"""

import json
import time
import urllib.request
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("training/data")
API_DIR = DATA_DIR / "api_data"
API_DIR.mkdir(parents=True, exist_ok=True)


def fetch_json(url, timeout=30):
    """Fetch JSON from URL."""
    req = urllib.request.Request(url, headers={"User-Agent": "MLBBDrafter/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_openmlbb_hero_stats():
    """Fetch hero statistics from OpenMLBB API (all 132 heroes, paginated)."""
    print("\n[1/4] Fetching hero statistics from OpenMLBB...")
    
    hero_stats = {}
    page = 1
    
    while True:
        url = f"https://mlbb.rone.dev/api/heroes/rank?lang=en&size=20&index={page}&order=desc"
        data = fetch_json(url)
        records = data.get("data", {}).get("records", [])
        
        if not records:
            break
        
        for rec in records:
            d = rec.get("data", {})
            hero_id = d.get("main_heroid")
            hero_name = d.get("main_hero", {}).get("data", {}).get("name", "")
            
            if hero_id and hero_name:
                hero_stats[hero_id] = {
                    "name": hero_name,
                    "win_rate": d.get("main_hero_win_rate", 0),
                    "pick_rate": d.get("main_hero_appearance_rate", 0),
                    "ban_rate": d.get("main_hero_ban_rate", 0),
                    "synergies": []
                }
                
                # Extract synergy data
                for sub in d.get("sub_hero", []):
                    synergy_hero_id = sub.get("heroid")
                    increase_wr = sub.get("increase_win_rate", 0)
                    if synergy_hero_id:
                        hero_stats[hero_id]["synergies"].append({
                            "hero_id": synergy_hero_id,
                            "increase_win_rate": increase_wr
                        })
        
        print(f"  Page {page}: {len(records)} heroes")
        page += 1
        time.sleep(0.5)
    
    print(f"  Got {len(hero_stats)} heroes with stats")
    return hero_stats


def fetch_openmlbb_counters():
    """Fetch counter data for all heroes from OpenMLBB API."""
    print("\n[2/4] Fetching counter data from OpenMLBB...")
    
    # First get hero list to build ID->name mapping
    url = "https://mlbb.rone.dev/api/heroes?size=150&index=1&order=desc&lang=en"
    data = fetch_json(url)
    heroes = data.get("data", {}).get("records", [])
    
    # Build ID->name mapping (counter API only returns hero IDs, not names)
    id_to_name = {}
    hero_ids_to_fetch = []
    for hero_rec in heroes:
        hero_info = hero_rec.get("data", {})
        hero_id = hero_info.get("hero_id")
        hero_name = hero_info.get("hero", {}).get("data", {}).get("name", "")
        if hero_id and hero_name:
            id_to_name[hero_id] = hero_name
            hero_ids_to_fetch.append((hero_id, hero_name))
    
    print(f"  Built ID->name mapping for {len(id_to_name)} heroes")
    
    counter_data = {}
    fetched = 0
    
    for hero_id, hero_name in hero_ids_to_fetch:
        # Fetch counter data
        try:
            counter_url = f"https://mlbb.rone.dev/api/heroes/{hero_id}/counters?lang=en"
            counter_resp = fetch_json(counter_url)
            counter_records = counter_resp.get("data", {}).get("records", [])
            
            if counter_records:
                counter_data[hero_name] = {
                    "hero_id": hero_id,
                    "counters": []
                }
                
                for rec in counter_records:
                    rec_data = rec.get("data", {})
                    for sub in rec_data.get("sub_hero", []):
                        counter_hero_id = sub.get("heroid")
                        win_rate = sub.get("hero_win_rate", 0)
                        appearance_rate = sub.get("hero_appearance_rate", 0)
                        
                        # Look up name from ID mapping
                        counter_name = id_to_name.get(counter_hero_id, "")
                        
                        if counter_hero_id and counter_name:
                            counter_data[hero_name]["counters"].append({
                                "hero_id": counter_hero_id,
                                "name": counter_name,
                                "win_rate": win_rate,
                                "appearance_rate": appearance_rate
                            })
                
                fetched += 1
                if fetched % 20 == 0:
                    print(f"  Fetched {fetched} heroes...")
            
            time.sleep(0.5)  # Rate limit
            
        except Exception as e:
            print(f"  Error fetching counters for {hero_name}: {e}")
    
    print(f"  Got counter data for {len(counter_data)} heroes")
    return counter_data


def fetch_pren7_winrate():
    """Fetch daily winrate data from Pren7/MLBB-Winrate."""
    print("\n[3/4] Fetching daily winrate from Pren7/MLBB-Winrate...")
    
    url = "https://raw.githubusercontent.com/Pren7/MLBB-Winrate/refs/heads/main/winrate.json"
    try:
        data = fetch_json(url)
        print(f"  Got {len(data)} heroes with daily winrate")
        return data
    except Exception as e:
        print(f"  Error: {e}")
        return []


def save_data(name, data):
    """Save data to JSON file."""
    filepath = API_DIR / f"{name}.json"
    
    # Load existing if available
    if filepath.exists():
        with open(filepath) as f:
            existing = json.load(f)
        # Merge
        if isinstance(existing, dict) and isinstance(data, dict):
            existing.update(data)
            data = existing
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    
    size = filepath.stat().st_size / 1024
    print(f"  Saved {name}.json ({size:.1f}KB)")
    return filepath


def main():
    print("=" * 60)
    print("MLBB COMPREHENSIVE DATA FETCHER")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Fetch all data
    hero_stats = fetch_openmlbb_hero_stats()
    counter_data = fetch_openmlbb_counters()
    winrate_data = fetch_pren7_winrate()
    
    # Save data
    print("\n[4/4] Saving data...")
    save_data("hero_stats_openmlbb", hero_stats)
    save_data("hero_counters_openmlbb", counter_data)
    save_data("hero_winrate_daily", winrate_data)
    
    # Summary
    print("\n" + "=" * 60)
    print("FETCH COMPLETE")
    print("=" * 60)
    print(f"  Hero stats: {len(hero_stats)} heroes")
    print(f"  Counter data: {len(counter_data)} heroes")
    print(f"  Daily winrate: {len(winrate_data)} heroes")
    print(f"\nData saved to: {API_DIR}")


if __name__ == "__main__":
    main()
