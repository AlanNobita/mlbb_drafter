#!/usr/bin/env python3
"""Fetch all MLBB data - MERGES with existing data (never overwrites)."""
import json
import os
import urllib.request
import time
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "api_data"
os.makedirs(DATA_DIR, exist_ok=True)


def load_json_safe(path):
    """Load JSON if file exists, else return empty dict/list."""
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_json(path, data):
    """Save JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def fetch_url(url, timeout=30):
    """Fetch URL and return parsed JSON."""
    req = urllib.request.Request(url, headers={"User-Agent": "MLBB-Drafter-Training/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def merge_json(existing, new_data):
    """Merge two JSON structures - new data overwrites same keys, old keys preserved."""
    if isinstance(existing, dict) and isinstance(new_data, dict):
        merged = existing.copy()
        merged.update(new_data)
        return merged
    elif isinstance(existing, list) and isinstance(new_data, list):
        # For lists, replace entirely (new data is complete)
        return new_data
    return new_data


def fetch_with_merge(name, url, filename, is_list=False):
    """Fetch data and merge with existing file."""
    filepath = DATA_DIR / filename
    existing = load_json_safe(filepath)
    
    print(f"Fetching {name}...")
    try:
        new_data = fetch_url(url)
        
        if is_list:
            # For list data (like hero_winrate), replace entirely
            merged = new_data
        else:
            # For dict data, merge
            merged = merge_json(existing, new_data)
        
        save_json(filepath, merged)
        size = os.path.getsize(filepath)
        
        # Count records
        if isinstance(merged, list):
            count = len(merged)
        elif isinstance(merged, dict) and "data" in merged:
            records = merged.get("data", {}).get("records", [])
            count = len(records) if isinstance(records, list) else len(merged)
        else:
            count = len(merged)
        
        print(f"  OK ({size/1024:.1f}KB) - {count} records")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def fetch_counters_for_hero(hero_id, hero_name):
    """Fetch counter data for a single hero."""
    try:
        data = fetch_url(f"https://mlbb.rone.dev/api/heroes/{hero_id}/counters?lang=en")
        return hero_name, data
    except Exception as e:
        return hero_name, None


def main():
    print("=" * 60)
    print("MLBB DATA FETCHER (MERGE MODE)")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Basic data fetches (complete replacement)
    sources = [
        ("MLBB-Winrate", "https://raw.githubusercontent.com/Pren7/MLBB-Winrate/refs/heads/main/winrate.json", "hero_winrate.json", True),
        ("OpenMLBB Heroes", "https://mlbb.rone.dev/api/heroes?size=150&index=1&order=desc&lang=en", "openmlbb_heroes.json", False),
        ("OpenMLBB Rank", "https://mlbb.rone.dev/api/heroes/rank?lang=en", "hero_rank.json", False),
        ("Academy Heroes", "https://mlbb.rone.dev/api/academy/heroes?size=150&index=1&order=desc&lang=en", "academy_heroes.json", False),
        ("Academy Roles", "https://mlbb.rone.dev/api/academy/roles?lang=en", "academy_roles.json", False),
        ("Academy Equipment", "https://mlbb.rone.dev/api/academy/equipment?lang=en", "academy_equipment.json", False),
        ("Academy Spells", "https://mlbb.rone.dev/api/academy/spells?lang=en", "academy_spells.json", False),
        ("Academy Emblems", "https://mlbb.rone.dev/api/academy/emblems?lang=en", "academy_emblems.json", False),
        ("Hero Meta", "https://raw.githubusercontent.com/p3hndrx/MLBB-API/main/v1/hero-meta-final.json", "hero_meta.json", False),
        ("Item Meta", "https://raw.githubusercontent.com/p3hndrx/MLBB-API/main/v1/item-meta-final.json", "item_meta.json", False),
        ("Emblem Meta", "https://raw.githubusercontent.com/p3hndrx/MLBB-API/main/v1/emblem-meta-final.json", "emblem_meta.json", False),
        ("Hero Icons", "https://mapi.mobilelegends.com/api/icon", "hero_icons.json", False),
    ]
    
    results = {}
    for name, url, filename, is_list in sources:
        ok = fetch_with_merge(name, url, filename, is_list)
        results[name] = ok
    
    # Counter data (MERGE - keep existing, add new)
    print("\nFetching hero counter data (MERGE MODE)...")
    existing_counters = load_json_safe(DATA_DIR / "hero_counters_all.json")
    print(f"  Existing: {len(existing_counters)} heroes with counter data")
    
    # Get hero list
    try:
        hero_data = fetch_url("https://mlbb.rone.dev/api/heroes?size=150&index=1&order=desc&lang=en")
        records = hero_data.get("data", {}).get("records", [])
        
        new_fetched = 0
        skipped = 0
        failed = 0
        
        for rec in records:
            hero_id = rec.get("data", {}).get("hero_id")
            hero_name = rec.get("data", {}).get("hero", {}).get("data", {}).get("name", "")
            
            if not hero_name or not hero_id:
                continue
            
            # Skip if already have data
            if hero_name in existing_counters:
                skipped += 1
                continue
            
            # Fetch new data
            name, data = fetch_counters_for_hero(hero_id, hero_name)
            if data:
                existing_counters[name] = data
                new_fetched += 1
            else:
                failed += 1
            
            time.sleep(0.2)  # Rate limit
        
        # Save merged counters
        save_json(DATA_DIR / "hero_counters_all.json", existing_counters)
        print(f"  Total: {len(existing_counters)} heroes")
        print(f"  New: {new_fetched}, Skipped: {skipped}, Failed: {failed}")
        
    except Exception as e:
        print(f"  Error fetching hero list: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("FETCH COMPLETE")
    print("=" * 60)
    
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")
    
    print(f"\nAll data saved to: {DATA_DIR}")
    print(f"Files are MERGED - existing data preserved, new data added/updated.")


if __name__ == "__main__":
    main()
