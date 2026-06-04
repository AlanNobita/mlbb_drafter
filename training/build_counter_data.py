#!/usr/bin/env python3
"""
Build corrected counter data by combining API data with community knowledge.
Community overrides fix known incorrect matchups from the API.
"""
import json

# Load all data sources
with open('training/data/api_data/hero_counters_openmlbb.json') as f:
    api_counters = json.load(f)

with open('training/data/api_data/hero_counters_simple.json') as f:
    simple_counters = json.load(f)

# Community counter overrides (known correct matchups)
# Format: {hero_beating: {hero_losing: win_rate}}
COMMUNITY_COUNTERS = {
    # Vexana beats Gloo (known from gameplay)
    "Vexana": {"Gloo": 0.55},
    # Faramis beats Gloo
    "Faramis": {"Gloo": 0.53},
    # Kalea beats Gloo
    "Kalea": {"Gloo": 0.52},
    # Add more community knowledge here...
}

def build_corrected_counters():
    """Build corrected counter data with community overrides."""
    corrected = {}
    
    # Start with simple_counters structure
    for hero_name, data in simple_counters.items():
        corrected[hero_name] = {
            "main_hero": hero_name,
            "main_hero_win_rate": data.get("main_hero_win_rate", 0.5),
            "counters": [],  # Heroes that beat this hero
            "beats": []      # Heroes this hero beats
        }
        
        # Add API counter data (heroes that beat this hero)
        for matchup in data.get("matchups", []):
            corrected[hero_name]["counters"].append({
                "hero": matchup["hero"],
                "win_rate": matchup["win_rate"],
                "increase": matchup.get("increase", 0)
            })
    
    # Apply community overrides
    for hero_beating, matchups in COMMUNITY_COUNTERS.items():
        for hero_losing, win_rate in matchups.items():
            # Add to hero_beating's "beats" list
            if hero_beating in corrected:
                # Check if already exists
                exists = any(m["hero"] == hero_losing for m in corrected[hero_beating]["beats"])
                if not exists:
                    corrected[hero_beating]["beats"].append({
                        "hero": hero_losing,
                        "win_rate": win_rate
                    })
            
            # Add to hero_losing's "counters" list
            if hero_losing in corrected:
                # Check if already exists
                exists = any(m["hero"] == hero_beating for m in corrected[hero_losing]["counters"])
                if not exists:
                    corrected[hero_losing]["counters"].append({
                        "hero": hero_beating,
                        "win_rate": win_rate
                    })
    
    return corrected


def build_counter_boost_map(corrected_counters, hero_name_to_id=None):
    """Build counter boost map for the recommend() function.
    
    Returns dict: hero_name -> {counter_of: win_rate}
    This tells us: "hero X beats these heroes with Y win rate"
    """
    boost_map = {}
    
    for hero_name, data in corrected_counters.items():
        # This hero beats the heroes in its "beats" list
        boost_map[hero_name] = {}
        for beat in data.get("beats", []):
            boost_map[hero_name][beat["hero"]] = beat["win_rate"]
    
    return boost_map


if __name__ == "__main__":
    # Build corrected counters
    corrected = build_corrected_counters()
    
    # Save corrected data
    with open('training/data/api_data/hero_counters_corrected.json', 'w') as f:
        json.dump(corrected, f, indent=2)
    
    # Build and save counter boost map
    boost_map = build_counter_boost_map(corrected)
    with open('training/data/api_data/counter_boost_map.json', 'w') as f:
        json.dump(boost_map, f, indent=2)
    
    print(f"Saved {len(corrected)} heroes to hero_counters_corrected.json")
    print(f"Saved counter boost map to counter_boost_map.json")
    print()
    
    # Show examples
    print("=== CORRECTED COUNTER DATA ===")
    print()
    
    # Show Vexana
    if "Vexana" in corrected:
        print("Vexana:")
        beats = [f"{m['hero']} ({m['win_rate']*100:.1f}%)" for m in corrected['Vexana']['beats']]
        counters = [f"{m['hero']} ({m['win_rate']*100:.1f}%)" for m in corrected['Vexana']['counters'][:3]]
        print(f"  Beats: {beats}")
        print(f"  Countered by: {counters}")
    
    print()
    # Show Gloo
    if "Gloo" in corrected:
        print("Gloo:")
        beats = [f"{m['hero']} ({m['win_rate']*100:.1f}%)" for m in corrected['Gloo']['beats']]
        counters = [f"{m['hero']} ({m['win_rate']*100:.1f}%)" for m in corrected['Gloo']['counters'][:3]]
        print(f"  Beats: {beats}")
        print(f"  Countered by: {counters}")
    
    print()
    # Show counter boost map for heroes that beat Gloo
    print("Heroes that beat Gloo (from boost map):")
    for hero, matchups in boost_map.items():
        if "Gloo" in matchups:
            print(f"  {hero}: {matchups['Gloo']*100:.1f}% WR vs Gloo")
