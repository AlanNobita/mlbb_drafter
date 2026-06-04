#!/usr/bin/env python3
"""Add lane assignments to hero_meta.json based on MLBB meta."""
import json

# Lane assignments based on MLBB meta (2025-2026)
# Format: hero_name -> [primary_lane, secondary_lanes]
# Lanes: EXP, Gold, Mid, Jungle, Roam

LANE_DATA = {
    # === TANKS ===
    "Akai":          ["Jungle", "EXP", "Roam"],
    "Atlas":         ["Roam"],
    "Belerick":      ["Roam", "EXP"],
    "Baxia":         ["Jungle", "Roam"],
    "Carmilla":      ["Roam"],
    "Cupid":         ["Roam"],
    "Franco":        ["Roam"],
    "Gatotkaca":     ["EXP", "Roam"],
    "Gloo":          ["EXP", "Roam"],
    "Grohk":         ["Roam"],
    "Hylos":         ["EXP", "Roam"],
    "Khufra":        ["Roam"],
    "Johnson":       ["Roam", "Jungle"],
    "Minotaur":      ["Roam"],
    "Phoveus":       ["EXP"],
    "Tigreal":       ["Roam"],
    "Uranus":        ["EXP", "Roam"],
    "Khaleed":       ["EXP"],

    # === FIGHTERS ===
    "Alpha":         ["Jungle", "EXP"],
    "Aldous":        ["EXP", "Jungle"],
    "Arlott":        ["EXP", "Jungle"],
    "Badang":        ["EXP"],
    "Balmond":       ["Jungle", "EXP"],
    "Chou":          ["EXP", "Roam"],
    "Cici":          ["EXP"],
    "Dyrroth":       ["EXP", "Jungle"],
    "Esmeralda":     ["EXP"],
    "Freya":         ["EXP", "Jungle", "gold"],
    "Guinevere":     ["EXP", "Mid"],
    "Hilda":         ["EXP", "Roam"],
    "Jawhead":       ["EXP", "Roam"],
    "Lapu-Lapu":     ["EXP"],
    "Leomord":       ["Jungle", "EXP"],
    "Masha":         ["EXP"],
    "Martis":        ["Jungle", "EXP"],
    "Minsitthar":    ["EXP"],
    "Paquito":       ["EXP", "Jungle"],
    "Terizla":       ["EXP"],
    "Thamuz":        ["EXP"],
    "X.Borg":        ["EXP"],
    "Yu Zhong":      ["EXP"],
    "Zilong":        ["EXP", "Jungle"],
    "Sun":           ["EXP", "Jungle"],
    "Yin":           ["EXP"],
    "Lukas":         ["EXP"],

    # === ASSASSINS ===
    "Aamon":         ["Jungle"],
    "Benedetta":     ["EXP", "Jungle"],
    "Fanny":         ["Jungle"],
    "Gusion":        ["Jungle"],
    "Hayabusa":      ["Jungle"],
    "Helcurt":       ["Jungle"],
    "Hanzo":         ["Jungle"],
    "Karina":        ["Jungle"],
    "Lancelot":      ["Jungle"],
    "Ling":          ["Jungle"],
    "Nolan":         ["Jungle"],
    "Saber":         ["Jungle"],
    "Selena":        ["Mid", "Roam"],
    "Natalia":       ["Roam", "Jungle"],
    "Joy":           ["Jungle"],
    "Lancelot":      ["Jungle"],
    "Yi Sun-shin":   ["Jungle"],
    "Suyou":         ["Jungle"],
    "Sora":          ["EXP", "Jungle"],

    # === MAGES ===
    "Alice":         ["EXP", "Mid"],
    "Aurora":        ["Mid"],
    "Cecilion":      ["Mid"],
    "Chang'e":       ["Mid"],
    "Cyclops":       ["Mid", "Jungle"],
    "Eudora":        ["Mid"],
    "Faramis":       ["Mid", "Roam"],
    "Gord":          ["Mid"],
    "Harith":        ["Mid", "Jungle"],
    "Kagura":        ["Mid"],
    "Kadita":        ["Mid", "Roam"],
    "Kimmy":         ["Mid", "Gold"],
    "Luo Yi":        ["Mid"],
    "Lylia":         ["Mid"],
    "Mathilda":      ["Roam", "Mid"],
    "Nana":          ["Mid"],
    "Novaria":       ["Mid"],
    "Odette":        ["Mid"],
    "Pharsa":        ["Mid"],
    "Valentina":     ["Mid"],
    "Vexana":        ["Mid"],
    "Xavier":        ["Mid"],
    "Yve":           ["Mid"],
    "Zhask":         ["Mid"],
    "Melissa":       ["Mid"],
    "Zhuxin":        ["Mid"],
    "Zetian":        ["Mid"],
    "Kalea":         ["Roam", "Mid"],

    # === MARKSMEN ===
    "Beatrix":       ["Gold"],
    "Brody":         ["Gold"],
    "Clint":         ["Gold"],
    "Claude":        ["Gold", "Jungle"],
    "Hanabi":        ["Gold"],
    "Irithel":       ["Gold"],
    "Karrie":        ["Gold"],
    "Lesley":        ["Gold"],
    "Layla":         ["Gold"],
    "Moskov":        ["Gold", "Jungle"],
    "Miya":          ["Gold"],
    "Natan":         ["Gold"],
    "Popol and Kupa": ["Gold", "Jungle"],
    "Roger":         ["Jungle", "Gold"],
    "Wanwan":        ["Gold"],
    "Wesker":        ["Gold"],
    "Ixia":          ["Gold"],
    "Melissa":       ["Gold"],
    "Obsidia":       ["Gold"],
    "Bane":          ["Gold", "Mid"],

    # === SUPPORTS ===
    "Angela":        ["Roam"],
    "Diggie":        ["Roam"],
    "Estes":         ["Roam"],
    "Floryn":        ["Roam"],
    "Lolita":        ["Roam"],
    "Natalia":       ["Roam"],
    "Rafaela":       ["Roam"],
    "Renevent":      ["Roam"],
    "Marcel":        ["Roam"],
}

def update_hero_meta():
    with open('shared/hero_meta.json') as f:
        meta = json.load(f)

    updated = 0
    missing = []
    
    for hero in meta['heroes']:
        name = hero['name']
        if name in LANE_DATA:
            lanes = LANE_DATA[name]
            hero['lanes'] = lanes
            hero['primary_lane'] = lanes[0]
            updated += 1
        else:
            missing.append(name)
            # Default based on role
            role = hero.get('role', '')
            if role == 'Tank':
                hero['lanes'] = ['Roam']
                hero['primary_lane'] = 'Roam'
            elif role == 'Fighter':
                hero['lanes'] = ['EXP']
                hero['primary_lane'] = 'EXP'
            elif role == 'Assassin':
                hero['lanes'] = ['Jungle']
                hero['primary_lane'] = 'Jungle'
            elif role == 'Mage':
                hero['lanes'] = ['Mid']
                hero['primary_lane'] = 'Mid'
            elif role == 'Marksman':
                hero['lanes'] = ['Gold']
                hero['primary_lane'] = 'Gold'
            elif role == 'Support':
                hero['lanes'] = ['Roam']
                hero['primary_lane'] = 'Roam'
            else:
                hero['lanes'] = []
                hero['primary_lane'] = ''
            updated += 1

    # Save updated meta
    with open('shared/hero_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'Updated {updated}/{len(meta["heroes"])} heroes')
    if missing:
        print(f'Missing from LANE_DATA (used role defaults): {missing}')

    # Print lane distribution
    lane_counts = {}
    for hero in meta['heroes']:
        primary = hero.get('primary_lane', '')
        if primary:
            lane_counts[primary] = lane_counts.get(primary, 0) + 1
    
    print('\nLane distribution:')
    for lane, count in sorted(lane_counts.items()):
        print(f'  {lane}: {count}')

if __name__ == '__main__':
    update_hero_meta()
