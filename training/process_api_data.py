#!/usr/bin/env python3
"""Process MLBB API data into training-ready format for GCN model."""
import json
import csv
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "api_data"
OUTPUT_DIR = Path(__file__).parent / "data"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def extract_hero_stats():
    """Extract hero win/pick/ban rates from winrate.json (MLBB-Winrate source)."""
    data = load_json(DATA_DIR / "hero_winrate.json")
    heroes = {}
    for h in data:
        name = h["name"]
        wr = float(h["winrate"].replace("%", "")) / 100
        br = float(h["banrate"].replace("%", "")) / 100
        pr = float(h["pickrate"].replace("%", "")) / 100
        heroes[name] = {
            "win_rate": wr,
            "ban_rate": br,
            "pick_rate": pr,
            "icon": h.get("icon", ""),
        }
    return heroes


def extract_hero_relations():
    """Extract counter/synergy relations from OpenMLBB hero list."""
    data = load_json(DATA_DIR / "openmlbb_heroes.json")
    relations = {}
    for rec in data.get("data", {}).get("records", []):
        hero_data = rec.get("data", {})
        hero_id = hero_data.get("hero_id")
        hero_name = hero_data.get("hero", {}).get("data", {}).get("name", "")
        rel = hero_data.get("relation", {})
        if hero_name:
            relations[hero_name] = {
                "hero_id": hero_id,
                "assist": rel.get("assist", {}).get("target_hero_id", []),
                "strong": rel.get("strong", {}).get("target_hero_id", []),
                "weak": rel.get("weak", {}).get("target_hero_id", []),
            }
    return relations


def extract_counter_data():
    """Extract detailed counter data from hero_counters_all.json."""
    try:
        data = load_json(DATA_DIR / "hero_counters_all.json")
        counters = {}
        for hero_name, counter_data in data.items():
            records = counter_data.get("data", {}).get("records", [])
            if records:
                rec = records[0]
                main_hero = rec.get("data", {}).get("main_hero", {}).get("data", {}).get("name", "")
                sub_heroes = rec.get("data", {}).get("sub_hero", [])
                counters[hero_name] = {
                    "win_rate": rec.get("data", {}).get("main_hero_win_rate", 0),
                    "ban_rate": rec.get("data", {}).get("main_hero_ban_rate", 0),
                    "pick_rate": rec.get("data", {}).get("main_hero_appearance_rate", 0),
                    "synergies": [
                        {
                            "name": sh.get("hero", {}).get("data", {}).get("name", ""),
                            "win_rate": sh.get("hero_win_rate", 0),
                            "increase_win_rate": sh.get("increase_win_rate", 0),
                        }
                        for sh in sub_heroes[:10]  # Top 10 synergies
                    ],
                }
        return counters
    except Exception:
        return {}


def extract_hero_id_map():
    """Map hero_id -> name from OpenMLBB data."""
    data = load_json(DATA_DIR / "openmlbb_heroes.json")
    id_map = {}
    for rec in data.get("data", {}).get("records", []):
        hero_data = rec.get("data", {})
        hero_id = hero_data.get("hero_id")
        hero_name = hero_data.get("hero", {}).get("data", {}).get("name", "")
        if hero_id and hero_name:
            id_map[hero_id] = hero_name
    return id_map


def extract_hero_meta():
    """Extract hero base attributes from p3hndrx data."""
    data = load_json(DATA_DIR / "hero_meta.json")
    heroes_list = data.get("data", []) if isinstance(data, dict) else data
    meta = {}
    if isinstance(heroes_list, list):
        for h in heroes_list:
            name = h.get("hero_name", "")
            if name and name != "None":
                lanes = h.get("laning", [])
                meta[name] = {
                    "role": h.get("class", ""),
                    "lane": ", ".join(lanes) if lanes else "",
                    "mlid": h.get("mlid", ""),
                    "uid": h.get("uid", ""),
                    "portrait": h.get("portrait", ""),
                }
    return meta


def extract_items():
    """Extract item data from p3hndrx."""
    data = load_json(DATA_DIR / "item_meta.json")
    items = data.get("data", data) if isinstance(data, dict) else data
    if isinstance(items, list):
        return {str(i.get("id", i.get("itemid", ""))): i for i in items}
    return items


def extract_emblems():
    """Extract emblem data from p3hndrx."""
    data = load_json(DATA_DIR / "emblem_meta.json")
    return data


def build_training_csv(heroes, relations, hero_meta, id_map):
    """Build a comprehensive CSV for GCN training."""
    rows = []
    for name, stats in heroes.items():
        row = {
            "hero_name": name,
            "win_rate": stats["win_rate"],
            "ban_rate": stats["ban_rate"],
            "pick_rate": stats["pick_rate"],
        }
        # Add relations
        if name in relations:
            rel = relations[name]
            row["hero_id"] = rel["hero_id"]
            row["counters"] = "|".join(
                id_map.get(cid, str(cid))
                for cid in rel["weak"]
                if cid in id_map or cid != 0
            )
            row["synergies"] = "|".join(
                id_map.get(cid, str(cid))
                for cid in rel["assist"]
                if cid in id_map or cid != 0
            )
            row["strong_against"] = "|".join(
                id_map.get(cid, str(cid))
                for cid in rel["strong"]
                if cid in id_map or cid != 0
            )
        # Add meta
        if name in hero_meta:
            m = hero_meta[name]
            row["role"] = m.get("role", "")
            row["lane"] = m.get("lane", "")
            row["specialty"] = m.get("specialty", "")
        rows.append(row)
    return rows


def build_adjacency_matrix(heroes, relations, id_map):
    """Build adjacency matrix for GCN from counter/synergy data."""
    hero_names = sorted(heroes.keys())
    name_to_idx = {name: i for i, name in enumerate(hero_names)}
    n = len(hero_names)

    # Initialize adjacency matrix
    adj = [[0.0] * n for _ in range(n)]

    for name, rel in relations.items():
        if name not in name_to_idx:
            continue
        i = name_to_idx[name]
        # Synergies (positive edges)
        for cid in rel.get("assist", []):
            cname = id_map.get(cid)
            if cname and cname in name_to_idx:
                j = name_to_idx[cname]
                adj[i][j] = 1.0
                adj[j][i] = 1.0
        # Strong against (positive edges)
        for cid in rel.get("strong", []):
            cname = id_map.get(cid)
            if cname and cname in name_to_idx:
                j = name_to_idx[cname]
                adj[i][j] = 0.5
        # Weak against (negative edges)
        for cid in rel.get("weak", []):
            cname = id_map.get(cid)
            if cname and cname in name_to_idx:
                j = name_to_idx[cname]
                adj[i][j] = -0.5

    return hero_names, adj


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    heroes = extract_hero_stats()
    print(f"  Winrate heroes: {len(heroes)}")

    relations = extract_hero_relations()
    print(f"  Heroes with relations: {len(relations)}")

    id_map = extract_hero_id_map()
    print(f"  Hero ID map: {len(id_map)} entries")

    hero_meta = extract_hero_meta()
    print(f"  Hero meta: {len(hero_meta)} entries")

    # Build CSV
    print("\nBuilding training CSV...")
    rows = build_training_csv(heroes, relations, hero_meta, id_map)
    csv_path = OUTPUT_DIR / "hero_stats_real.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} heroes to {csv_path}")

    # Build adjacency matrix
    print("\nBuilding adjacency matrix...")
    hero_names, adj = build_adjacency_matrix(heroes, relations, id_map)
    adj_path = OUTPUT_DIR / "adjacency_real.json"
    with open(adj_path, "w") as f:
        json.dump({"hero_names": hero_names, "adjacency": adj}, f)
    print(f"  Saved {len(hero_names)}x{len(hero_names)} matrix to {adj_path}")

    # Save consolidated JSON
    print("\nBuilding consolidated JSON...")
    consolidated = {
        "heroes": heroes,
        "relations": relations,
        "id_map": id_map,
        "hero_meta": hero_meta,
    }
    json_path = OUTPUT_DIR / "mlbb_data_consolidated.json"
    with open(json_path, "w") as f:
        json.dump(consolidated, f, indent=2)
    print(f"  Saved consolidated data to {json_path}")

    print("\nDone! Files saved to training/data/")
    print(f"  - {csv_path.name} (hero stats CSV)")
    print(f"  - {adj_path.name} (adjacency matrix)")
    print(f"  - {json_path.name} (consolidated JSON)")


if __name__ == "__main__":
    main()
