# MLBB Training Data Sources & Freshness Report

**Generated:** 2026-06-03  
**Current MLBB Patch:** 2.1.70 (as of June 2026)

---

## Data Files in `training/data/`

### Training-Ready Files (Used by train_gcn_enhanced.py)
| File | Size | Description | Used in Training |
|------|------|-------------|------------------|
| `gcn_model_v2.pt` | - | Trained GCN model (132 heroes) | - |
| `tournament_drafts.json` | 7.4MB | 13,171 real tournament matches | ✅ Yes |
| `adjacency_real.json` | 87KB | 132x132 hero relationship matrix | ✅ Yes |
| `hero_stats_temporal.csv` | 14KB | 122 heroes with temporal features | ✅ Yes |
| `hero_cooccurrence_tournament.json` | 960KB | Synergy stats (122 heroes) | ✅ Yes |
| `synergy_matrix_tournament.json` | 216KB | Tournament synergy matrix | ✅ Yes |
| `era_hero_stats.json` | 225KB | Per-patch hero stats (31 eras) | ❌ Available |
| `hero_temporal_stats.json` | 23KB | Time-decayed hero stats | ❌ Available |
| `patch_timeline.json` | 7.8KB | Patch date ranges (54 patches) | ❌ Available |
| `tournament_training.csv` | - | Tournament drafts as CSV | ❌ Alternative |
| `synthetic_drafts.csv` | 791KB | Original synthetic training data | ❌ Deprecated |
| `MLBB_TRAINING_DATA_FINAL.json` | 9.2MB | All data consolidated | ❌ Reference |

### Raw API Data in `training/data/api_data/`
| File | Size | Source | Freshness |
|------|------|--------|-----------|
| `hero_winrate.json` | 29KB | Pren7/MLBB-Winrate | **Daily** (cron job) |
| `openmlbb_heroes.json` | 119KB | OpenMLBB API | **Live API** (v4.0.9) |
| `hero_rank.json` | 50KB | OpenMLBB API | **Live API** |
| `hero_counters_all.json` | 1.1MB | OpenMLBB API | **Live API** (132 heroes) |
| `hero_meta.json` | 749KB | p3hndrx/MLBB-API | **Weekly** (manual) |
| `item_meta.json` | 135KB | p3hndrx/MLBB-API | **Weekly** (manual) |
| `emblem_meta.json` | 5.9KB | p3hndrx/MLBB-API | **Weekly** (manual) |
| `hero_icons.json` | 181KB | mapi.mobilelegends.com | **Semi-static** |
| `academy_heroes.json` | 37KB | OpenMLBB API | **Live API** |
| `academy_roles.json` | 9.6KB | OpenMLBB API | **Live API** |
| `academy_equipment.json` | 8.2KB | OpenMLBB API | **Live API** |
| `academy_spells.json` | 15KB | OpenMLBB API | **Live API** |
| `academy_emblems.json` | 20KB | OpenMLBB API | **Live API** |

---

## Training Commands

### Main Training (Recommended)
```bash
python3 training/train_gcn_enhanced.py --epochs 100
```

**Uses ALL data sources:**
- Tournament drafts (13,171 matches)
- Hero stats (win/pick/ban rates, 132 heroes)
- Temporal features (recency, meta score, trend)
- Adjacency matrix (132x132, 11,380 edges)
- Co-occurrence stats (synergy data)
- Synergy matrix (tournament scores)

### Fetch Latest Data
```bash
# Fetch from all APIs (merge mode - never overwrites)
python3 training/fetch_all_data.py

# Process tournament data
python3 training/process_drive_data.py

# Add temporal features
python3 training/add_temporal_features.py
```

### YOLO Training (Detection)
```bash
python3 training/train_yolo.py --data training/dataset.yaml --epochs 100
```

---

## Data Sources & Where to Get Latest Data

### 1. Pren7/MLBB-Winrate (BEST for daily stats)
- **URL:** `https://raw.githubusercontent.com/Pren7/MLBB-Winrate/refs/heads/main/winrate.json`
- **Data:** Hero win/ban/pick rates (All Ranks, Past 1 Day)
- **Update:** **Daily** via GitHub Actions cron job (00:00 WIB / 17:00 UTC)
- **Freshness:** VERY FRESH
- **How to update:** Just re-fetch the URL

### 2. OpenMLBB API (BEST for comprehensive data)
- **URL:** `https://mlbb.rone.dev/api/`
- **SDK:** `pip install OpenMLBB`
- **GitHub:** `https://github.com/ridwaanhall/api-mobilelegends`
- **Data:** 
  - Hero list with stats
  - Counter matchups
  - Synergy data
  - Hero relations (assist/strong/weak)
  - Equipment, emblems, spells
  - Hero rankings
- **Update:** **Actively maintained** (v4.0.9, May 2026)
- **Rate limit:** 500 req/day (standard), 500+ req/day (fastapicloud)

### 3. p3hndrx/MLBB-API (BEST for hero metadata)
- **GitHub:** `https://github.com/p3hndrx/MLBB-API`
- **Data:** Hero base stats, roles, lanes, skills, items, emblems
- **Update:** **Weekly** (manual updates)

### 4. Google Drive Tournament Data
- **File:** `drive-download-20260603T085330Z-3-001.zip`
- **Data:** 13,171 tournament matches (2017-2023)
- **Coverage:** M1-M5, MPL PH, MPL ID, MSC
- **Note:** Source files are never modified by scripts

---

## Data Coverage

| Metric | Coverage |
|--------|----------|
| Heroes with win/ban/pick rates | 132/132 (100%) |
| Heroes with counter data | 132/132 (100%) |
| Heroes with synergy data | 132/132 (100%) |
| Heroes with role/lane data | 132/132 (100%) |
| Heroes with temporal features | 122/132 (92%) |
| Tournament matches | 13,171 |
| Patch eras | 31 |
| Patches tracked | 54 |

---

## Missing Data

### 10 Heroes Without Temporal Features
These newer heroes lack tournament data:
- Chip, Cici, Kalea, Lukas, Marcel, Obsidia, Sora, Suyou, Zetian, Zhuxin

### 924 Drafts Skipped
Drafts with unmapped heroes (newer heroes not in tournament data yet).

---

## How Data Flows to Training

```
APIs → fetch_all_data.py → api_data/
Google Drive → process_drive_data.py → tournament_drafts.json
                    ↓
add_temporal_features.py → hero_stats_temporal.csv
                    ↓
train_gcn_enhanced.py → gcn_model_v2.pt
```

---

## Backup System

Before any data update, run:
```bash
python3 training/backup_data.py
```

Backups are saved to: `training/backups/YYYYMMDD_HHMMSS/`
