# MLBB Training Data Sources & Freshness Report

**Generated:** 2026-06-04  
**Current MLBB Patch:** 2.1.70 (as of June 2026)

---

## Data Files in `training/data/`

### Training-Ready Files (Used by train_gcn_enhanced.py)
| File | Size | Description | Used in Training |
|------|------|-------------|------------------|
| `gcn_model_v2.pt` | - | Trained GCN model (132 heroes, 18.7K params) | - |
| `tournament_drafts.json` | 904KB | 37,373 tournament + ranked matches | ✅ Yes |
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
| `kaggle_ranked_drafts.json` | - | Kaggle ranked draft data | ✅ Yes (merged) |

### Raw API Data in `training/data/api_data/`
| File | Size | Source | Freshness |
|------|------|--------|-----------|
| `hero_winrate.json` | 29KB | Pren7/MLBB-Winrate | **Daily** (cron job) |
| `hero_winrate_daily.json` | 29KB | Pren7/MLBB-Winrate | **Daily** (cron job) |
| `openmlbb_heroes.json` | 119KB | OpenMLBB API | **Live API** (v4.0.9) |
| `hero_rank.json` | 50KB | OpenMLBB API | **Live API** |
| `hero_stats_openmlbb.json` | 81KB | OpenMLBB API | **Live API** (132 heroes) |
| `hero_counters_all.json` | 1.1MB | OpenMLBB API | **Live API** (132 heroes) |
| `hero_counters_openmlbb.json` | 95KB | OpenMLBB API | **Live API** (132 heroes, 264 entries) |
| `hero_counters_top20.json` | 180KB | OpenMLBB API | **Live API** |
| `hero_meta.json` | 749KB | p3hndrx/MLBB-API | **Weekly** (manual) |
| `item_meta.json` | 135KB | p3hndrx/MLBB-API | **Weekly** (manual) |
| `emblem_meta.json` | 5.9KB | p3hndrx/MLBB-API | **Weekly** (manual) |
| `hero_icons.json` | 181KB | mapi.mobilelegends.com | **Semi-static** |
| `academy_heroes.json` | 37KB | OpenMLBB Academy | **Live API** |
| `academy_roles.json` | 9.6KB | OpenMLBB Academy | **Live API** |
| `academy_equipment.json` | 8.2KB | OpenMLBB Academy | **Live API** |
| `academy_spells.json` | 15KB | OpenMLBB Academy | **Live API** |
| `academy_emblems.json` | 20KB | OpenMLBB Academy | **Live API** |

### Kaggle Raw Data in `training/data/kaggle_raw/`
| Directory | Content | Count |
|-----------|---------|-------|
| `mobile-legends-match-results/` | Ranked matches (patch 1.7.58/1.7.68) | 10,664 drafts |
| `mobile-legend-m5-world-knockout-stage-results/` | M5 World Championship | 12 drafts |
| `m7-world-championship-2026-mlbb-result-stats/` | M7 World Championship | 61 matches (series-level) |
| `mlbb-draft-breakdown-patch-1768/` | Draft breakdown (same as above) | - |

---

## Training Commands

### Main Training (Recommended)
```bash
# Standard training
python3 training/train_gcn_enhanced.py --epochs 100

# Recommended training with temporal weighting
python3 training/train_gcn_enhanced.py \
  --epochs 500 \
  --lr 0.00005 \
  --hidden-dim 256 \
  --batch-size 64 \
  --min-year 2021 \
  --temporal-decay 0.5
```

**Uses ALL data sources:**
- Tournament drafts (37,373 matches)
- Hero stats (win/pick/ban rates, 132 heroes)
- Temporal features (recency, meta score, trend)
- Adjacency matrix (132x132, 11,380 edges)
- Co-occurrence stats (synergy data)
- Synergy matrix (tournament scores)
- Counter-pick data (264 counter relationships)

### Fetch Latest Data
```bash
# Fetch from all APIs (merge mode - never overwrites)
python3 training/fetch_all_data.py

# Fetch hero stats from OpenMLBB
python3 training/fetch_hero_stats_api.py

# Fetch from mlbb.io (requires API key from parse.bot)
python3 training/fetch_mlbb_io.py --api-key YOUR_KEY

# Download Kaggle datasets
python3 training/fetch_kaggle_datasets.py --download-all

# Process tournament data
python3 training/process_drive_data.py

# Add temporal features
python3 training/add_temporal_features.py

# Create consolidated training file
python3 training/create_final_training_data.py
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

### 4. mlbb.io Parse API (BEST for rank-specific stats)
- **URL:** `https://parse.bot/scraper/574cbcf8-811b-4ed8-8128-c5e9a39efcc8`
- **Data:** Hero statistics by rank (All/Mythic/Legend/Epic), tier lists
- **Update:** **Real-time**
- **API Key:** Free at parse.bot (100 credits/month, 5 req/min)
- **How to fetch:** `python3 training/fetch_mlbb_io.py --api-key YOUR_KEY`

### 5. Google Drive Tournament Data
- **File:** `drive-download-20260603T085330Z-3-001.zip`
- **Data:** 13,171 tournament matches (2017-2023)
- **Coverage:** M1-M5, MPL PH, MPL ID, MSC
- **Note:** Source files are never modified by scripts

### 6. Kaggle Datasets
- **MLBB Match Results:** `rizqinur/mobile-legends-match-results` (10,664 ranked matches)
- **M5 World:** `bcakra/mobile-legend-m5-world-knockout-stage-results` (12 drafts)
- **M7 World:** `double0x2/m7-world-championship-2026-mlbb-result-stats` (61 matches, series-level)
- **How to fetch:** `python3 training/fetch_kaggle_datasets.py --download-all`
- **Requires:** Kaggle API authentication

### 7. Liquipedia API
- **URL:** `https://liquipedia.net/mobilelegends/api.php`
- **Data:** Tournament drafts (MPL, M-series, MSC)
- **Rate limit:** Strict (1 req/2s, IP-level blocks)
- **How to fetch:** `python3 training/fetch_liquipedia.py`
- **Note:** IP may be rate-limited from earlier scraping

---

## Data Coverage

| Metric | Coverage |
|--------|----------|
| Heroes with win/ban/pick rates | 132/132 (100%) |
| Heroes with counter data | 132/132 (100%) |
| Heroes with synergy data | 132/132 (100%) |
| Heroes with role/lane data | 132/132 (100%) |
| Heroes with temporal features | 122/132 (92%) |
| Tournament matches | 37,373 |
| Patch eras | 31 |
| Patches tracked | 54 |
| Counter relationships | 264 |
| Drafts with bans | 24,348 |
| Drafts with dates | 27,242 |

### Drafts by Year
| Year | Count |
|------|-------|
| 2017 | 70 |
| 2018 | 558 |
| 2019 | 1,359 |
| 2020 | 1,814 |
| 2021 | 4,489 |
| 2022 | 6,598 |
| 2023 | 11,565 |
| 2024 | 777 |

### Drafts by Source
| Source | Count | Type |
|--------|-------|------|
| Tournament (MPL, M-series) | 27,242 | Professional matches |
| Kaggle Ranked | 10,131 | Ranked matches (2023) |
| M5 World Championship | 12 | Tournament |
| **Total** | **37,373** | |

---

## Missing Data

### 10 Heroes Without Temporal Features
These newer heroes lack tournament data:
- Chip, Cici, Kalea, Lukas, Marcel, Obsidia, Sora, Suyou, Zetian, Zhuxin

### Year Coverage Gaps
- **2025-2026:** Zero drafts (need Liquipedia when rate limit resets)
- **M7 World Championship:** Series-level only, no per-game drafts

### mlbb.io Ranked Stats
- **Status:** Needs API key (free at parse.bot)
- **Data:** Rank-specific hero win rates (Mythic/Legend/Epic)
- **How to get:** `python3 training/fetch_mlbb_io.py --api-key YOUR_KEY`

---

## How Data Flows to Training

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources                              │
├─────────────────────────────────────────────────────────────┤
│  Google Drive → process_drive_data.py → tournament_drafts.json │
│  Kaggle → fetch_kaggle_datasets.py → tournament_drafts.json    │
│  OpenMLBB → fetch_hero_stats_api.py → hero_stats_openmlbb.json│
│  Pren7 → fetch_all_data.py → hero_winrate.json                │
│  Liquipedia → fetch_liquipedia.py → tournament_drafts.json     │
│  mlbb.io → fetch_mlbb_io.py → hero_stats_mlbb_io.json         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Processing                                │
├─────────────────────────────────────────────────────────────┤
│  process_api_data.py → adjacency_real.json (132x132 matrix)  │
│  add_temporal_features.py → hero_stats_temporal.csv           │
│  create_final_training_data.py → MLBB_TRAINING_DATA_FINAL.json│
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Training                                  │
├─────────────────────────────────────────────────────────────┤
│  train_gcn_enhanced.py → gcn_model_v2.pt (132 heroes)       │
└─────────────────────────────────────────────────────────────┘
```

---

## Backup System

Before any data update, run:
```bash
python3 training/backup_data.py
```

Backups are saved to: `training/backups/YYYYMMDD_HHMMSS/`

---

## Manual Setup Required

### 1. mlbb.io Parse API Key (Free)
The mlbb.io API provides rank-specific hero statistics (Mythic/Legend/Epic win rates).

1. Go to [parse.bot](https://parse.bot) and create a free account
2. Get your API key
3. Run:
   ```bash
   python3 training/fetch_mlbb_io.py --api-key YOUR_KEY
   ```

### 2. Kaggle API (Optional)
For downloading MLBB tournament datasets from Kaggle:

1. Create a [Kaggle account](https://www.kaggle.com/account/login)
2. Go to [Kaggle API Settings](https://www.kaggle.com/settings/api) and generate a token
3. Either:
   - Set environment variable: `export KAGGLE_API_TOKEN=your_token`
   - Or save to file: `~/.kaggle/access_token`
4. Run:
   ```bash
   python3 training/fetch_kaggle_datasets.py --download-all
   ```
