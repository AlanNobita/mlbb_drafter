#!/usr/bin/env python3
"""Backup all training data before updates."""
import os
import shutil
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "training" / "data"
BACKUP_DIR = BASE_DIR / "training" / "backups"


def backup():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / timestamp
    os.makedirs(backup_path, exist_ok=True)
    
    # Backup api_data
    api_src = DATA_DIR / "api_data"
    api_dst = backup_path / "api_data"
    if api_src.exists():
        shutil.copytree(api_src, api_dst)
        print(f"Backed up api_data ({len(list(api_src.iterdir()))} files)")
    
    # Backup processed files
    for f in DATA_DIR.glob("*.csv"):
        shutil.copy2(f, backup_path / f.name)
    for f in DATA_DIR.glob("*.json"):
        if f.parent.name != "backups" and f.parent.name != "api_data":
            shutil.copy2(f, backup_path / f.name)
    
    print(f"Backup saved to: {backup_path}")
    print(f"Total size: {sum(f.stat().st_size for f in backup_path.rglob('*') if f.is_file()) / 1024 / 1024:.1f}MB")
    return backup_path


if __name__ == "__main__":
    backup()
