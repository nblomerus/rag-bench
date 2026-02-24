#!/usr/bin/env python3
"""Migrate existing eval_results/*.{json,md} into eval_results/manual/.

One-time migration to support the production/manual subdirectory split.
Safe to run multiple times — skips if no root-level eval files exist.
"""

import json
import shutil
from pathlib import Path

EVAL_DIR = Path(__file__).parent.parent / "eval_results"
MANUAL_DIR = EVAL_DIR / "manual"
PRODUCTION_DIR = EVAL_DIR / "production"


def migrate():
    if not EVAL_DIR.exists():
        print("No eval_results/ directory found. Nothing to migrate.")
        return

    MANUAL_DIR.mkdir(exist_ok=True)
    PRODUCTION_DIR.mkdir(exist_ok=True)

    # Move root-level eval files to manual/
    moved = 0
    for f in sorted(EVAL_DIR.iterdir()):
        if f.is_file() and f.name.startswith("eval_") and f.suffix in (".json", ".md"):
            dest = MANUAL_DIR / f.name
            shutil.move(str(f), str(dest))
            moved += 1

    print(f"Moved {moved} files to eval_results/manual/")

    # Patch run_type into JSON metadata for migrated files
    patched = 0
    for json_file in sorted(MANUAL_DIR.glob("eval_*.json")):
        try:
            data = json.loads(json_file.read_text())
            meta = data.get("metadata", {})
            if "run_type" not in meta:
                meta["run_type"] = "manual"
                data["metadata"] = meta
                json_file.write_text(json.dumps(data, indent=2, default=str))
                patched += 1
        except Exception as e:
            print(f"  Warning: Could not patch {json_file.name}: {e}")

    print(f"Patched run_type into {patched} JSON files")
    print("Created directories: eval_results/production/, eval_results/manual/")


if __name__ == "__main__":
    migrate()
