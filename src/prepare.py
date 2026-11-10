"""Phase 2 in one command: raw snapshot -> validated, cleaned, labeled, split data + baseline.

Steps, in order:
    1. load_raw       read the newest snapshot in data/raw/ (ingest.py)
    2. validate       Pandera checks on the raw values; stops here if any rule breaks (validate.py)
    3. clean          recover broken runtimes (clean.py)
    4. build_features label + features, checked by a second Pandera schema (features.py)
    5. time_split     last 20 weeks become the holdout, untouched until production validation
    6. rank_baseline  PR-AUC of the rank rule on the training weeks (baseline.py)
    7. save           data/processed/train.parquet, holdout.parquet, metadata.json

metadata.json records which snapshot the data came from (name and MD5 hash),
row counts, date ranges, the feature list and the baseline score, so every
model trained later can be traced back to its exact data.

Usage (from the repo root):
    python -m src.ingest      # get this week's snapshot (optional)
    python -m src.prepare     # build data/processed/
    dvc add data/processed    # version the outputs, then commit data/processed.dvc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.baseline import rank_baseline
from src.clean import clean, runtime_source_summary
from src.features import FEATURE_COLUMNS, LABEL, build_features, time_split
from src.ingest import REPO_ROOT, latest_snapshot, load_raw
from src.validate import validate

logger = logging.getLogger(__name__)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
HOLDOUT_WEEKS = 20


def _md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _date_range(df) -> list[str]:
    return [str(df["week"].min().date()), str(df["week"].max().date())]


def prepare(
    raw_path: Path | str | None = None,
    processed_dir: Path = PROCESSED_DIR,
    holdout_weeks: int = HOLDOUT_WEEKS,
) -> dict:
    """Run every Phase 2 step and save the outputs. Returns the metadata that was saved."""
    raw_path = Path(raw_path) if raw_path else latest_snapshot()
    logger.info("Loading %s", raw_path.name)

    raw = load_raw(raw_path)
    validated = validate(raw)  # raises pandera.errors.SchemaErrors if the data breaks a rule
    cleaned = clean(validated)
    features = build_features(cleaned)  # validated again by features.feature_schema
    train, holdout = time_split(features, holdout_weeks)
    baseline = rank_baseline(train)

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(processed_dir / "train.parquet", index=False)
    holdout.to_parquet(processed_dir / "holdout.parquet", index=False)

    metadata = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_snapshot": raw_path.name,
        "source_md5": _md5(raw_path),
        "holdout_weeks": holdout_weeks,
        "train": {"rows": len(train), "weeks": _date_range(train)},
        "holdout": {"rows": len(holdout), "weeks": _date_range(holdout)},
        "label": LABEL,
        "feature_columns": FEATURE_COLUMNS,
        "runtime_sources": {k: int(v) for k, v in runtime_source_summary(cleaned).items()},
        "baseline_train": baseline,
    }
    (processed_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Saved train (%d rows) and holdout (%d rows) to %s", len(train), len(holdout), processed_dir)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/processed/ from the newest raw snapshot.")
    parser.add_argument("--raw", type=Path, help="Snapshot to use (default: newest in data/raw/)")
    parser.add_argument("--holdout-weeks", type=int, default=HOLDOUT_WEEKS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    meta = prepare(raw_path=args.raw, holdout_weeks=args.holdout_weeks)
    b = meta["baseline_train"]
    print(f"Source:   {meta['source_snapshot']}")
    print(f"Train:    {meta['train']['rows']:,} rows, {meta['train']['weeks'][0]} to {meta['train']['weeks'][1]}")
    print(f"Holdout:  {meta['holdout']['rows']:,} rows, {meta['holdout']['weeks'][0]} to {meta['holdout']['weeks'][1]}")
    print(f"Baseline: PR-AUC {b['pr_auc']} (random guessing: {b['positive_rate']})")


if __name__ == "__main__":
    main()
