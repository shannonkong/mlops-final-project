"""Ingestion: get Netflix's global weekly Top 10 file into data/raw/ and load it.

Each run saves a dated snapshot named like ``2026-10-23_global_weekly.xlsx``, so
any model can be traced back to the exact data it was trained on (DVC versions
the snapshots). The file can come from a download URL or from a file someone
already downloaded by hand.

Usage (from the repo root):
    python -m src.ingest --source "C:/Users/me/Downloads/global_weekly.xlsx"
    python -m src.ingest --url https://...        # download directly
    python -m src.ingest                          # uses NETFLIX_TOP10_URL or DEFAULT_URL

Other code (features, validation, the Prefect flow) should load data with
``load_raw(latest_snapshot())`` so everything reads the file the same way.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# --- Configuration -----------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

# Snapshot filename: <YYYY-MM-DD>_global_weekly.xlsx (date = day the data was pulled)
FILENAME_SUFFIX = "_global_weekly.xlsx"

# Netflix's download link for the global weekly Top 10 file. Check that this
# matches the "download" link on the Netflix Top 10 site, or override it with
# the NETFLIX_TOP10_URL environment variable (e.g. in your .env file).
DEFAULT_URL = "https://www.netflix.com/tudum/top10/data/all-weeks-global.xlsx"

# Standard column names used across the project, with other spellings the
# file might use. Headings are lowercased with spaces turned into underscores first.
COLUMN_ALIASES = {
    "week": ["week", "week_ending", "week_of"],
    "category": ["category", "list"],
    "weekly_rank": ["weekly_rank", "rank"],
    "show_title": ["show_title", "title"],
    "season_title": ["season_title", "season"],
    "weekly_hours_viewed": ["weekly_hours_viewed", "hours_viewed"],
    "runtime": ["runtime", "runtime_hours"],
    "weekly_views": ["weekly_views", "views"],
    "cumulative_weeks_in_top_10": ["cumulative_weeks_in_top_10", "weeks_in_top_10", "cumulative_weeks"],
}
REQUIRED_COLUMNS = [
    "week",
    "category",
    "weekly_rank",
    "show_title",
    "weekly_hours_viewed",
    "cumulative_weeks_in_top_10",
]
OPTIONAL_COLUMNS = ["season_title", "runtime", "weekly_views"]


# --- Finding snapshots -------------------------------------------------------


def snapshot_name(snapshot_date: date) -> str:
    """Filename for a snapshot taken on ``snapshot_date``, e.g. 2026-10-23_global_weekly.xlsx."""
    return f"{snapshot_date.isoformat()}{FILENAME_SUFFIX}"


def is_snapshot(path: Path) -> bool:
    """True for real snapshot files; False for Excel lock files (~$...) and hidden files."""
    return path.is_file() and path.name.endswith(FILENAME_SUFFIX) and not path.name.startswith(("~$", "."))


def latest_snapshot(raw_dir: Path = RAW_DIR) -> Path:
    """Newest snapshot in ``raw_dir``. Names start with the date, so sorting by name sorts by date."""
    snapshots = sorted(p for p in raw_dir.glob(f"*{FILENAME_SUFFIX}") if is_snapshot(p))
    if not snapshots:
        raise FileNotFoundError(
            f"No *{FILENAME_SUFFIX} snapshot in {raw_dir}. Run `python -m src.ingest` "
            "or `dvc pull` first."
        )
    return snapshots[-1]


# --- Getting the file --------------------------------------------------------


def _looks_like_xlsx(content: bytes) -> bool:
    # .xlsx files are zip archives, which always start with the bytes "PK".
    return content[:2] == b"PK"


def download(url: str, dest: Path, timeout: int = 60) -> Path:
    """Download the Excel file at ``url`` to ``dest``.

    Raises ValueError if the response isn't an Excel file (for example, the
    site returned a web page instead), so bad data never lands in data/raw.
    """
    logger.info("Downloading %s", url)
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "mlops-final-project/0.1"})
    response.raise_for_status()
    if not _looks_like_xlsx(response.content):
        raise ValueError(
            f"{url} did not return an .xlsx file. Check the download link on the "
            "Netflix Top 10 site, or download it by hand and use --source."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    return dest


def copy_local(source: Path, dest: Path) -> Path:
    """Copy a file someone already downloaded into data/raw under the standard name."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    if source.suffix.lower() != ".xlsx":
        raise ValueError(f"Expected an .xlsx file, got {source.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest


def ingest(
    source: Path | str | None = None,
    url: str | None = None,
    raw_dir: Path = RAW_DIR,
    snapshot_date: date | None = None,
    overwrite: bool = False,
) -> Path:
    """Save today's snapshot to ``raw_dir`` and return its path.

    Uses ``source`` (a local file) if given, otherwise downloads from ``url``,
    the NETFLIX_TOP10_URL environment variable, or DEFAULT_URL. If a snapshot
    for the date already exists, it's reused unless ``overwrite`` is True, so
    rerunning the pipeline on the same day doesn't download again.
    """
    # Dates use UTC so the snapshot name is the same whichever teammate's computer runs this.
    dest = Path(raw_dir) / snapshot_name(snapshot_date or datetime.now(UTC).date())
    if dest.exists() and not overwrite:
        logger.info("Snapshot already exists, reusing %s", dest)
        return dest

    if source is not None:
        path = copy_local(Path(source), dest)
    else:
        path = download(url or os.getenv("NETFLIX_TOP10_URL", DEFAULT_URL), dest)

    load_raw(path)  # fail now, not later, if the file can't be read
    logger.info("Saved snapshot %s", path)
    return path


# --- Loading -----------------------------------------------------------------


def _find_header_row(path: Path, max_rows: int = 20) -> int:
    """Row number of the real header. Some downloads have title rows above it; the header contains 'rank'."""
    preview = pd.read_excel(path, header=None, nrows=max_rows)
    for i, row in preview.iterrows():
        if row.astype(str).str.contains("rank", case=False).any():
            return int(i)
    return 0


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase/underscore headings, then map known alternative spellings to standard names."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    rename = {
        alias: standard
        for standard, aliases in COLUMN_ALIASES.items()
        for alias in aliases
        if alias in df.columns and alias != standard and standard not in df.columns
    }
    return df.rename(columns=rename)


def load_raw(path: Path | str) -> pd.DataFrame:
    """Read a snapshot into a DataFrame with standard column names and a datetime ``week``.

    Raises KeyError if a required column is missing. Optional columns that are
    absent are added as empty, so later code can rely on them existing.
    """
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, header=_find_header_row(path))
    else:  # also accept Netflix's .tsv format
        df = pd.read_csv(path, sep="\t")

    df = standardize_columns(df).dropna(how="all")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"{path.name} is missing columns {missing}. Columns found: {list(df.columns)}. "
            "If a column is just spelled differently, add the spelling to COLUMN_ALIASES."
        )
    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    df["week"] = pd.to_datetime(df["week"])
    return df.reset_index(drop=True)


# --- Command line ------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a dated snapshot of Netflix's global weekly Top 10 file.")
    parser.add_argument("--source", type=Path, help="Path to an .xlsx you already downloaded")
    parser.add_argument("--url", help="Download URL (default: NETFLIX_TOP10_URL or DEFAULT_URL)")
    parser.add_argument("--overwrite", action="store_true", help="Replace today's snapshot if it exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path = ingest(source=args.source, url=args.url, overwrite=args.overwrite)
    df = load_raw(path)
    print(
        f"{path.name}: {len(df):,} rows, weeks {df['week'].min().date()} "
        f"to {df['week'].max().date()}"
    )


if __name__ == "__main__":
    main()
