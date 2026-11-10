"""Validation: check that a raw snapshot's values make sense before features are built.

Runs right after ingestion in the pipeline:
    ingest -> load_raw -> validate -> build_features -> ...

``load_raw`` (in ingest.py) already checked that the file can be read and has the
right columns. This module checks the values themselves, using rules taken from
the EDA (notebooks/01_eda.ipynb):

- 4 lists, each with exactly 10 rows per week, ranks 1-10 used once each
- weeks are Sundays (Netflix's week-ending day)
- hours viewed always present and above 0
- runtime and views are blank before Netflix's June 2023 metric switch, and
  present from then on
- runtime can be 0 (about 2.5% of rows since the switch), so 0 is allowed, but a
  large share of zeros raises a warning instead of stopping the run
- no title appears twice in the same week's list

If any hard rule breaks, ``validate`` raises ``pandera.errors.SchemaErrors``
listing every failing row, and the pipeline stops before training.

Usage (from the repo root):
    python -m src.validate        # validates the newest snapshot in data/raw/
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from src.ingest import latest_snapshot, load_raw

# Netflix switched its ranking metric from hours viewed to views on this week.
# Runtime and weekly views are only published from this week onward.
METRIC_SWITCH = pd.Timestamp("2023-06-18")

CATEGORIES = ["Films (English)", "Films (Non-English)", "TV (English)", "TV (Non-English)"]

# Longest runtime in the EDA was 47.6 hours (a long TV season). Anything above
# this is almost certainly a units or data-entry error.
MAX_RUNTIME_HOURS = 100

# Share of zero runtimes since the switch was ~2.5% in the EDA. Above this, warn.
MAX_ZERO_RUNTIME_SHARE = 0.05

TITLE_KEY = ["week", "category", "show_title", "season_title"]


# --- Whole-table checks ------------------------------------------------------
# Each returns True when the data passes. Pandera reports the check's name on failure.


def _ten_rows_ranked_1_to_10(df: pd.DataFrame) -> bool:
    """Every (week, list) has exactly 10 rows, with ranks 1-10 each used once."""
    ranks = df.groupby(["week", "category"])["weekly_rank"].apply(sorted)
    return bool(ranks.apply(lambda r: r == list(range(1, 11))).all())


def _no_duplicate_titles(df: pd.DataFrame) -> bool:
    """No title appears twice in the same week's list. Films have a blank season, so fill it first."""
    # Cast to object first: if every season is blank (e.g. films only), pandas stores the
    # column as numbers, and filling numbers with text is deprecated.
    keys = df[TITLE_KEY].assign(season_title=df["season_title"].astype(object).fillna("N/A"))
    return not keys.duplicated().any()


def _runtime_and_views_present_after_switch(df: pd.DataFrame) -> bool:
    """From the metric switch onward, Netflix publishes runtime and views for every row."""
    after = df[df["week"] >= METRIC_SWITCH]
    return bool(after["runtime"].notna().all() and after["weekly_views"].notna().all())


def _zero_runtime_share_is_normal(df: pd.DataFrame) -> bool:
    """Soft check: a jump in zero runtimes suggests a broken upstream field."""
    after = df[df["week"] >= METRIC_SWITCH]
    if after.empty:
        return True
    return float((after["runtime"] == 0).mean()) <= MAX_ZERO_RUNTIME_SHARE


# --- Schema ------------------------------------------------------------------

schema = pa.DataFrameSchema(
    columns={
        "week": pa.Column(
            "datetime64[ns]",
            pa.Check(lambda s: s.dt.day_of_week == 6, error="week must be a Sunday"),
            coerce=True,
        ),
        "category": pa.Column(str, pa.Check.isin(CATEGORIES)),
        "weekly_rank": pa.Column(int, pa.Check.in_range(1, 10), coerce=True),
        "show_title": pa.Column(str, pa.Check.str_length(min_value=1)),
        "season_title": pa.Column(str, nullable=True),
        "weekly_hours_viewed": pa.Column(float, pa.Check.gt(0), coerce=True),
        "runtime": pa.Column(
            float, pa.Check.in_range(0, MAX_RUNTIME_HOURS), nullable=True, coerce=True
        ),
        "weekly_views": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
        "cumulative_weeks_in_top_10": pa.Column(int, pa.Check.ge(1), coerce=True),
    },
    checks=[
        pa.Check(_ten_rows_ranked_1_to_10, name="ten_rows_ranked_1_to_10_per_week_and_list"),
        pa.Check(_no_duplicate_titles, name="no_duplicate_titles_in_a_week"),
        pa.Check(_runtime_and_views_present_after_switch, name="runtime_and_views_present_after_switch"),
        pa.Check(
            _zero_runtime_share_is_normal,
            name="zero_runtime_share_is_normal",
            raise_warning=True,  # warns instead of failing
        ),
    ],
    strict=False,  # extra columns (e.g. is_staggered_launch) are allowed and passed through
)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Return the validated (type-converted) DataFrame, or raise SchemaErrors listing every problem.

    ``lazy=True`` makes Pandera collect all failures in one go instead of stopping at the first,
    so a bad file can be fixed in one pass.
    """
    return schema.validate(df, lazy=True)


def main() -> None:
    path = latest_snapshot()
    df = validate(load_raw(path))
    print(f"{path.name}: {len(df):,} rows passed validation")


if __name__ == "__main__":
    main()
