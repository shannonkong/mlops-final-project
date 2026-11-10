"""Cleaning: repair known problems in validated data, without changing the raw values.

Runs after validation in the pipeline:
    ingest -> load_raw -> validate -> clean -> build_features -> ...

Runtime repair
--------------
Netflix's ``runtime`` is the total length of a film or of a whole TV season, in
hours. The EDA found 172 rows (all TV) where it's 0, which is wrong: e.g. a
40-episode drama at 45 minutes an episode should be about 30 hours. Those rows
still have ``weekly_views``, and Netflix defines views = hours viewed / runtime,
so runtime can be recovered as hours viewed / views.

Rather than overwrite ``runtime``, two columns are added, so it's always clear
where a number came from:

- ``runtime_filled``: best available runtime in hours
- ``runtime_source``: one of
    "netflix"               runtime as published (above 0)
    "derived_from_views"    hours viewed / weekly views, same row
    "same_title_other_week" Netflix's runtime for the same title in another week
    "missing"               no runtime available (e.g. most weeks before June 2023)

Views are published rounded to the nearest 100,000, so a derived runtime can be
off by a few percent for titles with fewer views. That's fine for a model
feature, and the source column lets anyone exclude derived values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TITLE_KEY = ["category", "show_title", "season_title"]

SOURCE_NETFLIX = "netflix"
SOURCE_DERIVED = "derived_from_views"
SOURCE_OTHER_WEEK = "same_title_other_week"
SOURCE_MISSING = "missing"


def fill_runtime(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``runtime_filled`` and ``runtime_source`` added.

    Fills in this order, using each method only where the earlier ones found nothing:
    1. Netflix's runtime, when it's above 0
    2. hours viewed / weekly views from the same row
    3. Netflix's runtime for the same title in another week (median, if it varies)
    """
    out = df.copy()
    published = out["runtime"].where(out["runtime"] > 0)

    views = out["weekly_views"].where(out["weekly_views"] > 0)
    derived = out["weekly_hours_viewed"] / views

    # Median published runtime per title, for titles with a usable value in some week.
    # season_title is blank for films, so fill it before grouping.
    keys = out[TITLE_KEY].assign(season_title=out["season_title"].astype(object).fillna("N/A"))
    other_week = published.groupby([keys[c] for c in TITLE_KEY]).transform("median")

    out["runtime_filled"] = published.fillna(derived).fillna(other_week)
    out["runtime_source"] = np.select(
        [published.notna(), derived.notna(), other_week.notna()],
        [SOURCE_NETFLIX, SOURCE_DERIVED, SOURCE_OTHER_WEEK],
        default=SOURCE_MISSING,
    )
    return out


def runtime_source_summary(df: pd.DataFrame) -> pd.Series:
    """Row counts per runtime source: a quick check for the logs, the README, or a slide."""
    return df["runtime_source"].value_counts()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """All cleaning steps, in order. Add new steps here as they're needed."""
    return fill_runtime(df)
