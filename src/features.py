"""Feature building: turn cleaned weekly rows into one labeled row per title per week.

Runs after cleaning in the pipeline:
    ingest -> load_raw -> validate -> clean -> build_features -> split -> baseline

The label
---------
``survives`` = 1 if the same title (same list, show and season) is on the same
Top 10 list the following week, else 0. The newest week has no "next week" yet,
so its label is unknown: those rows are dropped for training, or kept with a
blank label when scoring live data.

The features (only information available when this week's list is published)
-----------------------------------------------------------------------------
weekly_rank         rank this week (1 = top)
prev_rank           rank last week; 11 if the title wasn't on the list last week
rank_change         prev_rank - weekly_rank (positive = moving up)
on_list_last_week   1 if the title was on this list last week
log_hours           log of hours viewed this week
hours_change        change in log hours since last week (0 if not on the list last week)
hours_vs_cutoff     log hours above the #10 title this week (how safely inside the Top 10)
list_share          share of this list's total hours this week
weeks_in_top_10     weeks on the chart so far (Netflix's cumulative count)
week_of_year        ISO week number, for seasonal patterns such as December
is_tv, is_english   which of the four lists the title is on

Runtime isn't a feature yet: about 4,000 rows before June 2023 have no runtime.
The plan is to backfill those from TMDB first (see clean.py), then add it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa

LABEL = "survives"
ID_COLUMNS = ["week", "category", "show_title", "season_title", "title_key"]
FEATURE_COLUMNS = [
    "weekly_rank",
    "prev_rank",
    "rank_change",
    "on_list_last_week",
    "log_hours",
    "hours_change",
    "hours_vs_cutoff",
    "list_share",
    "weeks_in_top_10",
    "week_of_year",
    "is_tv",
    "is_english",
]

NOT_ON_LIST_RANK = 11  # stand-in "rank" for titles that weren't in last week's Top 10
ONE_WEEK = pd.Timedelta(days=7)


def add_title_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``title_key``: list + show + season, the identity of a title from week to week.

    Films have no season, so a blank season becomes "N/A".
    """
    out = df.copy()
    # Cast to object first: if every season is blank (e.g. films only), pandas stores the
    # column as numbers, and filling numbers with text is deprecated.
    season = out["season_title"].astype(object).fillna("N/A")
    out["title_key"] = out["category"] + " | " + out["show_title"] + " | " + season
    return out


def _add_label(df: pd.DataFrame) -> pd.DataFrame:
    """``survives`` = 1 if the title is on the same list next week; blank for the newest week."""
    out = df.copy()
    present = set(zip(out["title_key"], out["week"], strict=True))
    out[LABEL] = [int((key, week + ONE_WEEK) in present) for key, week in zip(out["title_key"], out["week"], strict=True)]
    out[LABEL] = out[LABEL].astype("Int64")
    out.loc[out["week"] == out["week"].max(), LABEL] = pd.NA  # next week not published yet
    return out


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Last week's rank and hours for the same title, matched on exactly 7 days earlier.
    # (A title that left the list and came back later counts as "not on the list last week".)
    last_week = out[["title_key", "week", "weekly_rank", "weekly_hours_viewed"]].copy()
    last_week["week"] = last_week["week"] + ONE_WEEK
    last_week = last_week.rename(columns={"weekly_rank": "prev_rank", "weekly_hours_viewed": "prev_hours"})
    out = out.merge(last_week, on=["title_key", "week"], how="left")

    on_list = out["prev_rank"].notna()
    out["on_list_last_week"] = on_list.astype("int64")
    out["prev_rank"] = out["prev_rank"].fillna(NOT_ON_LIST_RANK).astype("int64")
    out["rank_change"] = (out["prev_rank"] - out["weekly_rank"]).astype("int64")

    out["log_hours"] = np.log1p(out["weekly_hours_viewed"].astype(float))
    out["hours_change"] = np.where(on_list, out["log_hours"] - np.log1p(out["prev_hours"]), 0.0)

    by_list_week = out.groupby(["week", "category"])["weekly_hours_viewed"]
    out["hours_vs_cutoff"] = out["log_hours"] - np.log1p(by_list_week.transform("min").astype(float))
    out["list_share"] = out["weekly_hours_viewed"] / by_list_week.transform("sum")

    out["weeks_in_top_10"] = out["cumulative_weeks_in_top_10"].astype("int64")
    out["week_of_year"] = out["week"].dt.isocalendar().week.astype("int64")
    out["is_tv"] = out["category"].str.startswith("TV").astype("int64")
    out["is_english"] = out["category"].str.contains("(English)", regex=False).astype("int64")
    return out.drop(columns=["prev_hours"])


# Pandera check on the finished feature table: catches feature-building bugs
# (e.g. a bad merge creating blanks) before anything is saved or trained on.
feature_schema = pa.DataFrameSchema(
    {
        "week": pa.Column("datetime64[ns]"),
        "title_key": pa.Column(str),
        "weekly_rank": pa.Column("int64", pa.Check.in_range(1, 10)),
        "prev_rank": pa.Column("int64", pa.Check.in_range(1, NOT_ON_LIST_RANK)),
        "rank_change": pa.Column("int64", pa.Check.in_range(-9, 10)),
        "on_list_last_week": pa.Column("int64", pa.Check.isin([0, 1])),
        "log_hours": pa.Column(float, pa.Check.gt(0)),
        "hours_change": pa.Column(float),
        "hours_vs_cutoff": pa.Column(float, pa.Check.ge(0)),
        "list_share": pa.Column(float, pa.Check.in_range(0, 1, include_min=False)),
        "weeks_in_top_10": pa.Column("int64", pa.Check.ge(1)),
        "week_of_year": pa.Column("int64", pa.Check.in_range(1, 53)),
        "is_tv": pa.Column("int64", pa.Check.isin([0, 1])),
        "is_english": pa.Column("int64", pa.Check.isin([0, 1])),
        LABEL: pa.Column("Int64", pa.Check.isin([0, 1]), nullable=True),
    },
    unique=["title_key", "week"],
    strict=False,  # cleaned columns (runtime_filled etc.) are kept alongside
)


def build_features(df: pd.DataFrame, drop_unlabeled: bool = True) -> pd.DataFrame:
    """Build the labeled feature table from cleaned data, validated with ``feature_schema``.

    ``drop_unlabeled=True`` (training) drops the newest week, whose label isn't known yet.
    ``drop_unlabeled=False`` keeps it with a blank label, e.g. to score this week's titles.
    """
    out = _add_features(_add_label(add_title_key(df)))
    if drop_unlabeled:
        out = out[out[LABEL].notna()]
    out = out.sort_values(["week", "category", "weekly_rank"]).reset_index(drop=True)
    return feature_schema.validate(out, lazy=True)


def time_split(features: pd.DataFrame, holdout_weeks: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: the last ``holdout_weeks`` weeks are the holdout, everything before is training.

    Splitting by time (not at random) mimics predicting the future, and keeps every
    row of a given week on the same side.
    """
    weeks = np.sort(features["week"].unique())
    if len(weeks) <= holdout_weeks:
        raise ValueError(f"Need more than {holdout_weeks} weeks of data, got {len(weeks)}")
    cutoff = weeks[-holdout_weeks]
    train = features[features["week"] < cutoff].reset_index(drop=True)
    holdout = features[features["week"] >= cutoff].reset_index(drop=True)
    return train, holdout
