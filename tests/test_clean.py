"""Tests for src/clean.py: each runtime source is used in the right order, and raw values stay untouched."""

import numpy as np
import pandas as pd
import pytest

from src import clean as c


def row(week, title, hours, runtime, views, season="Season 1", category="TV (Non-English)"):
    return {
        "week": pd.Timestamp(week),
        "category": category,
        "show_title": title,
        "season_title": season,
        "weekly_hours_viewed": hours,
        "runtime": runtime,
        "weekly_views": views,
    }


@pytest.fixture
def df():
    return pd.DataFrame(
        [
            row("2024-01-07", "Drama A", 30_000_000, 30.0, 1_000_000),  # 0: published
            row("2024-01-07", "Drama B", 60_000_000, 0.0, 2_000_000),  # 1: zero -> derive 30.0
            row("2022-01-02", "Drama C", 8_000_000, np.nan, np.nan),  # 2: pre-switch -> other week
            row("2024-01-07", "Drama C", 9_000_000, 8.0, 1_100_000),  # 3: published
            row("2022-01-02", "Drama D", 5_000_000, np.nan, np.nan),  # 4: nothing -> missing
            row("2024-01-07", "Film E", 4_000_000, 2.0, 2_000_000, season=np.nan, category="Films (English)"),
        ]
    )


def test_sources_assigned_in_order(df):
    out = c.fill_runtime(df)
    assert list(out["runtime_source"]) == [
        c.SOURCE_NETFLIX,
        c.SOURCE_DERIVED,
        c.SOURCE_OTHER_WEEK,
        c.SOURCE_NETFLIX,
        c.SOURCE_MISSING,
        c.SOURCE_NETFLIX,
    ]


def test_zero_runtime_derived_from_views(df):
    out = c.fill_runtime(df)
    assert out.loc[1, "runtime_filled"] == pytest.approx(30.0)  # 60M hours / 2M views


def test_pre_switch_row_uses_same_titles_runtime(df):
    out = c.fill_runtime(df)
    assert out.loc[2, "runtime_filled"] == pytest.approx(8.0)


def test_missing_stays_missing(df):
    assert np.isnan(c.fill_runtime(df).loc[4, "runtime_filled"])


def test_films_with_blank_season_are_handled(df):
    out = c.fill_runtime(df)
    assert out.loc[5, "runtime_filled"] == pytest.approx(2.0)


def test_raw_runtime_is_not_changed(df):
    original = df["runtime"].copy()
    out = c.fill_runtime(df)
    pd.testing.assert_series_equal(out["runtime"], original)
    pd.testing.assert_series_equal(df["runtime"], original)  # input not modified either


def test_zero_views_does_not_divide_by_zero(df):
    df.loc[1, "weekly_views"] = 0
    out = c.fill_runtime(df)
    assert out.loc[1, "runtime_source"] == c.SOURCE_MISSING
    assert not np.isinf(out["runtime_filled"]).any()


def test_zero_runtime_is_not_used_as_other_week_value():
    df = pd.DataFrame(
        [
            row("2022-01-02", "Drama F", 5_000_000, np.nan, np.nan),
            row("2024-01-07", "Drama F", 6_000_000, 0.0, np.nan),  # zero and no views: unusable
        ]
    )
    out = c.fill_runtime(df)
    assert set(out["runtime_source"]) == {c.SOURCE_MISSING}


def test_summary_counts(df):
    summary = c.runtime_source_summary(c.fill_runtime(df))
    assert summary[c.SOURCE_NETFLIX] == 3
    assert summary.sum() == len(df)
