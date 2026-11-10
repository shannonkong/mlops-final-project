"""Tests for src/validate.py: a small valid table passes, and each kind of bad data is caught."""

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src import validate as v


@pytest.fixture
def valid_df():
    """Two weeks either side of the metric switch: 4 lists x 10 ranks each, like the real file."""
    rows = []
    for week in [pd.Timestamp("2023-06-11"), v.METRIC_SWITCH]:
        after = week >= v.METRIC_SWITCH
        for category in v.CATEGORIES:
            is_tv = category.startswith("TV")
            for rank in range(1, 11):
                title = f"{category} title {rank}"
                rows.append(
                    {
                        "week": week,
                        "category": category,
                        "weekly_rank": rank,
                        "show_title": title,
                        "season_title": f"{title}: Season 1" if is_tv else np.nan,
                        "weekly_hours_viewed": 1_000_000.0 * (11 - rank),
                        "runtime": (8.0 if is_tv else 2.0) if after else np.nan,
                        "weekly_views": (500_000.0 * (11 - rank)) if after else np.nan,
                        "cumulative_weeks_in_top_10": 1,
                    }
                )
    return pd.DataFrame(rows)


def test_valid_data_passes(valid_df):
    out = v.validate(valid_df)
    assert len(out) == 80


def test_extra_columns_are_allowed(valid_df):
    valid_df["is_staggered_launch"] = False
    assert "is_staggered_launch" in v.validate(valid_df).columns


def test_single_zero_runtime_is_allowed(valid_df):
    # The real data has some zero runtimes, so one zero must not fail or warn.
    valid_df.loc[valid_df["week"] == v.METRIC_SWITCH, "runtime"] = [0.0] + [2.0] * 39
    v.validate(valid_df)


def test_many_zero_runtimes_warns_but_passes(valid_df):
    valid_df.loc[valid_df["week"] == v.METRIC_SWITCH, "runtime"] = 0.0
    with pytest.warns(UserWarning, match="zero_runtime_share_is_normal"):
        v.validate(valid_df)


@pytest.mark.parametrize(
    "column, bad_value",
    [
        ("weekly_rank", 11),  # out of range (also breaks the 1-10 per week rule)
        ("weekly_hours_viewed", 0),  # hours must be above 0
        ("weekly_hours_viewed", -5),
        ("runtime", -1.0),
        ("runtime", 500.0),  # implausibly long: likely a units error
        ("cumulative_weeks_in_top_10", 0),
        ("category", "Movies (English)"),  # not one of the four lists
    ],
)
def test_bad_values_are_rejected(valid_df, column, bad_value):
    valid_df.loc[45, column] = bad_value  # a row after the switch
    with pytest.raises(SchemaErrors):
        v.validate(valid_df)


def test_rank_as_text_is_rejected(valid_df):
    valid_df["weekly_rank"] = valid_df["weekly_rank"].astype(object)
    valid_df.loc[0, "weekly_rank"] = "first"
    with pytest.raises(SchemaErrors):
        v.validate(valid_df)


def test_missing_row_in_a_week_is_rejected(valid_df):
    with pytest.raises(SchemaErrors, match="ten_rows_ranked_1_to_10"):
        v.validate(valid_df.drop(index=5))


def test_duplicate_title_in_a_week_is_rejected(valid_df):
    valid_df.loc[1, "show_title"] = valid_df.loc[0, "show_title"]  # same film twice, same week
    with pytest.raises(SchemaErrors, match="no_duplicate_titles"):
        v.validate(valid_df)


def test_missing_runtime_after_switch_is_rejected(valid_df):
    valid_df.loc[45, "runtime"] = np.nan
    with pytest.raises(SchemaErrors, match="runtime_and_views_present_after_switch"):
        v.validate(valid_df)


def test_missing_runtime_before_switch_is_allowed(valid_df):
    assert valid_df.loc[valid_df["week"] < v.METRIC_SWITCH, "runtime"].isna().all()
    v.validate(valid_df)


def test_non_sunday_week_is_rejected(valid_df):
    valid_df["week"] = valid_df["week"] + pd.Timedelta(days=1)  # Mondays
    with pytest.raises(SchemaErrors, match="Sunday"):
        v.validate(valid_df)


def test_all_problems_reported_at_once(valid_df):
    valid_df.loc[45, "weekly_hours_viewed"] = -5
    valid_df.loc[46, "category"] = "Movies"
    with pytest.raises(SchemaErrors) as excinfo:
        v.validate(valid_df)
    failed_columns = set(excinfo.value.failure_cases["column"].dropna())
    assert {"weekly_hours_viewed", "category"} <= failed_columns
