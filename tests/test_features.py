"""Tests for src/features.py: the label, each feature, the leakage guard, and the time split."""

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src.clean import clean
from src.features import FEATURE_COLUMNS, LABEL, NOT_ON_LIST_RANK, build_features, time_split
from src.validate import validate


@pytest.fixture
def cleaned(raw_df):
    return clean(validate(raw_df))


def one_title(features, title, week):
    match = features[(features["show_title"] == title) & (features["week"] == pd.Timestamp(week))]
    assert len(match) == 1
    return match.iloc[0]


def test_synthetic_data_passes_validation(raw_df):
    validate(raw_df)  # if this fails, the shared test data no longer looks like the real file


def test_newest_week_dropped_for_training(cleaned):
    features = build_features(cleaned)
    assert features["week"].max() < cleaned["week"].max()
    assert features[LABEL].notna().all()


def test_newest_week_kept_unlabeled_for_scoring(cleaned):
    features = build_features(cleaned, drop_unlabeled=False)
    newest = features[features["week"] == cleaned["week"].max()]
    assert len(newest) == 40
    assert newest[LABEL].isna().all()


def test_label_matches_next_weeks_list(cleaned):
    features = build_features(cleaned)
    row = features.iloc[0]
    next_week = cleaned[cleaned["week"] == row["week"] + pd.Timedelta(days=7)]
    on_next_list = (
        (next_week["category"] == row["category"]) & (next_week["show_title"] == row["show_title"])
    ).any()
    assert row[LABEL] == int(on_next_list)


def test_features_have_no_blanks(cleaned):
    features = build_features(cleaned)
    assert not features[FEATURE_COLUMNS].isna().any().any()


def test_previous_week_features():
    df = pd.DataFrame(
        [
            # Film A: rank 3 in week 1, rank 1 in week 2. Film B: only in week 2.
            {"week": "2023-04-02", "show_title": "Film A", "weekly_rank": 3, "weekly_hours_viewed": 1_000_000},
            {"week": "2023-04-09", "show_title": "Film A", "weekly_rank": 1, "weekly_hours_viewed": 4_000_000},
            {"week": "2023-04-09", "show_title": "Film B", "weekly_rank": 2, "weekly_hours_viewed": 2_000_000},
            {"week": "2023-04-16", "show_title": "Film A", "weekly_rank": 1, "weekly_hours_viewed": 3_000_000},
        ]
    ).assign(
        week=lambda d: pd.to_datetime(d["week"]),
        category="Films (English)",
        season_title=np.nan,
        cumulative_weeks_in_top_10=1,
    )
    f = build_features(df)

    a = one_title(f, "Film A", "2023-04-09")
    assert a["prev_rank"] == 3
    assert a["rank_change"] == 2  # moved up from 3 to 1
    assert a["on_list_last_week"] == 1
    assert a["hours_change"] == pytest.approx(np.log1p(4e6) - np.log1p(1e6))
    assert a[LABEL] == 1  # on the list again on 2023-04-16

    b = one_title(f, "Film B", "2023-04-09")
    assert b["prev_rank"] == NOT_ON_LIST_RANK
    assert b["on_list_last_week"] == 0
    assert b["hours_change"] == 0.0
    assert b[LABEL] == 0  # gone by 2023-04-16


def test_cutoff_and_share_features(cleaned):
    f = build_features(cleaned)
    one_list = f[(f["week"] == f["week"].min()) & (f["category"] == "TV (English)")]
    assert one_list.loc[one_list["weekly_rank"] == 10, "hours_vs_cutoff"].iloc[0] == pytest.approx(0)
    assert one_list["list_share"].sum() == pytest.approx(1.0)


def test_list_flags(cleaned):
    f = build_features(cleaned)
    flags = f.groupby("category")[["is_tv", "is_english"]].first()
    assert flags.loc["TV (English)"].tolist() == [1, 1]
    assert flags.loc["Films (Non-English)"].tolist() == [0, 0]


def test_no_future_leakage(cleaned):
    """Changing next week's data must not change this week's features (only the label may change)."""
    weeks = np.sort(cleaned["week"].unique())
    cut = weeks[len(weeks) // 2]
    before = build_features(cleaned, drop_unlabeled=False)

    tampered = cleaned.copy()
    future = tampered["week"] > cut
    tampered.loc[future, "weekly_hours_viewed"] *= 10
    tampered.loc[future, "show_title"] = tampered.loc[future, "show_title"] + " (changed)"
    after = build_features(tampered, drop_unlabeled=False)

    keep = ["title_key", "week", *FEATURE_COLUMNS]
    old = before.loc[before["week"] <= cut, keep].reset_index(drop=True)
    new = after.loc[after["week"] <= cut, keep].reset_index(drop=True)
    pd.testing.assert_frame_equal(old, new)


def test_feature_schema_catches_bad_values(cleaned):
    f = build_features(cleaned)
    f.loc[0, "list_share"] = 1.5
    from src.features import feature_schema

    with pytest.raises(SchemaErrors):
        feature_schema.validate(f, lazy=True)


def test_time_split(cleaned):
    f = build_features(cleaned)
    train, holdout = time_split(f, holdout_weeks=5)
    assert holdout["week"].nunique() == 5
    assert train["week"].max() < holdout["week"].min()
    assert len(train) + len(holdout) == len(f)


def test_time_split_needs_enough_weeks(cleaned):
    with pytest.raises(ValueError):
        time_split(build_features(cleaned), holdout_weeks=500)
