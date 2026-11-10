"""End-to-end test of the Phase 2 run: snapshot file in, processed files and metadata out."""

import json

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src.baseline import rank_baseline
from src.features import FEATURE_COLUMNS, LABEL
from src.prepare import prepare
from tests.conftest import make_raw


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "2026-11-09_global_weekly.xlsx"
    make_raw(n_weeks=30).to_excel(path, index=False)
    return path


def test_prepare_writes_outputs(snapshot, tmp_path):
    out = tmp_path / "processed"
    meta = prepare(raw_path=snapshot, processed_dir=out, holdout_weeks=5)

    train = pd.read_parquet(out / "train.parquet")
    holdout = pd.read_parquet(out / "holdout.parquet")
    saved = json.loads((out / "metadata.json").read_text())

    assert saved == json.loads(json.dumps(meta))
    assert saved["source_snapshot"] == snapshot.name
    assert holdout["week"].nunique() == 5
    assert train["week"].max() < holdout["week"].min()
    # 30 weeks x 40 rows, minus the newest (unlabeled) week
    assert len(train) + len(holdout) == 29 * 40
    assert {*FEATURE_COLUMNS, LABEL} <= set(train.columns)
    assert 0 < saved["baseline_train"]["pr_auc"] <= 1


def test_prepare_stops_on_invalid_data(tmp_path):
    bad = make_raw(n_weeks=30)
    bad.loc[0, "weekly_rank"] = 11
    path = tmp_path / "2026-11-09_global_weekly.xlsx"
    bad.to_excel(path, index=False)

    out = tmp_path / "processed"
    with pytest.raises(SchemaErrors):
        prepare(raw_path=path, processed_dir=out, holdout_weeks=5)
    assert not out.exists()  # nothing saved from bad data


def test_baseline_rule():
    df = pd.DataFrame({"weekly_rank": [1, 2, 6, 9], LABEL: [1, 1, 0, 1]})
    b = rank_baseline(df)
    assert b["rule_precision"] == 1.0  # both "rank <= 5" titles survived
    assert b["rule_recall"] == pytest.approx(2 / 3, abs=1e-4)  # missed the rank-9 survivor
    assert b["positive_rate"] == 0.75
