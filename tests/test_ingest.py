"""Tests for src/ingest.py. They build small Excel files in a temp folder, so no real data is needed."""

from datetime import date

import pandas as pd
import pytest

from src import ingest


def make_xlsx(path, title_row=False, columns=None):
    """Write a tiny Excel file shaped like Netflix's (Title Case headings, optional title row on top)."""
    df = pd.DataFrame(
        {
            "Week": ["2026-10-11", "2026-10-11", "2026-10-18"],
            "Category": ["Films (English)", "TV (English)", "Films (English)"],
            "Weekly Rank": [1, 1, 2],
            "Show Title": ["Film A", "Show B", "Film A"],
            "Season Title": [None, "Show B: Season 1", None],
            "Weekly Hours Viewed": [5_000_000, 9_000_000, 3_000_000],
            "Runtime": [2.0, 8.5, 2.0],
            "Weekly Views": [2_500_000, 1_100_000, 1_500_000],
            "Cumulative Weeks in Top 10": [1, 1, 2],
        }
    )
    if columns is not None:
        df = df[columns]
    with pd.ExcelWriter(path) as writer:
        if title_row:
            pd.DataFrame([["Netflix Top 10 - global weekly"]]).to_excel(writer, header=False, index=False)
            df.to_excel(writer, index=False, startrow=2)
        else:
            df.to_excel(writer, index=False)
    return path


def test_snapshot_name_format():
    assert ingest.snapshot_name(date(2026, 10, 23)) == "2026-10-23_global_weekly.xlsx"


def test_latest_snapshot_picks_newest_and_skips_lock_files(tmp_path):
    for name in ["2026-10-16_global_weekly.xlsx", "2026-10-23_global_weekly.xlsx"]:
        make_xlsx(tmp_path / name)
    (tmp_path / "~$2026-10-30_global_weekly.xlsx").write_bytes(b"lock")  # Excel lock file
    assert ingest.latest_snapshot(tmp_path).name == "2026-10-23_global_weekly.xlsx"


def test_latest_snapshot_errors_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest.latest_snapshot(tmp_path)


@pytest.mark.parametrize("title_row", [False, True])
def test_load_raw_standardizes_columns(tmp_path, title_row):
    df = ingest.load_raw(make_xlsx(tmp_path / "x_global_weekly.xlsx", title_row=title_row))
    for column in ingest.REQUIRED_COLUMNS + ingest.OPTIONAL_COLUMNS:
        assert column in df.columns
    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df["week"])


def test_load_raw_adds_missing_optional_columns(tmp_path):
    keep = ["Week", "Category", "Weekly Rank", "Show Title", "Weekly Hours Viewed", "Cumulative Weeks in Top 10"]
    df = ingest.load_raw(make_xlsx(tmp_path / "x_global_weekly.xlsx", columns=keep))
    assert df["weekly_views"].isna().all()


def test_load_raw_errors_on_missing_required_column(tmp_path):
    keep = ["Week", "Category", "Show Title", "Weekly Hours Viewed", "Cumulative Weeks in Top 10"]  # no rank
    with pytest.raises(KeyError, match="weekly_rank"):
        ingest.load_raw(make_xlsx(tmp_path / "x_global_weekly.xlsx", columns=keep))


def test_ingest_copies_local_file_with_dated_name(tmp_path):
    source = make_xlsx(tmp_path / "downloaded.xlsx")
    raw_dir = tmp_path / "raw"
    path = ingest.ingest(source=source, raw_dir=raw_dir, snapshot_date=date(2026, 10, 23))
    assert path == raw_dir / "2026-10-23_global_weekly.xlsx"
    assert path.exists()


def test_ingest_reuses_existing_snapshot(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    existing = make_xlsx(raw_dir / "2026-10-23_global_weekly.xlsx")
    # No source and no network needed: the existing snapshot is reused.
    assert ingest.ingest(raw_dir=raw_dir, snapshot_date=date(2026, 10, 23)) == existing


def test_download_rejects_non_excel_response(tmp_path, monkeypatch):
    class FakeResponse:
        content = b"<html>not a spreadsheet</html>"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(ingest.requests, "get", lambda *args, **kwargs: FakeResponse())
    with pytest.raises(ValueError, match="did not return an .xlsx"):
        ingest.download("https://example.com/file.xlsx", tmp_path / "out.xlsx")
    assert not (tmp_path / "out.xlsx").exists()
