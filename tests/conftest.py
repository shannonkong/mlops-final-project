"""Shared test data: a small synthetic dataset shaped like Netflix's global weekly file.

It follows the same rules as the real data, so it passes validation: Sunday weeks,
4 lists x 10 ranks per week, hours falling with rank, and runtime/views blank
before the June 2023 metric switch.
"""

import numpy as np
import pandas as pd
import pytest

CATEGORIES = ["Films (English)", "Films (Non-English)", "TV (English)", "TV (Non-English)"]
SWITCH = pd.Timestamp("2023-06-18")


def make_raw(n_weeks: int = 30, start: str = "2023-04-02", seed: int = 0) -> pd.DataFrame:
    """Synthetic raw data in the shape ``load_raw`` returns. ``start`` must be a Sunday."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(start, periods=n_weeks, freq="7D")
    rows, next_id = [], 0
    for category in CATEGORIES:
        is_tv = category.startswith("TV")
        active: dict[str, int] = {}  # title -> weeks on chart so far
        for week in weeks:
            # Each title stays with a probability that falls as it ages; new titles fill the gaps.
            kept = {t: age for t, age in active.items() if rng.random() < max(0.2, 0.8 - 0.1 * age)}
            while len(kept) < 10:
                kept[f"{category[:5]} title {next_id}"] = 0
                next_id += 1
            kept = dict(list(kept.items())[:10])
            hours = {t: float(rng.lognormal(15 - 0.2 * age, 0.6)) for t, age in kept.items()}
            for rank, title in enumerate(sorted(hours, key=hours.get, reverse=True), start=1):
                after = week >= SWITCH
                runtime = (8.0 if is_tv else 2.0) if after else np.nan
                rows.append(
                    {
                        "week": week,
                        "category": category,
                        "weekly_rank": rank,
                        "show_title": title,
                        "season_title": f"{title}: Season 1" if is_tv else np.nan,
                        "weekly_hours_viewed": round(hours[title]),
                        "runtime": runtime,
                        "weekly_views": round(hours[title] / runtime, -5) if after else np.nan,
                        "cumulative_weeks_in_top_10": kept[title] + 1,
                    }
                )
            active = {t: age + 1 for t, age in kept.items()}
    return pd.DataFrame(rows)


@pytest.fixture
def raw_df():
    return make_raw()
