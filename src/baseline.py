"""Baseline: the simple rank rule every model has to beat.

Rule: a title ranked 5 or better this week survives to next week.
For PR-AUC, the rule's ranking score is ``11 - rank`` (rank 1 scores highest).

Phase 3's registration gate compares the model's PR-AUC against ``pr_auc`` here,
computed on the training weeks only (the holdout stays untouched).
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

from src.features import LABEL

RULE_MAX_RANK = 5


def rank_score(df: pd.DataFrame) -> pd.Series:
    """Baseline survival score: higher for better ranks."""
    return 11 - df["weekly_rank"]


def rank_baseline(df: pd.DataFrame) -> dict:
    """PR-AUC of the rank score, plus precision/recall of the "rank <= 5 survives" rule."""
    y = df[LABEL].astype(int)
    rule = (df["weekly_rank"] <= RULE_MAX_RANK).astype(int)
    return {
        "pr_auc": round(float(average_precision_score(y, rank_score(df))), 4),
        "rule_precision": round(float(precision_score(y, rule)), 4),
        "rule_recall": round(float(recall_score(y, rule)), 4),
        "positive_rate": round(float(y.mean()), 4),  # PR-AUC of random guessing
        "n_rows": len(df),
    }
