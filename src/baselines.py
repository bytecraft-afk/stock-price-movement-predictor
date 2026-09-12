"""
baselines.py
============
Two naive baseline predictors used as benchmarks for the ML models.

Baseline 1 — Persistence (momentum):
    pred[t] = actual direction at t-1
    Rationale: if markets trend, yesterday's direction predicts today.

Baseline 2 — Majority Class:
    pred[t] = majority class from the TRAINING set, for every test row.
    The majority class is determined from y_train ONLY — no test leakage.
    This is the "dumbest possible classifier" benchmark.

Both baselines are evaluated on the TEST set with accuracy and F1.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute accuracy, precision, recall, and macro-F1."""
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
    }


def persistence_baseline(
    y_test: pd.Series,
    y_train: pd.Series,
) -> tuple:
    """
    Baseline 1: Persistence — predict tomorrow = same direction as today.

    For each test row t, the prediction is the ACTUAL direction at t-1.
    For the FIRST test row, "yesterday" is the last row of the training set.

    Implementation
    --------------
    We prepend the last training label to the test labels, then shift by 1.
    This correctly propagates the training/test boundary without needing
    the raw price series.

    Parameters
    ----------
    y_test  : test-set labels (pd.Series with DatetimeIndex)
    y_train : training-set labels (used only for the first test row)

    Returns
    -------
    y_pred   : np.ndarray of predictions (same length as y_test)
    metrics  : dict with Accuracy, Precision, Recall, F1
    """
    # Prepend the last training label so the first test prediction has a "yesterday"
    y_chain = pd.concat([y_train.iloc[[-1]], y_test])  # length = len(y_test) + 1

    # shift(1): each position gets the value that was one position earlier
    # After dropna(), length = len(y_test)
    y_pred = y_chain.shift(1).dropna().values.astype(int)

    m = _metrics(y_test.values, y_pred)
    print(
        f"  Baseline 1 — Persistence     "
        f"Acc={m['Accuracy']:.4f}  Prec={m['Precision']:.4f}  "
        f"Rec={m['Recall']:.4f}  F1={m['F1']:.4f}"
    )
    return y_pred, m


def majority_class_baseline(
    y_test: pd.Series,
    y_train: pd.Series,
) -> tuple:
    """
    Baseline 2: Always predict the majority class found in the training set.

    LEAKAGE-SAFE: majority class is derived from y_train only.
    The test set distribution is never inspected when forming predictions.

    Parameters
    ----------
    y_test  : test-set labels
    y_train : training-set labels (used to determine majority class)

    Returns
    -------
    y_pred   : np.ndarray of constant predictions (same length as y_test)
    metrics  : dict with Accuracy, Precision, Recall, F1
    """
    majority_class = int(y_train.mode().iloc[0])   # from training set only
    direction_str  = "UP (1)" if majority_class == 1 else "DOWN (0)"
    print(f"  [Majority class from train set: {direction_str}]")

    y_pred = np.full(len(y_test), majority_class, dtype=int)

    m = _metrics(y_test.values, y_pred)
    print(
        f"  Baseline 2 — Majority Class  "
        f"Acc={m['Accuracy']:.4f}  Prec={m['Precision']:.4f}  "
        f"Rec={m['Recall']:.4f}  F1={m['F1']:.4f}"
    )
    return y_pred, m
