"""
split_scale.py
==============
Chronological train/test split and StandardScaler fitting.

═══════════════════════════════════════════════════════════════════
LEAKAGE PREVENTION — CRITICAL STEPS IN THIS MODULE
═══════════════════════════════════════════════════════════════════
Split rule:
    We sort rows by date (already guaranteed by the data loader) and
    use the first TRAIN_RATIO fraction as training data, the rest as
    test data.  No shuffling — shuffling would mix future rows into
    training and past rows into testing, destroying temporal causality.

Scaler rule:
    StandardScaler.fit() is called ONLY on X_train.
    The scaler learns μ (mean) and σ (std) from TRAINING rows only.
    Both X_train and X_test are then transformed using those training
    statistics.  If we fitted on the full dataset or on X_test, the
    model would indirectly "see" test-set statistics during training —
    a subtle but real form of data leakage.
═══════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.80,
):
    """
    Split features and labels using a strict chronological cut.

    Parameters
    ----------
    X : pd.DataFrame  — feature matrix (DatetimeIndex, sorted ascending)
    y : pd.Series     — target series (same index as X)
    train_ratio : float — fraction of rows used for training

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    n         = len(X)
    split_idx = int(n * train_ratio)   # integer cut point

    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]

    print(
        f"[split] Train : {X_train.index[0].date()} → {X_train.index[-1].date()} "
        f"({len(X_train)} rows, {train_ratio*100:.0f}%)\n"
        f"[split] Test  : {X_test.index[0].date()} → {X_test.index[-1].date()} "
        f"({len(X_test)} rows, {(1-train_ratio)*100:.0f}%)"
    )
    return X_train, X_test, y_train, y_test


def _report_class_balance(y_train: pd.Series, y_test: pd.Series, label: str) -> None:
    """Print % UP / % DOWN for train and test sets separately."""
    def _fmt(y, split_name):
        up   = int(y.sum())
        down = len(y) - up
        print(
            f"[balance|{label}] {split_name:<6}: "
            f"{up} UP ({up/len(y)*100:.1f}%) | "
            f"{down} DOWN ({down/len(y)*100:.1f}%)"
        )
    _fmt(y_train, "Train")
    _fmt(y_test,  "Test")


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Fit StandardScaler on X_train ONLY, then transform both sets.

    LEAKAGE PREVENTION (repeated for clarity):
        scaler.fit(X_train)          ← learns μ, σ from training data only
        scaler.transform(X_train)    ← scales training set
        scaler.transform(X_test)     ← scales test set using TRAINING statistics
                                        (no test information seen by scaler)

    Returns
    -------
    X_train_scaled : np.ndarray
    X_test_scaled  : np.ndarray
    scaler         : fitted StandardScaler instance
    """
    scaler = StandardScaler()

    # ── FIT: only on training data ─────────────────────────────────────────
    scaler.fit(X_train)

    # ── TRANSFORM: apply training statistics to both sets ─────────────────
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.80,
    label: str = "features",
) -> dict:
    """
    Convenience wrapper: chronological split → class balance report → scale.

    Returns
    -------
    dict with keys:
        X_train, X_test  : np.ndarray (scaled)
        y_train, y_test  : pd.Series
        scaler           : fitted StandardScaler
        X_train_idx      : pd.DatetimeIndex of training rows
        X_test_idx       : pd.DatetimeIndex of test rows
    """
    X_train, X_test, y_train, y_test = chronological_split(X, y, train_ratio)
    _report_class_balance(y_train, y_test, label)
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    return {
        "X_train":     X_train_sc,
        "X_test":      X_test_sc,
        "y_train":     y_train,
        "y_test":      y_test,
        "scaler":      scaler,
        "X_train_idx": X_train.index,
        "X_test_idx":  X_test.index,
    }
