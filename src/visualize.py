"""
visualize.py
============
Results comparison table and prediction plot.

Outputs
-------
outputs/results_table.csv   — 6-row × 4-metric comparison DataFrame
outputs/predictions_plot.png — test-period actual vs predicted direction
"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SRC_DIR)
OUTPUT_DIR  = os.path.join(_ROOT_DIR, "outputs")


# ──────────────────────────────────────────────────────────────────────────────
# Results Table
# ──────────────────────────────────────────────────────────────────────────────

def build_results_table(
    baseline_results: list,
    model_results: list,
) -> pd.DataFrame:
    """
    Build and print the full comparison DataFrame.

    Rows (in order):
        Baseline 1 — Persistence
        Baseline 2 — Majority Class
        Raw + LogisticRegression
        Raw + RandomForest
        Engineered + LogisticRegression
        Engineered + RandomForest

    Columns: Accuracy, Precision, Recall, F1

    Parameters
    ----------
    baseline_results : list of (name: str, metrics: dict)
    model_results    : list of result dicts from models.run_all_models()

    Returns
    -------
    pd.DataFrame (indexed by model name)
    """
    rows = []

    for name, m in baseline_results:
        rows.append({
            "Model":     name,
            "Accuracy":  round(m["Accuracy"],  4),
            "Precision": round(m["Precision"], 4),
            "Recall":    round(m["Recall"],    4),
            "F1":        round(m["F1"],        4),
        })

    for r in model_results:
        rows.append({
            "Model":     r["label"],
            "Accuracy":  round(r["Accuracy"],  4),
            "Precision": round(r["Precision"], 4),
            "Recall":    round(r["Recall"],    4),
            "F1":        round(r["F1"],        4),
        })

    df = pd.DataFrame(rows).set_index("Model")

    # Pretty-print to console
    separator = "=" * 72
    print(f"\n{separator}")
    print("  RESULTS TABLE")
    print(separator)
    print(df.to_string())
    print(separator + "\n")

    # Save to CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "results_table.csv")
    df.to_csv(csv_path)
    print(f"[visualize] Results table saved → {csv_path}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Prediction Plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_predictions(
    df_labeled: pd.DataFrame,
    test_idx: pd.DatetimeIndex,
    y_test: pd.Series,
    model_results: list,
) -> None:
    """
    Two-panel plot of actual vs predicted direction over the test period.

    Panel 1 (top): SPY closing price with ▲/▼ markers for actual next-day direction.
    Panel 2 (bottom): Step chart comparing actual vs best-model predicted direction.

    The "best engineered model" is selected by highest F1 score.

    Parameters
    ----------
    df_labeled   : full labeled DataFrame (contains Close prices)
    test_idx     : DatetimeIndex of test rows
    y_test       : actual test labels
    model_results: list of result dicts from models.run_all_models()
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Pick best engineered model by F1 ─────────────────────────────────
    eng_results = [r for r in model_results if r["feature_set"] == "Engineered"]
    best        = max(eng_results, key=lambda r: r["F1"])
    y_pred      = best["y_pred"]
    y_actual    = y_test.values
    label       = best["label"]

    # ── Price series for the test period ──────────────────────────────────
    close_test = df_labeled.loc[test_idx, "Close"]
    dates      = close_test.index

    # ── Figure setup ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )
    fig.patch.set_facecolor("#f8f9fa")
    for ax in axes:
        ax.set_facecolor("#f8f9fa")

    fig.suptitle(
        f"SPY — Next-Day Direction Prediction  |  Test Period\n"
        f"Best Model: {label}  "
        f"(Acc={best['Accuracy']:.3f}, F1={best['F1']:.3f})",
        fontsize=12, y=0.99, fontweight="bold",
    )

    # ── Panel 1: Price + actual direction markers ──────────────────────────
    ax1 = axes[0]
    ax1.plot(dates, close_test.values, color="#2c7bb6", lw=1.4,
             label="SPY Close", zorder=2)

    up_mask   = y_actual == 1
    down_mask = y_actual == 0
    ax1.scatter(
        dates[up_mask], close_test.values[up_mask],
        marker="^", color="#1a9641", s=25, alpha=0.7,
        label="Actual: UP next day", zorder=3,
    )
    ax1.scatter(
        dates[down_mask], close_test.values[down_mask],
        marker="v", color="#d7191c", s=25, alpha=0.7,
        label="Actual: DOWN next day", zorder=3,
    )
    ax1.set_ylabel("Close Price (USD)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.7)
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.set_title("Actual Next-Day Direction vs. Price", fontsize=10, pad=4)

    # ── Panel 2: Actual vs Predicted direction (step plot) ─────────────────
    ax2 = axes[1]
    ax2.step(dates, y_actual, where="post", color="#2c7bb6", lw=1.8,
             label="Actual Direction", alpha=0.85)
    ax2.step(dates, y_pred, where="post", color="#fd8d3c", lw=1.6,
             label=f"Predicted ({label})", alpha=0.85, linestyle="--")

    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["DOWN (0)", "UP (1)"], fontsize=9)
    ax2.set_ylabel("Direction", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.7)
    ax2.grid(True, alpha=0.25, linestyle="--")
    ax2.set_title("Actual vs Predicted Direction", fontsize=10, pad=4)

    # ── X-axis formatting ──────────────────────────────────────────────────
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=40, ha="right", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plot_path = os.path.join(OUTPUT_DIR, "predictions_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Plot saved → {plot_path}")
