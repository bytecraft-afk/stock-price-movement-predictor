"""
main.py
=======
Stock Price Movement Predictor — Pipeline Orchestrator
======================================================

Runs the complete pipeline top-to-bottom with a single command:
    python main.py

Pipeline stages
---------------
1. Load / cache SPY OHLCV data  (yfinance → data/SPY_raw.csv)
2. Build labels + features       (no-leakage label construction + indicators)
3. Chronological train/test split + StandardScaler (fit on train only)
4. Evaluate two naive baselines  (persistence, majority class)
5. Train and evaluate 4 models   (LR, RF × raw, engineered features)
6. Print results table + save plot to outputs/

Reproducibility
---------------
RANDOM_SEED = 42 is passed to every stochastic component.
numpy.random.seed(42) is set globally at startup.
The data is cached to disk so yfinance is not called on repeat runs.

Usage
-----
    pip install -r requirements.txt
    python main.py
"""

import numpy as np

# ── Global random seed — must be set before any sklearn/numpy calls ──────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

from src.data_loader import load_data
from src.features    import build_all_features
from src.split_scale import split_and_scale
from src.baselines   import persistence_baseline, majority_class_baseline
from src.models      import run_all_models
from src.visualize   import build_results_table, plot_predictions

# ── Configuration ─────────────────────────────────────────────────────────────
TICKER      = "SPY"
START_DATE  = "2019-01-01"
END_DATE    = "2024-12-31"
TRAIN_RATIO = 0.80          # 80% train, 20% test


def main() -> None:
    banner = "=" * 70
    print(f"\n{banner}")
    print("  STOCK PRICE MOVEMENT PREDICTOR")
    print(f"  Ticker: {TICKER}  |  {START_DATE} → {END_DATE}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(banner)

    # ── Stage 1: Data ─────────────────────────────────────────────────────
    print("\n[1/6] Loading OHLCV data ...")
    df = load_data(ticker=TICKER, start=START_DATE, end=END_DATE)

    # ── Stage 2: Labels + Features ────────────────────────────────────────
    print("\n[2/6] Constructing labels and features ...")
    df_labeled, X_raw, X_eng, y = build_all_features(df)

    print(f"\n  Raw feature columns ({len(X_raw.columns)}):        {list(X_raw.columns)}")
    print(f"  Engineered feature columns ({len(X_eng.columns)}): {list(X_eng.columns)}")

    # ── Stage 3: Split + Scale ────────────────────────────────────────────
    print("\n[3/6] Chronological split + feature scaling ...")

    print("\n  ── Raw features ──")
    splits_raw = split_and_scale(X_raw, y, train_ratio=TRAIN_RATIO, label="raw")

    print("\n  ── Engineered features ──")
    splits_eng = split_and_scale(X_eng, y, train_ratio=TRAIN_RATIO, label="eng")

    # y_train / y_test are identical for both splits (same rows, same labels)
    y_train = splits_raw["y_train"]
    y_test  = splits_raw["y_test"]

    # ── Stage 4: Baselines ────────────────────────────────────────────────
    print("\n[4/6] Evaluating naive baselines ...")
    _, b1_metrics = persistence_baseline(y_test,  y_train)
    _, b2_metrics = majority_class_baseline(y_test, y_train)

    baseline_results = [
        ("Baseline 1 — Persistence",    b1_metrics),
        ("Baseline 2 — Majority Class", b2_metrics),
    ]

    # ── Stage 5: ML Models ────────────────────────────────────────────────
    print("\n[5/6] Training and evaluating ML models ...")
    model_results = run_all_models(splits_raw, splits_eng)

    # ── Stage 6: Results + Plot ───────────────────────────────────────────
    print("\n[6/6] Building results table and saving plot ...")
    build_results_table(baseline_results, model_results)

    plot_predictions(
        df_labeled=df_labeled,
        test_idx=splits_eng["X_test_idx"],
        y_test=y_test,
        model_results=model_results,
    )

    print("\n✓ Pipeline complete.")
    print("  outputs/results_table.csv    — metrics for all 6 models")
    print("  outputs/predictions_plot.png — actual vs predicted direction\n")


if __name__ == "__main__":
    main()
