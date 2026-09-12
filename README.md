# Stock Price Movement Predictor

A machine learning project that predicts the **next-day price direction (up/down)** of SPY (S&P 500 ETF). Two ML models trained on different feature sets are benchmarked against two naive baselines using a rigorous, leakage-free time-series methodology.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [How to Run](#how-to-run)
3. [Project Structure](#project-structure)
4. [Data Source](#data-source)
5. [Feature List](#feature-list)
6. [Methodology — No-Leakage Design](#methodology--no-leakage-design)
7. [Results](#results)
8. [Honest Interpretation](#honest-interpretation)

---

## Project Overview

**Task**: Binary classification — predict whether SPY's closing price on day `t+1` will be *higher* (`1`) or *lower/equal* (`0`) than its closing price on day `t`.

**Models compared**:

| # | Model | Feature Set |
|---|-------|-------------|
| 1 | Baseline: Persistence | — |
| 2 | Baseline: Majority Class | — |
| 3 | Logistic Regression | Raw OHLCV |
| 4 | Random Forest | Raw OHLCV |
| 5 | Logistic Regression | Engineered (indicators + lags) |
| 6 | Random Forest | Engineered (indicators + lags) |

**Tech stack**: Python · pandas · scikit-learn · matplotlib · yfinance
*(The `ta` library is listed as a dependency but all indicators are hand-computed in pandas as a learning exercise.)*

---

## How to Run

```bash
# 1. Clone or download the project
cd "Stock price aiml project"

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Run the pipeline
python main.py
```

**Expected outputs**:
- `data/SPY_raw.csv` — cached OHLCV data (auto-downloaded on first run)
- `outputs/results_table.csv` — 6-row × 4-metric comparison table
- `outputs/predictions_plot.png` — actual vs predicted direction chart

The full pipeline takes ~30 seconds on first run (downloads data) and ~5 seconds on subsequent runs (uses cache).

---

## Project Structure

```
Stock price aiml project/
├── data/
│   └── SPY_raw.csv             ← cached OHLCV (auto-created)
├── outputs/
│   ├── results_table.csv       ← metric comparison table
│   └── predictions_plot.png    ← actual vs predicted plot
├── src/
│   ├── __init__.py
│   ├── data_loader.py          ← yfinance fetch + local cache
│   ├── features.py             ← label construction + feature engineering
│   ├── split_scale.py          ← chronological split + StandardScaler
│   ├── baselines.py            ← persistence + majority-class baselines
│   ├── models.py               ← LogisticRegression + RandomForest
│   └── visualize.py            ← results table + matplotlib plot
├── main.py                     ← pipeline orchestrator
├── requirements.txt            ← pinned dependencies
└── README.md                   ← this file
```

---

## Data Source

- **Ticker**: SPY (SPDR S&P 500 ETF Trust) — one of the most liquid instruments in the world
- **Provider**: [yfinance](https://pypi.org/project/yfinance/)
- **Date range**: 2019-01-01 → 2024-12-31 (~1,500 trading days)
- **Adjustment**: `auto_adjust=True` (prices adjusted for splits and dividends)
- **Columns**: Open, High, Low, Close, Volume
- **Cache**: Saved to `data/SPY_raw.csv` after first download for reproducibility

---

## Feature List

### Raw Feature Set (5 features)
Directly observable market data — no transformation beyond StandardScaler:

| Feature | Description |
|---------|-------------|
| `Open`  | Opening price |
| `High`  | Daily high |
| `Low`   | Daily low |
| `Close` | Closing price |
| `Volume`| Trading volume |

> **Note**: Price levels are non-stationary. The raw feature set is intentionally naive to serve as a fair comparison baseline for the engineered set.

### Engineered Feature Set (13 features)
Hand-computed technical indicators and lag returns — all strictly causal (no future data):

| Feature | Description | Leakage note |
|---------|-------------|--------------|
| `rsi_14` | RSI (14-period, Wilder's EMA) | `diff()` and `ewm()` are backward-only |
| `macd_line` | EMA(12) − EMA(26) | EWMs use past data only |
| `macd_signal` | EMA(9) of MACD line | Same |
| `macd_hist` | MACD line − signal | Same |
| `bb_width` | (Upper − Lower) / SMA(20) | `rolling(20)` looks back |
| `bb_pct` | %B: price position in Bollinger Bands | Same |
| `rolling_vol_10` | 10-day rolling std of log returns | `rolling(10).std()` backward |
| `ema_ratio_20` | Close / EMA(20) | EWM backward |
| `return_lag_1` | Log return 1 day ago | `shift(1)` = yesterday |
| `return_lag_2` | Log return 2 days ago | `shift(2)` |
| `return_lag_5` | Log return 5 days ago | `shift(5)` |
| `return_lag_10` | Log return 10 days ago | `shift(10)` |

---

## Methodology — No-Leakage Design

This section documents every step taken to ensure zero data leakage between train and test sets.

### 1. Label Construction
```python
# target[t] = 1 if close[t+1] > close[t] else 0
next_close = df["Close"].shift(-1)          # tomorrow's close in today's row
df["target"] = (next_close > df["Close"]).astype(int)
df.dropna(subset=["target"], inplace=True)  # drop last row (no future label)
```
`shift(-1)` is the only mechanism that accesses tomorrow's price — and it is used **only to create the label (y), never as an input feature**. The final row is immediately dropped.

### 2. Rolling Features Are Causal
All `rolling(window=W)` calls at row `t` aggregate rows `[t-W+1 … t]`. All `ewm(span=S, adjust=False)` calls decay from the beginning up to row `t`. No `center=True` or forward-fill operations are used.

### 3. Lag Returns Are Explicitly Shifted
```python
feats[f"return_lag_{lag}"] = log_ret.shift(lag)  # lag=1: yesterday's return
```
`shift(lag)` at row `t` yields `log_ret[t-lag]` — strictly past data.

### 4. Warm-Up NaN Rows Dropped Before Splitting
The first ~26 rows have NaN values (MACD slow EMA needs 26 bars). These are dropped **before** the chronological split. This is leakage-safe because they all occur at the very start of the dataset — entirely within the training window under any reasonable (≥70%) train ratio.

### 5. Chronological Train/Test Split — No Shuffling
```
Train: 2019-01-xx → ~2023-06-xx  (first 80% of rows)
Test : ~2023-06-xx → 2024-12-31  (last 20% of rows)
```
No `shuffle=True`, no `random_state` on the splitter — the temporal order is never disturbed.

### 6. StandardScaler Fitted on Train Set Only
```python
scaler = StandardScaler()
scaler.fit(X_train)             # learns μ and σ from training data ONLY
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # uses training μ and σ
```
Fitting the scaler on test data would reveal its distribution to the model — a subtle but real leakage.

### Leakage Prevention Summary Table

| Risk | Prevention |
|------|-----------|
| Label uses future close | `shift(-1)` + immediate `dropna()` on target |
| Scaler sees test data | `scaler.fit(X_train)` only |
| Temporal mixing | Chronological 80/20 split, no shuffle |
| Rolling features use future | All `rolling()` / `ewm()` are backward-only |
| Lag features use today's return | `shift(lag)` with lag ≥ 1 |
| Warm-up NaN rows | Dropped from start of series (pre-split) |

---

## Results

> Run `python main.py` to reproduce. Results below reflect SPY 2019–2024 with seed 42.

```
### Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Baseline 1 — Persistence | 54.70% | 62.43% | 62.78% | 62.60% |
| Baseline 2 — Majority Class | 60.40% | 60.40% | 100.00% | 75.31% |
| Raw + Logistic Regression | 41.28% | 66.67% | 5.56% | 10.26% |
| Raw + Random Forest | 41.95% | 68.42% | 7.22% | 13.07% |
| Engineered + Logistic Regression | 60.40% | 61.40% | 92.78% | 73.89% |
| Engineered + Random Forest | 56.38% | 59.92% | 83.89% | 69.91% |

### Interpretation

The engineered Logistic Regression achieved 60.40% accuracy, which exactly matches the majority-class baseline. The engineered Random Forest achieved 56.38%, which was below the majority baseline.

Therefore, the engineered features did not demonstrate a meaningful predictive advantage over the naive baselines on the held-out test period. This is an honest and expected outcome for next-day stock direction prediction, which is a difficult problem and can behave close to a coin flip.

The raw-price models performed substantially worse than the baselines, with both models predicting very few UP days.

Training and test performance were also compared to check for overfitting. The Random Forest models had higher training accuracy than test accuracy, indicating some degree of overfitting, while the constrained tree depth and minimum leaf size helped limit it.
```

---

## Honest Interpretation

### Does the engineered model beat the baselines?

In practice, any outperformance over the naive baselines is **small and may not be statistically significant**. A realistic expectation:

- **Baselines**: ~52–54% accuracy (markets are slightly trending on average)
- **ML models**: ~51–56% accuracy — barely better, sometimes worse

The engineered feature model may beat the persistence baseline by 1–3 percentage points. This is the expected outcome, not a failure of the methodology.

### Next-Day Direction Is Close to a Coin Flip

SPY daily price direction is approximately 53% UP / 47% DOWN historically. This means:

- A model that **always predicts UP** achieves ~53% accuracy with zero market insight.
- The Efficient Market Hypothesis (EMH) posits that all publicly available information is already priced in — technical indicators derived from price history are public information and should have limited predictive power.
- Any signal embedded in RSI, MACD, or Bollinger Bands has been extensively arbitraged by professional traders using far more sophisticated models.

**Do not interpret 53% accuracy as evidence of a profitable trading strategy.** Transaction costs, slippage, and bid-ask spreads would typically eliminate such small edges.

### Signs of Overfitting

Random Forest with its many decision trees is prone to memorising the training set:

- **If train accuracy >> test accuracy** (e.g., 70% train vs 52% test), the model is overfitting.
- We mitigate this with `max_depth=5` and `min_samples_leaf=20` constraints.
- Logistic Regression (a linear model) typically shows much smaller train/test gaps.

Logistic Regression's simplicity makes it a more honest baseline: if it beats Random Forest on the test set, it suggests the data genuinely lacks complex non-linear patterns — consistent with EMH.

### What This Project Demonstrates (Correctly)

1. **Proper time-series ML methodology**: chronological split, no-leakage scaler, causal features.
2. **Realistic expectations**: near-baseline performance is the honest outcome for next-day direction.
3. **Benchmark value**: the persistence and majority-class baselines are non-trivial — any model must clearly beat both to be considered useful.
4. **Feature engineering discipline**: hand-computing indicators forces understanding of what each indicator measures and when it would introduce lookahead bias.

---

*Random seed: 42. Data: SPY daily OHLCV from yfinance (2019–2024). All indicators hand-computed in pandas.*
