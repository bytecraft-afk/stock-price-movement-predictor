"""
features.py
===========
Label construction and feature engineering for next-day direction prediction.

═══════════════════════════════════════════════════════════════════
LEAKAGE PREVENTION — KEY RULES APPLIED IN THIS MODULE
═══════════════════════════════════════════════════════════════════
Rule 1 — Label construction:
    target[t] = 1 if close[t+1] > close[t] else 0
    We use .shift(-1) to bring tomorrow's close into today's row.
    The LAST ROW is immediately dropped (it has no future close → NaN).
    The target column is NEVER used as an input feature.

Rule 2 — Technical indicators use only past data:
    All rolling() and ewm() calls look backward only.
    At row t, rolling(window=W).mean() averages rows [t-W+1 … t].
    ewm(span=S, adjust=False) decays from the beginning up to row t.
    No future information enters any indicator.

Rule 3 — Lag features are explicitly shifted:
    return_lag_1[t] = log_return[t-1]  (yesterday's return)
    return_lag_k[t] = log_return[t-k]  (k days ago)
    The shift(k) guarantees we never use today's return as a predictor
    — it is already the target-adjacent signal.

Rule 4 — NaN warm-up rows are dropped BEFORE the train/test split:
    The first ~26 rows have NaN indicators (MACD slow EMA needs 26 bars).
    Dropping them before splitting is leakage-safe because they all
    occur at the very START of the dataset — well inside the training
    period under any reasonable (≥ 70 %) train ratio.
═══════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Label Construction
# ──────────────────────────────────────────────────────────────────────────────

def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the next-day direction label to every row.

    Formula: target[t] = 1 if close[t+1] > close[t] else 0

    .shift(-1) moves tomorrow's Close into today's row.
    The final row has no next-day Close, so its label is removed.

    NO LEAKAGE: the target is only an output, never an input feature.
    """
    df = df.copy()

    # shift(-1): tomorrow's close slides into today's row
    next_close = df["Close"].shift(-1)

    # target[t] = 1 if close[t+1] > close[t] else 0
    # Keep the final value as NaN because there is no next-day close.
    target = (next_close > df["Close"]).where(next_close.notna())

    n_before = len(df)

    # Drop the final row because it has no valid future label.
    df["target"] = target
    df.dropna(subset=["target"], inplace=True)

    # Convert valid labels to 0/1.
    df["target"] = df["target"].astype(int)

    n_dropped = n_before - len(df)

    print(
        f"[features] Label construction: dropped {n_dropped} trailing row(s) "
        f"(no future close available)."
    )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Hand-Computed Technical Indicators  (all strictly causal)
# ──────────────────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index via Wilder's exponential smoothing.

    Wilder's EMA uses alpha = 1/period (equivalent to span = 2*period - 1).
    diff() looks backward: delta[t] = close[t] - close[t-1].
    ewm(adjust=False) recursively weights past values — no future data used.
    """
    delta    = close.diff()                              # backward difference
    gain     = delta.clip(lower=0)                       # positive moves
    loss     = (-delta).clip(lower=0)                    # negative moves (abs)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD = EMA(fast) − EMA(slow); Signal = EMA(MACD, signal).

    All EWMs are backward-looking: ewm(span=N, adjust=False) uses only
    data from the beginning of the series up to the current row.
    """
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return pd.DataFrame(
        {"macd_line": macd_line, "macd_signal": signal_line, "macd_hist": histogram},
        index=close.index,
    )


def _bollinger_bands(close: pd.Series, period: int = 20) -> pd.DataFrame:
    """
    Bollinger Bands: SMA(period) ± 2 × rolling std(period).

    rolling(window=period) at row t uses rows [t-period+1 … t] — no future.

    Returns two normalised features:
        bb_width : (upper - lower) / sma  — measures volatility
        bb_pct   : (close - lower) / (upper - lower)  — price position (0–1)
    """
    sma   = close.rolling(window=period).mean()
    std   = close.rolling(window=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std

    bb_width = (upper - lower) / sma              # relative band width
    bb_pct   = (close - lower) / (upper - lower)  # %B: position within bands
    return pd.DataFrame(
        {"bb_width": bb_width, "bb_pct": bb_pct},
        index=close.index,
    )


def _rolling_volatility(close: pd.Series, period: int = 10) -> pd.Series:
    """
    Historical volatility: rolling std of log returns over `period` days.

    log_return[t] = log(close[t] / close[t-1]) — uses only past close.
    rolling(period).std() aggregates the past `period` log returns.
    """
    log_ret = np.log(close / close.shift(1))      # backward log return
    return log_ret.rolling(window=period).std()


def _ema_ratio(close: pd.Series, span: int = 20) -> pd.Series:
    """
    EMA Ratio: close[t] / EMA(span)[t].

    Values > 1 indicate price is above its trend (overbought signal).
    Values < 1 indicate price is below its trend (oversold signal).
    EWM is backward-looking by design.
    """
    ema = close.ewm(span=span, adjust=False).mean()
    return close / ema


# ──────────────────────────────────────────────────────────────────────────────
# Feature Set Builders
# ──────────────────────────────────────────────────────────────────────────────

def build_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    RAW feature set: only the five OHLCV columns.

    No derived indicators — purely observable market data.
    StandardScaler is applied downstream to normalise the different scales.

    Note: price levels (Open, High, Low, Close) are non-stationary.
    This is a known limitation of the raw feature set and is discussed
    in the README honest-interpretation section.
    """
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ENGINEERED feature set: hand-computed technical indicators + lag returns.

    Every feature at row t uses ONLY data available at or before time t.
    See module-level docstring for the full leakage-prevention argument.

    Features
    --------
    rsi_14          : RSI with 14-period Wilder smoothing
    macd_line       : MACD line (EMA12 − EMA26)
    macd_signal     : Signal line (EMA9 of MACD)
    macd_hist       : MACD histogram (line − signal)
    bb_width        : Bollinger Band relative width (20-period)
    bb_pct          : %B position within Bollinger Bands
    rolling_vol_10  : 10-day rolling std of log returns
    ema_ratio_20    : Close / EMA(20)
    return_lag_1    : log return 1 day ago  [LEAKAGE NOTE: shift(1)]
    return_lag_2    : log return 2 days ago [shift(2)]
    return_lag_5    : log return 5 days ago [shift(5)]
    return_lag_10   : log return 10 days ago [shift(10)]
    """
    feats   = pd.DataFrame(index=df.index)
    close   = df["Close"]
    log_ret = np.log(close / close.shift(1))   # daily log return (backward)

    # ── Indicator 1: RSI ───────────────────────────────────────────────────
    feats["rsi_14"] = _rsi(close, period=14)

    # ── Indicator 2: MACD ─────────────────────────────────────────────────
    feats = pd.concat([feats, _macd(close, fast=12, slow=26, signal=9)], axis=1)

    # ── Indicator 3: Bollinger Bands ──────────────────────────────────────
    feats = pd.concat([feats, _bollinger_bands(close, period=20)], axis=1)

    # ── Indicator 4: Rolling Volatility ───────────────────────────────────
    feats["rolling_vol_10"] = _rolling_volatility(close, period=10)

    # ── Indicator 5: EMA Ratio ────────────────────────────────────────────
    feats["ema_ratio_20"] = _ema_ratio(close, span=20)

    # ── Lag Returns (Rule 3 — explicit shift prevents leakage) ────────────
    # shift(k) at row t gives log_ret[t-k], i.e., the return k days ago.
    # shift(1) → yesterday's return; shift(2) → two days ago; etc.
    for lag in [1, 2, 5, 10]:
        feats[f"return_lag_{lag}"] = log_ret.shift(lag)

    return feats


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def build_all_features(df: pd.DataFrame):
    """
    Build labels, raw features, and engineered features from raw OHLCV data.

    Steps
    -----
    1. Attach next-day direction label (drops last row).
    2. Build raw (OHLCV) and engineered (indicator + lag) feature matrices.
    3. Drop indicator warm-up rows (NaNs at the START of the series).
       This is leakage-safe — they are the oldest rows, always in train territory.
    4. Return aligned matrices ready for splitting and scaling.

    Returns
    -------
    df_labeled : pd.DataFrame  — OHLCV + target column (for downstream use)
    X_raw      : pd.DataFrame  — raw OHLCV feature matrix
    X_eng      : pd.DataFrame  — engineered feature matrix
    y          : pd.Series     — binary target (0/1)
    """
    # Step 1: Labels
    df_labeled = build_labels(df)

    # Step 2: Feature matrices (both aligned to df_labeled's index)
    X_raw = build_raw_features(df_labeled)
    X_eng = build_engineered_features(df_labeled)
    y     = df_labeled["target"]

    # Step 3: Drop warm-up NaN rows (from start of series — leakage safe)
    #   The longest warm-up comes from MACD slow EMA (26 bars) + signal (9 bars).
    #   All NaN rows are at the very beginning — they fall in the training period.
    valid_idx = X_eng.dropna().index
    n_dropped = len(X_eng) - len(valid_idx)
    if n_dropped > 0:
        print(f"[features] Dropped {n_dropped} warm-up rows (NaN indicators, "
              f"start of series — leakage safe).")

    X_raw      = X_raw.loc[valid_idx]
    X_eng      = X_eng.loc[valid_idx]
    y          = y.loc[valid_idx]
    df_labeled = df_labeled.loc[valid_idx]

    # Summary
    n = len(y)
    up_pct = y.mean() * 100
    print(
        f"[features] Final dataset : {n} rows\n"
        f"[features] Class balance : {up_pct:.1f}% UP | {100-up_pct:.1f}% DOWN\n"
        f"[features] Date range    : {y.index[0].date()} → {y.index[-1].date()}"
    )
    return df_labeled, X_raw, X_eng, y
