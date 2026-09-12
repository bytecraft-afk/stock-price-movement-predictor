"""
data_loader.py
==============
Fetches and caches daily OHLCV data from yfinance.

No leakage risk here — this module retrieves raw market data only.
The cache ensures the pipeline is fully reproducible without hitting
the network on every run.
"""

import os
import pandas as pd
import yfinance as yf

# Resolve paths relative to the project root (one level up from src/)
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
DATA_DIR  = os.path.join(_ROOT_DIR, "data")


def load_data(
    ticker: str = "SPY",
    start: str = "2019-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Load OHLCV data for ``ticker`` between ``start`` and ``end``.

    Uses a local CSV cache at ``data/<TICKER>_raw.csv``.
    Downloads from yfinance if the cache file is missing.

    Parameters
    ----------
    ticker : str
        Ticker symbol (default "SPY" — S&P 500 ETF).
    start : str
        Start date in "YYYY-MM-DD" format.
    end : str
        End date in "YYYY-MM-DD" format (inclusive for yfinance).

    Returns
    -------
    pd.DataFrame
        DataFrame with a DatetimeIndex named "Date" and columns:
        Open, High, Low, Close, Volume.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, f"{ticker}_raw.csv")

    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    if os.path.exists(cache_path):
        print(f"[data_loader] Loading cached data from: {cache_path}")
        df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        if df.empty or not set(required_columns).issubset(df.columns):
            print(f"[data_loader] Ignoring invalid cache: {cache_path}")
            os.remove(cache_path)
            df = None
    else:
        df = None

    if df is None:
        print(f"[data_loader] Downloading {ticker} ({start} → {end}) via yfinance ...")
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,   # adjusts for splits/dividends automatically
            progress=False,
        )

        if raw.empty:
            raise RuntimeError(
                f"No data returned for {ticker}. Check the ticker, date range, "
                "network connection, or yfinance rate limit."
            )

        # yfinance ≥0.2 may return MultiIndex columns (ticker, field).
        # Flatten to single-level if needed.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Keep only the five canonical OHLCV columns
        missing_columns = set(required_columns) - set(raw.columns)
        if missing_columns:
            raise RuntimeError(
                f"Downloaded data is missing columns: {sorted(missing_columns)}."
            )
        df = raw[required_columns].copy()
        df.index.name = "Date"
        df.dropna(inplace=True)   # remove any rows with missing values
        if df.empty:
            raise RuntimeError(f"Downloaded data for {ticker} contains no valid rows.")
        df.to_csv(cache_path)
        print(f"[data_loader] Cached to: {cache_path}")

    # Ensure index is DatetimeIndex (in case of CSV re-load)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    print(
        f"[data_loader] Ticker  : {ticker}\n"
        f"[data_loader] Rows    : {len(df)}\n"
        f"[data_loader] Range   : {df.index.min().date()} → {df.index.max().date()}"
    )
    return df
