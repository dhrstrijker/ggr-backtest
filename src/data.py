"""Data fetching and caching module for Polygon.io API."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# Rate limiting configuration
REQUESTS_PER_SECOND = 10  # Polygon paid tier recommends staying under 100/sec


def get_api_key() -> str:
    """Load Polygon API key from environment."""
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError(
            "POLYGON_API_KEY not found. Please set it in your .env file."
        )
    return api_key


def fetch_prices(
    symbols: list[str],
    start_date: str,
    end_date: str,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch daily OHLC data from Polygon.io for multiple symbols.

    Handles delistings and new stocks by keeping ALL trading dates (union)
    and allowing NaN values for symbols that don't have data on certain dates.
    The per-cycle filtering in staggered.py will handle symbol selection.

    Args:
        symbols: List of stock tickers (e.g., ['AAPL', 'MSFT'])
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        api_key: Polygon API key (optional, will load from env if not provided)

    Returns:
        DataFrame with MultiIndex columns (symbol, ohlcv) and DatetimeIndex
    """
    if api_key is None:
        api_key = get_api_key()

    all_data = {}
    delay = 1.0 / REQUESTS_PER_SECOND  # Delay between requests for rate limiting
    failed_symbols = []

    for i, symbol in enumerate(symbols):
        # Progress indicator for large fetches
        if len(symbols) > 50 and (i + 1) % 50 == 0:
            print(f"  Fetched {i + 1}/{len(symbols)} symbols...")

        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
            f"{start_date}/{end_date}"
        )
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": api_key,
        }

        try:
            response = requests.get(url, params=params)

            # Handle rate limiting (429 Too Many Requests)
            if response.status_code == 429:
                print(f"  Rate limited, waiting 60 seconds...")
                time.sleep(60)
                response = requests.get(url, params=params)

            response.raise_for_status()
            data = response.json()

            if data.get("resultsCount", 0) == 0:
                print(f"  Warning: No data returned for {symbol}")
                failed_symbols.append(symbol)
                continue

            results = data.get("results", [])
            if not results:
                failed_symbols.append(symbol)
                continue

            df = pd.DataFrame(results)
            df["date"] = pd.to_datetime(df["t"], unit="ms")
            df = df.set_index("date")
            df = df.rename(columns={
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            })
            df = df[["open", "high", "low", "close", "volume"]]

            all_data[symbol] = df

        except requests.exceptions.RequestException as e:
            print(f"  Error fetching {symbol}: {e}")
            failed_symbols.append(symbol)

        # Rate limiting delay
        time.sleep(delay)

    if not all_data:
        raise ValueError("No data fetched for any symbol")

    if failed_symbols:
        print(f"\n  Failed to fetch {len(failed_symbols)} symbols: {failed_symbols[:10]}{'...' if len(failed_symbols) > 10 else ''}")

    # Combine all symbols into a single DataFrame with MultiIndex columns
    # Using join='outer' keeps ALL dates (union) - allows NaN for missing data
    combined = pd.concat(all_data, axis=1, join="outer")
    combined.columns = pd.MultiIndex.from_tuples(
        [(sym, col) for sym, col in combined.columns]
    )

    # Sort by date
    combined = combined.sort_index()

    # Report data coverage statistics
    symbol_counts = {}
    for symbol in combined.columns.get_level_values(0).unique():
        symbol_counts[symbol] = combined[symbol]["close"].notna().sum()

    max_count = max(symbol_counts.values())
    min_count = min(symbol_counts.values())

    print(f"\n  Data coverage: {min_count}-{max_count} trading days per symbol")
    print(f"  Total trading days in dataset: {len(combined)}")

    # Warn about symbols with significantly less data
    problem_symbols = {s: c for s, c in symbol_counts.items() if c < max_count * 0.5}

    if problem_symbols:
        print(f"\n  WARNING: {len(problem_symbols)} symbols have <50% data coverage")
        if len(problem_symbols) <= 10:
            for sym, count in sorted(problem_symbols.items(), key=lambda x: x[1]):
                pct = count / max_count * 100
                print(f"    {sym}: {count} days ({pct:.1f}%)")

    return combined


def get_close_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract close prices from the multi-index DataFrame.

    Args:
        prices_df: DataFrame from fetch_prices with MultiIndex columns

    Returns:
        DataFrame with symbols as columns and close prices
    """
    symbols = prices_df.columns.get_level_values(0).unique()
    # Use dict comprehension + pd.concat to avoid DataFrame fragmentation
    close_dict = {symbol: prices_df[(symbol, "close")] for symbol in symbols}
    return pd.DataFrame(close_dict, index=prices_df.index)


def get_open_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract open prices from the multi-index DataFrame.

    Args:
        prices_df: DataFrame from fetch_prices with MultiIndex columns

    Returns:
        DataFrame with symbols as columns and open prices
    """
    symbols = prices_df.columns.get_level_values(0).unique()
    # Use dict comprehension + pd.DataFrame to avoid fragmentation
    open_dict = {symbol: prices_df[(symbol, "open")] for symbol in symbols}
    return pd.DataFrame(open_dict, index=prices_df.index)


def cache_prices(df: pd.DataFrame, path: str | Path) -> None:
    """
    Cache price data to CSV file.

    Args:
        df: DataFrame to cache
        path: Path to save the CSV file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def load_cached(path: str | Path) -> pd.DataFrame | None:
    """
    Load cached price data from CSV file.

    Args:
        path: Path to the CSV file

    Returns:
        DataFrame if file exists, None otherwise
    """
    path = Path(path)
    if not path.exists():
        return None

    df = pd.read_csv(path, index_col=0, parse_dates=True, header=[0, 1])
    return df


def fetch_or_load(
    symbols: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str | Path = "data",
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch price data from Polygon.io or load from cache.

    Args:
        symbols: List of stock tickers
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        cache_dir: Directory for cached data
        api_key: Polygon API key (optional)

    Returns:
        DataFrame with price data
    """
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"prices_{start_date}_{end_date}.csv"

    # Try loading from cache
    cached = load_cached(cache_file)
    if cached is not None:
        # Check if all symbols are in cache
        cached_symbols = set(cached.columns.get_level_values(0).unique())
        if set(symbols).issubset(cached_symbols):
            print(f"Loaded {len(symbols)} symbols from cache")
            return cached

    # Fetch fresh data
    print(f"Fetching data for {len(symbols)} symbols from Polygon.io...")
    df = fetch_prices(symbols, start_date, end_date, api_key)
    cache_prices(df, cache_file)
    print(f"Cached data to {cache_file}")

    return df


def fetch_benchmark(
    start_date: str,
    end_date: str,
    symbol: str = "SPY",
    cache_dir: str | Path = "data",
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch benchmark (e.g., SPY) data for comparison.

    Uses the same caching infrastructure as fetch_or_load but with
    a separate cache file for benchmark data.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        symbol: Benchmark symbol (default SPY)
        cache_dir: Directory for cached data
        api_key: Polygon API key (optional)

    Returns:
        DataFrame with benchmark close prices indexed by date
    """
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"benchmark_{symbol}_{start_date}_{end_date}.csv"

    # Try loading from cache
    if cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"Loaded {symbol} benchmark from cache")
        return df

    # Fetch fresh data
    print(f"Fetching {symbol} benchmark from Polygon.io...")
    prices_df = fetch_prices([symbol], start_date, end_date, api_key)
    close_prices = get_close_prices(prices_df)

    # Cache the data
    close_prices.to_csv(cache_file)
    print(f"Cached benchmark to {cache_file}")

    return close_prices


# =============================================================================
# Data Validation Functions
# =============================================================================


def find_data_gaps(
    prices: pd.DataFrame,
    max_gap_days: int = 3,
) -> list[dict]:
    """
    Find gaps in trading data that exceed the maximum allowed.

    Normal gaps (weekends, single holidays) are 1-3 days.
    Anything longer indicates missing data.

    Args:
        prices: DataFrame with DatetimeIndex
        max_gap_days: Maximum allowed gap in calendar days (default 3 for weekends)

    Returns:
        List of dictionaries describing each gap found
    """
    if len(prices) < 2:
        return []

    gaps = []
    dates = prices.index.sort_values()

    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i - 1]).days
        if gap > max_gap_days:
            gaps.append({
                'start_date': dates[i - 1],
                'end_date': dates[i],
                'gap_days': gap,
            })

    return gaps


def validate_price_data(
    prices: pd.DataFrame,
    max_daily_change: float = 0.5,
) -> list[str]:
    """
    Validate price data for common issues.

    Checks for:
    - NaN values
    - Zero or negative prices
    - Extreme price jumps (potential data errors)

    Args:
        prices: DataFrame with price data
        max_daily_change: Maximum allowed daily price change as fraction (default 50%)

    Returns:
        List of issue descriptions (empty if data is valid)
    """
    issues = []

    # Check for NaN values
    nan_counts = prices.isna().sum()
    for col in nan_counts[nan_counts > 0].index:
        issues.append(f"NaN values found in {col}: {nan_counts[col]} rows")

    # Check for zero or negative prices
    for col in prices.columns:
        zeros = (prices[col] == 0).sum()
        negatives = (prices[col] < 0).sum()
        if zeros > 0:
            issues.append(f"Zero prices found in {col}: {zeros} rows")
        if negatives > 0:
            issues.append(f"Negative prices found in {col}: {negatives} rows")

    # Check for extreme price jumps
    for col in prices.columns:
        pct_change = prices[col].pct_change(fill_method=None).abs()
        extreme_jumps = pct_change[pct_change > max_daily_change]
        if len(extreme_jumps) > 0:
            for date, change in extreme_jumps.items():
                issues.append(
                    f"Extreme price jump in {col} on {date.date()}: {change:.1%}"
                )

    return issues


def get_trading_days_coverage(
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> dict:
    """
    Calculate what percentage of expected trading days are covered.

    Args:
        prices: DataFrame with price data
        start_date: Expected start date (YYYY-MM-DD)
        end_date: Expected end date (YYYY-MM-DD)

    Returns:
        Dictionary with coverage statistics
    """
    # Generate expected trading days (business days, excluding weekends)
    expected_dates = pd.bdate_range(start=start_date, end=end_date)
    actual_dates = prices.index

    # Find missing dates
    missing_dates = expected_dates.difference(actual_dates)

    return {
        'start_date': start_date,
        'end_date': end_date,
        'total_trading_days': len(expected_dates),
        'actual_days': len(actual_dates),
        'missing_days': len(missing_dates),
        'coverage_pct': len(actual_dates) / len(expected_dates) if len(expected_dates) > 0 else 0,
        'missing_dates': missing_dates.tolist(),
    }


