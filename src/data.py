"""Data fetching and caching module for Polygon.io API."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import requests
from dotenv import load_dotenv


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

    for symbol in symbols:
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

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("resultsCount", 0) == 0:
            print(f"Warning: No data returned for {symbol}")
            continue

        results = data.get("results", [])
        if not results:
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

    if not all_data:
        raise ValueError("No data fetched for any symbol")

    # Combine all symbols into a single DataFrame with MultiIndex columns
    combined = pd.concat(all_data, axis=1)
    combined.columns = pd.MultiIndex.from_tuples(
        [(sym, col) for sym, col in combined.columns]
    )

    # Check for symbols with significantly less data before alignment
    symbol_counts = {}
    for symbol in combined.columns.get_level_values(0).unique():
        symbol_counts[symbol] = combined[symbol].dropna().shape[0]

    max_count = max(symbol_counts.values())
    problem_symbols = {s: c for s, c in symbol_counts.items() if c < max_count * 0.9}

    if problem_symbols:
        print("\n" + "=" * 60)
        print("WARNING: Some symbols have significantly less data!")
        print("=" * 60)
        for sym, count in problem_symbols.items():
            pct = count / max_count * 100
            print(f"  {sym}: {count} rows ({pct:.1f}% of max)")
        print(f"\nMax rows: {max_count}")
        print("These symbols may cause data gaps after alignment.")
        print("Consider removing them from your universe.")
        print("=" * 60 + "\n")

    # Align all symbols to the same dates (intersection)
    rows_before = len(combined)
    combined = combined.dropna(how="any")
    rows_after = len(combined)

    if rows_after < rows_before * 0.8:
        print(f"WARNING: Alignment dropped {rows_before - rows_after} rows ({(1 - rows_after/rows_before)*100:.1f}%)")

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
    close_prices = pd.DataFrame(index=prices_df.index)
    for symbol in symbols:
        close_prices[symbol] = prices_df[(symbol, "close")]
    return close_prices


def get_open_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract open prices from the multi-index DataFrame.

    Args:
        prices_df: DataFrame from fetch_prices with MultiIndex columns

    Returns:
        DataFrame with symbols as columns and open prices
    """
    symbols = prices_df.columns.get_level_values(0).unique()
    open_prices = pd.DataFrame(index=prices_df.index)
    for symbol in symbols:
        open_prices[symbol] = prices_df[(symbol, "open")]
    return open_prices


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


def print_data_quality_report(
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> None:
    """
    Print a comprehensive data quality report.

    Args:
        prices: DataFrame with price data
        start_date: Expected start date
        end_date: Expected end date
    """
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    # Coverage
    coverage = get_trading_days_coverage(prices, start_date, end_date)
    print(f"\nDate Range: {start_date} to {end_date}")
    print(f"Expected trading days: {coverage['total_trading_days']}")
    print(f"Actual days in data: {coverage['actual_days']}")
    print(f"Coverage: {coverage['coverage_pct']:.1%}")

    # Gaps
    gaps = find_data_gaps(prices, max_gap_days=5)
    if gaps:
        print(f"\nData Gaps Found ({len(gaps)}):")
        for gap in gaps[:10]:  # Show first 10
            print(f"  {gap['start_date'].date()} to {gap['end_date'].date()} ({gap['gap_days']} days)")
        if len(gaps) > 10:
            print(f"  ... and {len(gaps) - 10} more")
    else:
        print("\nNo significant data gaps found.")

    # Validation issues
    issues = validate_price_data(prices)
    if issues:
        print(f"\nData Issues Found ({len(issues)}):")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("\nNo data quality issues found.")

    print("=" * 60)
