"""Pair formation module using Sum of Squared Differences (SSD)."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize prices by dividing each series by its first value.

    This makes all series start at 1.0 for fair comparison.
    Handles NaN values by dropping columns where first price is NaN.

    Args:
        prices: DataFrame with symbols as columns and prices as values

    Returns:
        DataFrame with normalized prices (all starting at 1.0)
    """
    # Check for NaN in first row - can't normalize without starting price
    first_row = prices.iloc[0]
    if first_row.isna().any():
        # Drop columns where first price is NaN
        valid_cols = first_row.dropna().index
        prices = prices[valid_cols]
    return prices / prices.iloc[0]


def calculate_ssd(series_a: pd.Series, series_b: pd.Series) -> float:
    """
    Calculate Sum of Squared Differences between two series.

    Handles NaN values by only using dates where both series have valid data.
    Returns infinity if less than 50% overlap to exclude pair from selection.

    Args:
        series_a: First price series (should be normalized)
        series_b: Second price series (should be normalized)

    Returns:
        Sum of squared differences (lower = more similar), or inf if insufficient overlap
    """
    # Only use dates where both have valid data
    valid_mask = series_a.notna() & series_b.notna()
    if valid_mask.sum() < len(series_a) * 0.5:  # Require 50% overlap
        return float('inf')  # Exclude pair from selection
    diff = series_a[valid_mask] - series_b[valid_mask]
    return float((diff ** 2).sum())


def _calculate_ssd_matrix_loop(normalized_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate SSD matrix using Python loop (original implementation).

    Kept for testing/comparison purposes. Use calculate_ssd_matrix() instead.

    Args:
        normalized_prices: DataFrame with normalized prices

    Returns:
        Symmetric DataFrame with SSD values for each pair
    """
    symbols = normalized_prices.columns.tolist()
    n = len(symbols)

    # Initialize matrix with zeros
    ssd_matrix = pd.DataFrame(
        np.zeros((n, n)),
        index=symbols,
        columns=symbols,
    )

    # Calculate SSD for each pair
    for sym_a, sym_b in combinations(symbols, 2):
        ssd = calculate_ssd(
            normalized_prices[sym_a],
            normalized_prices[sym_b],
        )
        ssd_matrix.loc[sym_a, sym_b] = ssd
        ssd_matrix.loc[sym_b, sym_a] = ssd  # Symmetric

    return ssd_matrix


def _calculate_ssd_matrix_vectorized(normalized_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate SSD matrix using NumPy broadcasting (vectorized).

    This is ~10-100x faster than the loop-based implementation for large universes.
    Uses broadcasting to compute all pair differences simultaneously.

    Args:
        normalized_prices: DataFrame with normalized prices

    Returns:
        Symmetric DataFrame with SSD values for each pair
    """
    symbols = normalized_prices.columns.tolist()
    n = len(symbols)
    n_days = len(normalized_prices)

    if n == 0:
        return pd.DataFrame()

    # Convert to NumPy array: shape (n_days, n_symbols)
    data = normalized_prices.values

    # Handle NaN values: create validity mask
    valid_mask = ~np.isnan(data)  # shape: (n_days, n_symbols)

    # Replace NaN with 0 for computation (will mask later)
    data_filled = np.nan_to_num(data, nan=0.0)

    # Compute pairwise overlap counts: (n_symbols, n_symbols)
    # overlap[i,j] = number of days where both i and j have valid data
    # Convert to int to ensure proper counting (bool @ bool can produce bool)
    overlap_counts = valid_mask.astype(np.int32).T @ valid_mask.astype(np.int32)  # (n, n)

    # Minimum overlap threshold (50% of data)
    min_overlap = n_days * 0.5

    # Compute SSD using broadcasting
    # diff[t,i,j] = data[t,i] - data[t,j]
    # Expand dimensions: (n_days, n_symbols, 1) - (n_days, 1, n_symbols)
    diff = data_filled[:, :, np.newaxis] - data_filled[:, np.newaxis, :]  # (n_days, n, n)

    # Create joint validity mask
    joint_valid = valid_mask[:, :, np.newaxis] & valid_mask[:, np.newaxis, :]  # (n_days, n, n)

    # Compute squared differences, masked (0 where not jointly valid)
    diff_squared = np.where(joint_valid, diff ** 2, 0.0)

    # Sum over time dimension
    ssd_values = diff_squared.sum(axis=0)  # (n, n)

    # Set pairs with insufficient overlap to infinity
    ssd_values = np.where(overlap_counts >= min_overlap, ssd_values, np.inf)

    # Set diagonal to 0 (same symbol pairs)
    np.fill_diagonal(ssd_values, 0.0)

    return pd.DataFrame(ssd_values, index=symbols, columns=symbols)


def _calculate_ssd_matrix_chunked(
    normalized_prices: pd.DataFrame,
    chunk_size: int = 100,
) -> pd.DataFrame:
    """
    Memory-efficient SSD calculation using chunks.

    For very large universes (1000+ stocks), processes in chunks to avoid
    creating massive intermediate arrays. The full vectorized version uses
    O(n² × days) memory which can be ~2GB+ for 1000 stocks.

    Args:
        normalized_prices: DataFrame with normalized prices
        chunk_size: Number of symbols to process per chunk

    Returns:
        Symmetric DataFrame with SSD values for each pair
    """
    symbols = normalized_prices.columns.tolist()
    n_symbols = len(symbols)
    n_days = len(normalized_prices)

    if n_symbols == 0:
        return pd.DataFrame()

    data = normalized_prices.values
    valid_mask = ~np.isnan(data)
    data_filled = np.nan_to_num(data, nan=0.0)

    min_overlap = n_days * 0.5
    # Convert to int to ensure proper counting (bool @ bool can produce bool)
    overlap_counts = valid_mask.astype(np.int32).T @ valid_mask.astype(np.int32)
    ssd_values = np.zeros((n_symbols, n_symbols))

    # Process in chunks to limit memory usage
    for i_start in range(0, n_symbols, chunk_size):
        i_end = min(i_start + chunk_size, n_symbols)

        for j_start in range(i_start, n_symbols, chunk_size):
            j_end = min(j_start + chunk_size, n_symbols)

            # Compute SSD for this chunk
            chunk_i = data_filled[:, i_start:i_end]  # (n_days, chunk_size)
            chunk_j = data_filled[:, j_start:j_end]  # (n_days, chunk_size)

            valid_i = valid_mask[:, i_start:i_end]
            valid_j = valid_mask[:, j_start:j_end]

            # Broadcasting within chunk
            diff = chunk_i[:, :, np.newaxis] - chunk_j[:, np.newaxis, :]
            joint_valid = valid_i[:, :, np.newaxis] & valid_j[:, np.newaxis, :]

            diff_squared = np.where(joint_valid, diff ** 2, 0.0)
            chunk_ssd = diff_squared.sum(axis=0)

            ssd_values[i_start:i_end, j_start:j_end] = chunk_ssd
            if i_start != j_start:
                ssd_values[j_start:j_end, i_start:i_end] = chunk_ssd.T

    # Apply overlap threshold
    ssd_values = np.where(overlap_counts >= min_overlap, ssd_values, np.inf)
    np.fill_diagonal(ssd_values, 0.0)

    return pd.DataFrame(ssd_values, index=symbols, columns=symbols)


def calculate_ssd_matrix(
    normalized_prices: pd.DataFrame,
    use_chunked: bool = False,
    chunk_size: int = 100,
) -> pd.DataFrame:
    """
    Calculate SSD matrix for all pairs of symbols.

    Uses vectorized NumPy broadcasting for ~10-100x speedup over loop-based
    implementation. For very large universes (1000+ stocks), use use_chunked=True
    to limit memory usage.

    Args:
        normalized_prices: DataFrame with normalized prices
        use_chunked: If True, use memory-efficient chunked computation
        chunk_size: Chunk size for chunked computation (default 100)

    Returns:
        Symmetric DataFrame with SSD values for each pair
    """
    if use_chunked:
        return _calculate_ssd_matrix_chunked(normalized_prices, chunk_size)
    return _calculate_ssd_matrix_vectorized(normalized_prices)


def select_top_pairs(
    ssd_matrix: pd.DataFrame,
    n: int = 5,
) -> list[tuple[str, str]]:
    """
    Select top N pairs with lowest SSD (most similar behavior).

    Args:
        ssd_matrix: SSD matrix from calculate_ssd_matrix
        n: Number of pairs to select

    Returns:
        List of (symbol_a, symbol_b) tuples, sorted by SSD ascending
    """
    symbols = ssd_matrix.columns.tolist()

    # Get all unique pairs with their SSD values
    pairs = []
    for sym_a, sym_b in combinations(symbols, 2):
        ssd = ssd_matrix.loc[sym_a, sym_b]
        pairs.append((sym_a, sym_b, ssd))

    # Sort by SSD (ascending) and take top N
    pairs.sort(key=lambda x: x[2])
    top_pairs = [(p[0], p[1]) for p in pairs[:n]]

    return top_pairs


def get_pair_stats(
    normalized_prices: pd.DataFrame,
    pair: tuple[str, str],
) -> dict:
    """
    Get statistics for a pair of symbols.

    Args:
        normalized_prices: DataFrame with normalized prices
        pair: Tuple of (symbol_a, symbol_b)

    Returns:
        Dictionary with pair statistics
    """
    sym_a, sym_b = pair
    series_a = normalized_prices[sym_a]
    series_b = normalized_prices[sym_b]

    ssd = calculate_ssd(series_a, series_b)
    correlation = series_a.corr(series_b)
    spread = series_a - series_b

    return {
        "pair": pair,
        "ssd": ssd,
        "correlation": correlation,
        "spread_mean": spread.mean(),
        "spread_std": spread.std(),
    }


