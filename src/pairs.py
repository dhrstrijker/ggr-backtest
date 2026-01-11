"""Pair formation module using Sum of Squared Differences (SSD)."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize prices by dividing each series by its first value.

    This makes all series start at 1.0 for fair comparison.

    Args:
        prices: DataFrame with symbols as columns and prices as values

    Returns:
        DataFrame with normalized prices (all starting at 1.0)
    """
    return prices / prices.iloc[0]


def calculate_ssd(series_a: pd.Series, series_b: pd.Series) -> float:
    """
    Calculate Sum of Squared Differences between two series.

    Args:
        series_a: First price series (should be normalized)
        series_b: Second price series (should be normalized)

    Returns:
        Sum of squared differences (lower = more similar)
    """
    diff = series_a - series_b
    return float((diff ** 2).sum())


def calculate_ssd_matrix(normalized_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate SSD matrix for all pairs of symbols.

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


