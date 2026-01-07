"""Signal generation module for GGR Distance Method.

Implements the original Gatev, Goetzmann, and Rouwenhorst (2006) methodology:
- Static σ calculated from formation period (not rolling)
- Entry when |spread| > 2σ (distance from parity)
- Exit when spread crosses 0 (prices converge)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# =============================================================================
# Core Functions (used by both methods)
# =============================================================================


def calculate_spread(
    prices_a: pd.Series,
    prices_b: pd.Series,
    normalize: bool = True,
) -> pd.Series:
    """
    Calculate the spread between two price series.

    For pair trading, we go LONG the spread when it's low (buy A, sell B)
    and SHORT the spread when it's high (sell A, buy B).

    Args:
        prices_a: Price series for stock A (long leg)
        prices_b: Price series for stock B (short leg)
        normalize: If True, normalize prices first (divide by first value)

    Returns:
        Series representing the spread
    """
    if normalize:
        # Normalize to make series comparable
        norm_a = prices_a / prices_a.iloc[0]
        norm_b = prices_b / prices_b.iloc[0]
    else:
        norm_a = prices_a
        norm_b = prices_b

    spread = norm_a - norm_b
    return spread


# =============================================================================
# Rolling Z-Score Method (Bollinger-style, NOT original GGR)
# =============================================================================


def calculate_zscore(
    spread: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """
    Calculate rolling Z-score of the spread.

    NOTE: This is a Bollinger-style approach, NOT the original GGR method.
    See calculate_formation_stats() and calculate_distance() for GGR.

    IMPORTANT: Uses only past data (no lookahead bias).
    Z-score at time t uses only data from t-lookback to t.

    Args:
        spread: Spread series from calculate_spread
        lookback: Number of periods for rolling mean/std calculation

    Returns:
        Series of Z-scores (NaN for warmup period)
    """
    # Rolling mean and std using ONLY past data (no lookahead)
    rolling_mean = spread.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = spread.rolling(window=lookback, min_periods=lookback).std()

    # Z-score: (current - mean) / std
    zscore = (spread - rolling_mean) / rolling_std

    return zscore


# =============================================================================
# GGR Distance Method (Original Paper Implementation)
# =============================================================================


def calculate_formation_stats(spread: pd.Series) -> dict:
    """
    Calculate static statistics from formation period.

    Per GGR paper: Mean and std are calculated ONCE over the entire
    formation period and remain fixed during trading.

    Args:
        spread: Spread series from formation period

    Returns:
        dict with 'mean' and 'std' keys
    """
    return {
        'mean': spread.mean(),  # Should be ~0 for normalized prices
        'std': spread.std(),
    }


def calculate_distance(
    spread: pd.Series,
    formation_std: float,
) -> pd.Series:
    """
    Calculate distance in terms of formation-period standard deviations.

    Per GGR paper: Distance = spread / σ_formation
    The spread reverts to 0 (parity), not to a rolling mean.

    Unlike rolling Z-score, this uses a FIXED σ from the formation period.

    Args:
        spread: Current spread series (normalized prices)
        formation_std: Standard deviation from formation period

    Returns:
        Series of distances (in σ units, can be positive or negative)
    """
    # Spread is already relative to 0 (parity) since normalized
    # No mean subtraction - we're measuring distance from parity
    distance = spread / formation_std
    return distance


def generate_signals_ggr(
    spread: pd.Series,
    formation_std: float,
    entry_threshold: float = 2.0,
) -> pd.Series:
    """
    Generate trading signals per GGR paper methodology.

    Entry: When |spread| > entry_threshold * formation_std
    Exit: When spread crosses 0 (prices cross/converge)

    This differs from Bollinger-style in two key ways:
    1. Uses static σ from formation (not rolling)
    2. Exits on spread crossing zero (not at a threshold)

    Args:
        spread: Spread series (P_A_norm - P_B_norm)
        formation_std: Static std from formation period
        entry_threshold: Number of sigmas for entry (default 2.0)

    Returns:
        Series with signal values:
        - 1: Entry long spread (buy A, sell B)
        - -1: Entry short spread (sell A, buy B)
        - 0: Exit or no action
    """
    signals = pd.Series(index=spread.index, data=0, dtype=float)
    position = 0  # Track current position: 0 = flat, 1 = long, -1 = short

    entry_level = entry_threshold * formation_std

    for i in range(len(spread)):
        current_spread = spread.iloc[i]

        if pd.isna(current_spread):
            continue

        if position == 0:
            # Not in a position - look for entry
            if current_spread > entry_level:
                # Spread too high - short the spread (sell A, buy B)
                signals.iloc[i] = -1
                position = -1
            elif current_spread < -entry_level:
                # Spread too low - long the spread (buy A, sell B)
                signals.iloc[i] = 1
                position = 1
        else:
            # In a position - look for exit (spread crossing zero)
            if i > 0:
                prev_spread = spread.iloc[i - 1]

                # Check for sign change (crossing zero)
                crossed_zero = (
                    (prev_spread > 0 and current_spread <= 0) or
                    (prev_spread < 0 and current_spread >= 0)
                )

                if crossed_zero:
                    # Signal exit (0 after being in position)
                    position = 0

    return signals


def generate_signals(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
) -> pd.Series:
    """
    Generate trading signals based on Z-score thresholds.

    Strategy:
    - ENTRY (Long spread): when zscore < -entry_threshold (spread too low)
    - ENTRY (Short spread): when zscore > entry_threshold (spread too high)
    - EXIT: when |zscore| < exit_threshold (spread normalized)

    Signal values:
    - 1: Entry long spread (buy A, sell B)
    - -1: Entry short spread (sell A, buy B)
    - 0: Exit or no action

    Args:
        zscore: Z-score series from calculate_zscore
        entry_threshold: Threshold for entry (default 2.0 = 2 sigma)
        exit_threshold: Threshold for exit (default 0.5 = 0.5 sigma)

    Returns:
        Series with signal values (1, -1, 0)
    """
    signals = pd.Series(index=zscore.index, data=np.nan, dtype=float)
    position = 0  # Track current position: 0 = flat, 1 = long, -1 = short

    for i, (date, z) in enumerate(zscore.items()):
        if pd.isna(z):
            signals.iloc[i] = 0
            continue

        if position == 0:
            # Not in a position - look for entry
            if z > entry_threshold:
                # Spread too high - short the spread (sell A, buy B)
                signals.iloc[i] = -1
                position = -1
            elif z < -entry_threshold:
                # Spread too low - long the spread (buy A, sell B)
                signals.iloc[i] = 1
                position = 1
            else:
                signals.iloc[i] = 0
        else:
            # In a position - look for exit
            if abs(z) < exit_threshold:
                # Spread normalized - exit
                signals.iloc[i] = 0
                position = 0
            else:
                # Stay in position (no new signal)
                signals.iloc[i] = 0

    return signals


def get_positions(signals: pd.Series) -> pd.Series:
    """
    Convert signals to position series.

    Position shows the current holding at each point in time.

    Args:
        signals: Signal series from generate_signals

    Returns:
        Series with position values (1, -1, 0)
    """
    positions = pd.Series(index=signals.index, data=0.0)
    position = 0

    for i, (date, signal) in enumerate(signals.items()):
        if signal == 1:
            position = 1
        elif signal == -1:
            position = -1
        elif signal == 0 and position != 0:
            # Check if we should exit based on previous position
            # Signal of 0 after being in position means exit
            if i > 0 and signals.iloc[i-1] != signal:
                position = 0

        positions.iloc[i] = position

    return positions


def get_signal_dates(signals: pd.Series) -> dict:
    """
    Extract entry and exit dates from signal series.

    Args:
        signals: Signal series from generate_signals

    Returns:
        Dictionary with 'long_entries', 'short_entries', 'exits' lists
    """
    long_entries = signals[signals == 1].index.tolist()
    short_entries = signals[signals == -1].index.tolist()

    # Find exits - need to track position changes
    exits = []
    position = 0
    for date, signal in signals.items():
        if signal == 1:
            position = 1
        elif signal == -1:
            position = -1
        elif signal == 0 and position != 0:
            exits.append(date)
            position = 0

    return {
        "long_entries": long_entries,
        "short_entries": short_entries,
        "exits": exits,
    }
