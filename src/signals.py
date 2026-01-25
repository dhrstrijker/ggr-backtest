"""Signal generation module for GGR Distance Method.

Implements the original Gatev, Goetzmann, and Rouwenhorst (2006) methodology:
- Static σ calculated from formation period (not rolling)
- Entry when |spread| > 2σ (distance from parity)
- Exit when spread crosses 0 (prices converge)
"""

from __future__ import annotations

import pandas as pd


def _crosses_zero(prev_spread: float, current_spread: float) -> bool:
    """Check if spread crossed zero between two values.

    NaN-safe: returns False if either value is NaN.

    Args:
        prev_spread: Previous spread value
        current_spread: Current spread value

    Returns:
        True if spread crossed zero (sign change or touched zero), False otherwise
    """
    if pd.isna(prev_spread) or pd.isna(current_spread):
        return False
    # Cross from positive to zero/negative, or negative to zero/positive
    # Also handle prev_spread exactly at zero moving away
    return (
        (prev_spread > 0 and current_spread <= 0) or
        (prev_spread < 0 and current_spread >= 0) or
        (prev_spread == 0 and current_spread != 0)  # Started at zero, moved away
    )


# =============================================================================
# Core Functions
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
        - 0: No action (not an explicit exit signal)

    Note:
        This function returns entry signals only. Exit conditions are tracked
        internally via position state. A value of 0 means "no new action",
        not an explicit exit signal. Exits are implicit when the spread
        crosses zero while in a position.
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

                # Check for sign change (crossing zero) - NaN-safe
                if _crosses_zero(prev_spread, current_spread):
                    # Signal exit (0 after being in position)
                    position = 0

    return signals
