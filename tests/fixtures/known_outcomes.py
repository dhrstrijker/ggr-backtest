"""Pre-calculated test fixtures for integration testing.

These fixtures contain carefully constructed price data with manually
calculated expected outcomes to verify the backtest produces correct results.
"""

import numpy as np
import pandas as pd


def _create_formation_prices(n_days: int = 50, base_price: float = 100.0):
    """Create formation period prices with known statistics."""
    dates = pd.bdate_range("2024-01-01", periods=n_days)

    # Create prices that stay close together (low SSD)
    # Use sine wave for slight variation
    t = np.arange(n_days)
    variation = np.sin(t * 0.2) * 0.5  # Small amplitude

    prices_a = base_price + variation
    prices_b = base_price + variation * 0.9  # Slightly correlated

    return pd.DataFrame({
        "A": prices_a,
        "B": prices_b,
    }, index=dates)


# =============================================================================
# FIXTURE 1: Simple Convergence Trade
# =============================================================================
# This fixture creates a scenario where:
# 1. Formation period establishes low spread volatility
# 2. Trading period has prices diverge beyond 2σ, then converge to crossing
# 3. Results in exactly ONE predictable trade

def _create_simple_convergence_data():
    """Create data for a simple convergence scenario."""
    # Formation: 50 days of prices moving together
    formation = _create_formation_prices(50, base_price=100.0)

    # Calculate formation spread statistics (what the backtest will use)
    spread_formation = (formation["A"] / formation["A"].iloc[0]) - \
                       (formation["B"] / formation["B"].iloc[0])
    formation_std = spread_formation.std()

    # Trading: 30 days
    trading_dates = pd.bdate_range("2024-03-15", periods=30)

    # Create controlled divergence and convergence
    trading_a_close = []
    trading_b_close = []
    trading_a_open = []
    trading_b_open = []

    for i in range(30):
        if i < 5:
            # Days 0-4: Prices together (spread near 0)
            a_close = 100.0
            b_close = 100.0
        elif i < 10:
            # Days 5-9: A diverges upward (spread becomes positive > 2σ)
            # This should trigger a SHORT spread entry (sell A, buy B)
            a_close = 100.0 + (i - 4) * 3.0  # 103, 106, 109, 112, 115
            b_close = 100.0
        elif i < 15:
            # Days 10-14: A converges back (spread crosses zero)
            a_close = 115.0 - (i - 9) * 3.0  # 112, 109, 106, 103, 100
            b_close = 100.0
        else:
            # Days 15+: Prices stable at parity
            a_close = 100.0
            b_close = 100.0

        trading_a_close.append(a_close)
        trading_b_close.append(b_close)
        # Open prices slightly different
        trading_a_open.append(a_close - 0.25)
        trading_b_open.append(b_close - 0.25)

    trading_close = pd.DataFrame({
        "A": trading_a_close,
        "B": trading_b_close,
    }, index=trading_dates)

    trading_open = pd.DataFrame({
        "A": trading_a_open,
        "B": trading_b_open,
    }, index=trading_dates)

    return formation, trading_close, trading_open, formation_std


_formation, _trading_close, _trading_open, _formation_std = _create_simple_convergence_data()

SIMPLE_CONVERGENCE_FIXTURE = {
    "formation_close": _formation,
    "trading_close": _trading_close,
    "trading_open": _trading_open,
    "pair": ("A", "B"),
    "formation_std": _formation_std,
    # Expected outcomes (approximately - exact values depend on normalization)
    "expected_min_trades": 1,
    "expected_max_trades": 2,
    "description": "Simple divergence then convergence - should generate 1-2 trades",
}


# =============================================================================
# FIXTURE 2: Known Metrics
# =============================================================================
# Pre-calculated metrics for manual verification

KNOWN_METRICS_FIXTURE = {
    "trades_data": [
        # Trade 1: Win (+$500)
        {
            "pnl": 500.0,
            "pnl_pct": 0.05,
            "direction": 1,
            "holding_days": 10,
        },
        # Trade 2: Loss (-$200)
        {
            "pnl": -200.0,
            "pnl_pct": -0.02,
            "direction": -1,
            "holding_days": 5,
        },
        # Trade 3: Win (+$300)
        {
            "pnl": 300.0,
            "pnl_pct": 0.03,
            "direction": 1,
            "holding_days": 8,
        },
        # Trade 4: Win (+$100)
        {
            "pnl": 100.0,
            "pnl_pct": 0.01,
            "direction": -1,
            "holding_days": 12,
        },
        # Trade 5: Loss (-$150)
        {
            "pnl": -150.0,
            "pnl_pct": -0.015,
            "direction": 1,
            "holding_days": 7,
        },
    ],
    # Manually calculated expected values
    "expected_total_trades": 5,
    "expected_total_pnl": 500 - 200 + 300 + 100 - 150,  # = 550
    "expected_win_rate": 3 / 5,  # 3 wins out of 5 = 0.6
    "expected_avg_win": (500 + 300 + 100) / 3,  # = 300
    "expected_avg_loss": (200 + 150) / 2,  # = 175
    "expected_profit_factor": (500 + 300 + 100) / (200 + 150),  # = 900/350 ≈ 2.57
    "expected_avg_holding_days": (10 + 5 + 8 + 12 + 7) / 5,  # = 8.4
    "expected_long_trades": 3,  # direction=1
    "expected_short_trades": 2,  # direction=-1
    "expected_long_win_rate": 2 / 3,  # 2 wins out of 3 longs ≈ 0.667
    "expected_short_win_rate": 1 / 2,  # 1 win out of 2 shorts = 0.5
}


# =============================================================================
# FIXTURE 3: No Trade Scenario
# =============================================================================
# Prices stay within 2σ - should generate no trades

def _create_no_trade_data():
    """Create data where spread never exceeds 2σ."""
    formation = _create_formation_prices(50, base_price=100.0)

    # Trading period with very small price movements
    trading_dates = pd.bdate_range("2024-03-15", periods=30)

    # Prices stay very close together (within 2σ)
    trading_a = [100.0 + np.sin(i * 0.1) * 0.1 for i in range(30)]
    trading_b = [100.0 + np.sin(i * 0.1) * 0.1 for i in range(30)]

    trading_close = pd.DataFrame({
        "A": trading_a,
        "B": trading_b,
    }, index=trading_dates)

    trading_open = pd.DataFrame({
        "A": [p - 0.05 for p in trading_a],
        "B": [p - 0.05 for p in trading_b],
    }, index=trading_dates)

    return formation, trading_close, trading_open


_nt_formation, _nt_trading_close, _nt_trading_open = _create_no_trade_data()

NO_TRADE_FIXTURE = {
    "formation_close": _nt_formation,
    "trading_close": _nt_trading_close,
    "trading_open": _nt_trading_open,
    "pair": ("A", "B"),
    "expected_trades": 0,
    "description": "Prices stay close - should generate no trades",
}
