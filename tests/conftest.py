"""Shared fixtures for GGR backtesting test suite."""

import sys
from pathlib import Path

# Add project root to path for imports when not installed as a package
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, Trade


# =============================================================================
# Date Fixtures
# =============================================================================


@pytest.fixture
def formation_dates():
    """50 trading days for formation period."""
    return pd.date_range('2023-01-01', periods=50, freq='D')


@pytest.fixture
def trading_dates():
    """30 trading days for trading period."""
    return pd.date_range('2023-03-01', periods=30, freq='D')


@pytest.fixture
def sample_dates():
    """756 trading days (~3 years) for staggered tests."""
    return pd.bdate_range(start="2020-01-01", periods=756, freq="B")


@pytest.fixture
def short_sample_dates():
    """400 trading days for quick tests."""
    return pd.bdate_range(start="2020-01-01", periods=400, freq="B")


# =============================================================================
# Price Fixtures - Formation Period
# =============================================================================


@pytest.fixture
def oscillating_formation_prices(formation_dates):
    """Formation prices with sine/cosine patterns (non-zero spread std).

    Uses different patterns for A and B to ensure the spread has non-zero
    standard deviation, which is required for valid GGR signal generation.
    """
    return pd.DataFrame({
        'A': [100.0 + np.sin(i/5) * 3 for i in range(50)],
        'B': [100.0 + np.cos(i/5) * 3 for i in range(50)],
    }, index=formation_dates)


@pytest.fixture
def flat_formation_prices(formation_dates):
    """Identical formation prices (zero spread std - edge case).

    Both stocks have identical prices, resulting in zero spread std.
    This is useful for testing edge cases like division by zero.
    """
    return pd.DataFrame({
        'A': [100.0] * 50,
        'B': [100.0] * 50,
    }, index=formation_dates)


# =============================================================================
# Price Fixtures - Trading Period
# =============================================================================


@pytest.fixture
def diverging_trading_prices(trading_dates):
    """Trading prices that diverge then converge (triggers trades).

    Returns (trading_close, trading_open) tuple.
    Pattern: prices start together, A jumps to 130, then returns to 100.
    """
    trading_close = pd.DataFrame({
        'A': [100.0] * 5 + [130.0] * 10 + [100.0] * 15,
        'B': [100.0] * 30,
    }, index=trading_dates)
    trading_open = trading_close.copy()
    return trading_close, trading_open


@pytest.fixture
def diverging_trading_prices_with_gaps(trading_dates):
    """Trading prices that diverge then converge with slight open/close gaps.

    Returns (trading_close, trading_open) tuple.
    The open prices are slightly different from close prices.
    """
    trading_close = pd.DataFrame({
        'A': [100.0] * 5 + [130.0] * 10 + [100.0] * 15,
        'B': [100.0] * 30,
    }, index=trading_dates)
    trading_open = pd.DataFrame({
        'A': [100.5] * 5 + [129.5] * 10 + [100.5] * 15,
        'B': [100.5] * 30,
    }, index=trading_dates)
    return trading_close, trading_open


# =============================================================================
# Multi-Symbol Price Fixtures (for staggered backtest tests)
# =============================================================================


@pytest.fixture
def sample_prices(sample_dates):
    """Multi-symbol price data for staggered tests.

    Returns (close_prices, open_prices) tuple.
    8 symbols with correlated random walks.
    """
    np.random.seed(42)
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM"]

    close_data = {}
    open_data = {}

    for i, sym in enumerate(symbols):
        base_price = 100 + i * 20
        returns = np.random.normal(0.0003, 0.02, len(sample_dates))
        prices = base_price * np.exp(np.cumsum(returns))
        close_data[sym] = prices
        open_data[sym] = prices * (1 + np.random.normal(0, 0.001, len(sample_dates)))

    close_prices = pd.DataFrame(close_data, index=sample_dates)
    open_prices = pd.DataFrame(open_data, index=sample_dates)

    return close_prices, open_prices


@pytest.fixture
def short_sample_prices(short_sample_dates):
    """Shorter sample for quick tests.

    Returns (close_prices, open_prices) tuple.
    4 symbols with 400 days of data.
    """
    np.random.seed(42)
    symbols = ["A", "B", "C", "D"]

    close_data = {}
    open_data = {}

    for i, sym in enumerate(symbols):
        base_price = 100 + i * 10
        returns = np.random.normal(0.0003, 0.02, len(short_sample_dates))
        prices = base_price * np.exp(np.cumsum(returns))
        close_data[sym] = prices
        open_data[sym] = prices * (1 + np.random.normal(0, 0.001, len(short_sample_dates)))

    close_prices = pd.DataFrame(close_data, index=short_sample_dates)
    open_prices = pd.DataFrame(open_data, index=short_sample_dates)

    return close_prices, open_prices


@pytest.fixture
def prices_with_missing_data():
    """Prices with some symbols having missing data.

    Returns (close_prices, open_prices) tuple.
    - Symbol C starts late (NaN for first 100 days)
    - Symbol D ends early (NaN for last 50 days)
    """
    np.random.seed(42)
    dates = pd.bdate_range(start="2020-01-01", periods=500, freq="B")
    symbols = ["A", "B", "C", "D"]

    close_data = {}
    open_data = {}

    for i, sym in enumerate(symbols):
        base_price = 100 + i * 10
        returns = np.random.normal(0.0003, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        close_data[sym] = prices.copy()
        open_data[sym] = prices * (1 + np.random.normal(0, 0.001, len(dates)))

    # Add NaN values to symbol "C" (starts later)
    close_data["C"][:100] = np.nan
    open_data["C"][:100] = np.nan

    # Add NaN values to symbol "D" (ends earlier)
    close_data["D"][-50:] = np.nan
    open_data["D"][-50:] = np.nan

    close_prices = pd.DataFrame(close_data, index=dates)
    open_prices = pd.DataFrame(open_data, index=dates)

    return close_prices, open_prices


# =============================================================================
# Config Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Default backtest configuration."""
    return BacktestConfig(
        entry_threshold=2.0,
        max_holding_days=50,
        capital_per_trade=10000,
    )


@pytest.fixture
def loose_entry_config():
    """Configuration with lower entry threshold (easier trade entry)."""
    return BacktestConfig(
        entry_threshold=1.0,
        max_holding_days=50,
        capital_per_trade=10000,
    )


@pytest.fixture
def wait_zero_config():
    """Configuration with wait_days=0 (same-day execution)."""
    return BacktestConfig(
        entry_threshold=2.0,
        max_holding_days=50,
        capital_per_trade=10000,
        wait_days=0,
    )


# =============================================================================
# Factory Fixtures
# =============================================================================


@pytest.fixture
def trade_factory():
    """Factory for creating Trade objects in tests.

    Usage:
        trade = trade_factory(pnl=100.0)
        trade = trade_factory(pnl=-50.0, direction=-1, holding_days=5)
    """
    def _create(
        pnl: float,
        direction: int = 1,
        holding_days: int = 10,
        pair: tuple[str, str] = ("A", "B"),
        exit_reason: str = "crossing",
    ) -> Trade:
        entry_date = datetime(2024, 1, 1)
        exit_date = entry_date + timedelta(days=holding_days)

        return Trade(
            pair=pair,
            direction=direction,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price_a=100.0,
            entry_price_b=100.0,
            exit_price_a=100.0 + (pnl / 2 if direction == 1 else -pnl / 2),
            exit_price_b=100.0 - (pnl / 2 if direction == 1 else pnl / 2),
            shares_a=50.0,
            shares_b=50.0,
            pnl=pnl,
            pnl_pct=pnl / 10000,
            holding_days=holding_days,
            entry_distance=2.5,
            exit_distance=0.1,
            exit_reason=exit_reason,
            max_adverse_spread=2.5,
        )

    return _create


# =============================================================================
# Test Data Fixture (for wait_days tests)
# =============================================================================


@pytest.fixture
def wait_days_test_data():
    """Test data that generates trades with specific divergence pattern.

    Returns (formation_close, trading_close, trading_open) tuple.

    Formation period: 30 days with low volatility but different patterns for A/B.
    Trading period: 30 days with divergence (A rises) then convergence.
    """
    dates = pd.date_range("2024-01-01", periods=60, freq="B")

    # Formation period: 30 days with low volatility
    formation_dates = dates[:30]
    formation_a = pd.Series(100.0, index=formation_dates)
    formation_b = pd.Series(100.0, index=formation_dates)

    # Add DIFFERENT variations to get non-zero spread std
    formation_a = formation_a + np.sin(np.arange(30)) * 0.5
    formation_b = formation_b + np.cos(np.arange(30)) * 0.5

    formation_close = pd.DataFrame({
        "A": formation_a,
        "B": formation_b,
    })

    # Trading period: 30 days with divergence then convergence
    trading_dates = dates[30:]

    trading_a_close = pd.Series(index=trading_dates, dtype=float)
    trading_b_close = pd.Series(index=trading_dates, dtype=float)
    trading_a_open = pd.Series(index=trading_dates, dtype=float)
    trading_b_open = pd.Series(index=trading_dates, dtype=float)

    for i, date in enumerate(trading_dates):
        if i < 5:
            # Initial period - prices close together
            trading_a_close[date] = 100.0 + i * 0.1
            trading_b_close[date] = 100.0 + i * 0.1
            trading_a_open[date] = 100.0 + i * 0.1 - 0.05
            trading_b_open[date] = 100.0 + i * 0.1 - 0.05
        elif i < 10:
            # Divergence - A rises, B stays flat
            trading_a_close[date] = 100.5 + (i - 5) * 2.0
            trading_b_close[date] = 100.5
            trading_a_open[date] = trading_a_close[date] - 0.5
            trading_b_open[date] = 100.5
        elif i < 20:
            # Convergence - A falls back to B
            trading_a_close[date] = 108.5 - (i - 10) * 0.85
            trading_b_close[date] = 100.5
            trading_a_open[date] = trading_a_close[date] + 0.3
            trading_b_open[date] = 100.5
        else:
            # Post-convergence
            trading_a_close[date] = 100.0
            trading_b_close[date] = 100.0
            trading_a_open[date] = 100.0
            trading_b_open[date] = 100.0

    trading_close = pd.DataFrame({
        "A": trading_a_close,
        "B": trading_b_close,
    })
    trading_open = pd.DataFrame({
        "A": trading_a_open,
        "B": trading_b_open,
    })

    return formation_close, trading_close, trading_open
