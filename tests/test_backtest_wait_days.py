"""Tests for wait_days parameter in BacktestConfig.

The GGR paper discusses waiting one day after signal to trade, to account
for bid-ask bounce. This module tests both execution modes:
- wait_days=1 (default): Execute at next-day OPEN
- wait_days=0: Execute at same-day CLOSE
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, run_backtest_single_pair


def create_test_data():
    """Create test data that will generate a trade."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B")

    # Formation period: 30 days with low volatility
    formation_dates = dates[:30]
    formation_a = pd.Series(100.0, index=formation_dates)
    formation_b = pd.Series(100.0, index=formation_dates)

    # Add DIFFERENT variations to get non-zero spread std (Bug #8 fix requires this)
    formation_a = formation_a + np.sin(np.arange(30)) * 0.5
    formation_b = formation_b + np.cos(np.arange(30)) * 0.5  # Different pattern!

    formation_close = pd.DataFrame({
        "A": formation_a,
        "B": formation_b,
    })

    # Trading period: 30 days with divergence then convergence
    trading_dates = dates[30:]

    # Create divergence that triggers entry, then convergence
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
            # Divergence - A rises, B stays flat (triggers short spread entry)
            trading_a_close[date] = 100.5 + (i - 5) * 2.0  # Rising
            trading_b_close[date] = 100.5                   # Flat
            trading_a_open[date] = trading_a_close[date] - 0.5  # Open slightly lower
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


class TestWaitDaysParameter:
    """Tests for wait_days configuration parameter."""

    def test_wait_days_default_is_one(self):
        """Default wait_days should be 1."""
        config = BacktestConfig()
        assert config.wait_days == 1

    def test_wait_days_can_be_set_to_zero(self):
        """wait_days should be configurable to 0."""
        config = BacktestConfig(wait_days=0)
        assert config.wait_days == 0

    def test_wait_days_zero_vs_one_different_entry_prices(self):
        """Entry prices should differ between wait_days=0 and wait_days=1."""
        formation_close, trading_close, trading_open = create_test_data()
        pair = ("A", "B")

        # Run with wait_days=1 (next-day open)
        config_wait_1 = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
            wait_days=1,
        )
        result_wait_1 = run_backtest_single_pair(
            formation_close, trading_close, trading_open, pair, config_wait_1
        )

        # Run with wait_days=0 (same-day close)
        config_wait_0 = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
            wait_days=0,
        )
        result_wait_0 = run_backtest_single_pair(
            formation_close, trading_close, trading_open, pair, config_wait_0
        )

        # Both MUST have trades - test data is designed to trigger entries
        assert len(result_wait_1.trades) > 0, \
            "Test data should trigger trade with wait_days=1"
        assert len(result_wait_0.trades) > 0, \
            "Test data should trigger trade with wait_days=0"

        trade_1 = result_wait_1.trades[0]
        trade_0 = result_wait_0.trades[0]

        # Entry prices should be different (open vs close)
        # wait_days=1 uses next-day OPEN
        # wait_days=0 uses same-day CLOSE
        assert trade_1.entry_price_a != trade_0.entry_price_a or \
               trade_1.entry_price_b != trade_0.entry_price_b, \
               "Entry prices should differ between wait modes"

    def test_wait_days_affects_entry_date(self):
        """Entry date should differ by one day between wait_days=0 and wait_days=1."""
        formation_close, trading_close, trading_open = create_test_data()
        pair = ("A", "B")

        config_wait_1 = BacktestConfig(entry_threshold=2.0, max_holding_days=50, wait_days=1)
        config_wait_0 = BacktestConfig(entry_threshold=2.0, max_holding_days=50, wait_days=0)

        result_wait_1 = run_backtest_single_pair(
            formation_close, trading_close, trading_open, pair, config_wait_1
        )
        result_wait_0 = run_backtest_single_pair(
            formation_close, trading_close, trading_open, pair, config_wait_0
        )

        # Both MUST have trades
        assert len(result_wait_1.trades) > 0, "Test data should trigger trade with wait_days=1"
        assert len(result_wait_0.trades) > 0, "Test data should trigger trade with wait_days=0"

        # wait_days=0 should enter on signal day
        # wait_days=1 should enter day after signal
        entry_date_0 = result_wait_0.trades[0].entry_date
        entry_date_1 = result_wait_1.trades[0].entry_date

        # Entry date with wait_days=1 should be strictly later
        assert entry_date_1 > entry_date_0, \
            f"wait_days=1 entry ({entry_date_1}) should be after wait_days=0 entry ({entry_date_0})"

    def test_wait_days_one_uses_open_prices(self):
        """wait_days=1 should execute at OPEN prices."""
        formation_close, trading_close, trading_open = create_test_data()
        pair = ("A", "B")

        config = BacktestConfig(entry_threshold=2.0, max_holding_days=50, wait_days=1)
        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, pair, config
        )

        # Must have trades
        assert len(result.trades) > 0, "Test data should trigger at least one trade"

        trade = result.trades[0]
        entry_date = trade.entry_date

        # Entry price should match OPEN price on entry date
        expected_open_a = trading_open.loc[entry_date, "A"]
        expected_open_b = trading_open.loc[entry_date, "B"]

        assert trade.entry_price_a == expected_open_a, \
            f"Entry price A should be open price: {expected_open_a}, got {trade.entry_price_a}"
        assert trade.entry_price_b == expected_open_b, \
            f"Entry price B should be open price: {expected_open_b}, got {trade.entry_price_b}"

    def test_wait_days_zero_uses_close_prices(self):
        """wait_days=0 should execute at CLOSE prices."""
        formation_close, trading_close, trading_open = create_test_data()
        pair = ("A", "B")

        config = BacktestConfig(entry_threshold=2.0, max_holding_days=50, wait_days=0)
        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, pair, config
        )

        # Must have trades
        assert len(result.trades) > 0, "Test data should trigger at least one trade"

        trade = result.trades[0]
        entry_date = trade.entry_date

        # Entry price should match CLOSE price on entry date
        expected_close_a = trading_close.loc[entry_date, "A"]
        expected_close_b = trading_close.loc[entry_date, "B"]

        assert trade.entry_price_a == expected_close_a, \
            f"Entry price A should be close price: {expected_close_a}, got {trade.entry_price_a}"
        assert trade.entry_price_b == expected_close_b, \
            f"Entry price B should be close price: {expected_close_b}, got {trade.entry_price_b}"

    def test_both_wait_modes_produce_valid_trades(self):
        """Both wait modes should produce valid trades with positive shares."""
        formation_close, trading_close, trading_open = create_test_data()
        pair = ("A", "B")

        for wait_days in [0, 1]:
            config = BacktestConfig(
                entry_threshold=2.0,
                max_holding_days=50,
                wait_days=wait_days,
            )
            result = run_backtest_single_pair(
                formation_close, trading_close, trading_open, pair, config
            )

            # Must have at least one trade - test data is designed to trigger entries
            assert len(result.trades) > 0, \
                f"wait_days={wait_days} should produce at least one trade with test data"

            # All trades should have valid data
            for trade in result.trades:
                assert trade.shares_a > 0, "Shares A should be positive"
                assert trade.shares_b > 0, "Shares B should be positive"
                assert trade.entry_price_a > 0, "Entry price A should be positive"
                assert trade.entry_price_b > 0, "Entry price B should be positive"
                assert not pd.isna(trade.pnl), "P&L should not be NaN"
