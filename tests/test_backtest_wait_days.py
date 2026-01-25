"""Tests for wait_days parameter in BacktestConfig.

The GGR paper discusses waiting one day after signal to trade, to account
for bid-ask bounce. This module tests both execution modes:
- wait_days=1 (default): Execute at next-day OPEN
- wait_days=0: Execute at same-day CLOSE

Note: Uses wait_days_test_data fixture from conftest.py.
"""

import pandas as pd
import pytest

from src.backtest import BacktestConfig, run_backtest_single_pair


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

    def test_wait_days_zero_vs_one_different_entry_prices(self, wait_days_test_data):
        """Entry prices should differ between wait_days=0 and wait_days=1."""
        formation_close, trading_close, trading_open = wait_days_test_data
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
        # At least one stock's entry price must differ (both would be even stronger)
        price_a_differs = trade_1.entry_price_a != trade_0.entry_price_a
        price_b_differs = trade_1.entry_price_b != trade_0.entry_price_b

        assert price_a_differs or price_b_differs, \
            f"Entry prices should differ between wait modes. " \
            f"wait_days=0: A={trade_0.entry_price_a}, B={trade_0.entry_price_b}. " \
            f"wait_days=1: A={trade_1.entry_price_a}, B={trade_1.entry_price_b}"

        # Additionally verify entry dates differ (this is the primary behavioral difference)
        assert trade_1.entry_date != trade_0.entry_date, \
            f"Entry dates should differ: wait_days=0 on {trade_0.entry_date}, " \
            f"wait_days=1 on {trade_1.entry_date}"

    def test_wait_days_affects_entry_date(self, wait_days_test_data):
        """Entry date should differ by one day between wait_days=0 and wait_days=1."""
        formation_close, trading_close, trading_open = wait_days_test_data
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

    def test_wait_days_one_uses_open_prices(self, wait_days_test_data):
        """wait_days=1 should execute at OPEN prices."""
        formation_close, trading_close, trading_open = wait_days_test_data
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

    def test_wait_days_zero_uses_close_prices(self, wait_days_test_data):
        """wait_days=0 should execute at CLOSE prices."""
        formation_close, trading_close, trading_open = wait_days_test_data
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

    def test_both_wait_modes_produce_valid_trades(self, wait_days_test_data):
        """Both wait modes should produce valid trades with positive shares."""
        formation_close, trading_close, trading_open = wait_days_test_data
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
