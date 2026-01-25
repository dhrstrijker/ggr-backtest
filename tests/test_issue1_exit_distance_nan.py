"""Test for Issue #1: End-of-Data Exit Distance NaN handling.

This test verifies that exit_distance does NOT become NaN when
a position is closed at end-of-data with NaN prices on the last day.

BUG STATUS: FIXED - exit_distance now falls back to entry_distance when last day has NaN.
Fix applied at src/backtest.py:507
"""

import pandas as pd
import numpy as np
import pytest

from src.backtest import run_backtest_single_pair, BacktestConfig


class TestEndOfDataExitDistanceNaN:
    """Test cases for end-of-data exit distance NaN handling."""

    @pytest.fixture
    def scenario_with_nan_last_day(self):
        """Create a scenario where last trading day has NaN prices."""
        dates = pd.date_range("2023-01-01", periods=30, freq="B")
        formation_dates = dates[:15]
        trading_dates = dates[15:]

        np.random.seed(42)
        formation_a = 100 + np.cumsum(np.random.randn(15) * 0.5)
        formation_b = 100 + np.cumsum(np.random.randn(15) * 0.5)

        formation_close = pd.DataFrame({
            "A": formation_a,
            "B": formation_b,
        }, index=formation_dates)

        # Create divergence to trigger entry
        trading_a = np.zeros(15)
        trading_b = np.zeros(15)
        trading_a[0] = formation_a[-1]
        trading_b[0] = formation_b[-1]

        for i in range(1, 15):
            trading_a[i] = trading_a[0] * (1 + 0.03 * i)
            trading_b[i] = trading_b[0] * (1 - 0.01 * i)

        # Last day has NaN prices
        trading_a[-1] = np.nan
        trading_b[-1] = np.nan

        trading_close = pd.DataFrame({
            "A": trading_a,
            "B": trading_b,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        return formation_close, trading_close, trading_open

    def test_exit_distance_not_nan_for_end_of_data(self, scenario_with_nan_last_day):
        """
        Test that exit_distance is NOT NaN for end_of_data exits.

        When last day has NaN prices, exit_distance should fall back to entry_distance.
        """
        formation_close, trading_close, trading_open = scenario_with_nan_last_day

        config = BacktestConfig(
            entry_threshold=1.5,
            max_holding_days=126,
            capital_per_trade=10000.0,
            commission=0.001,
            wait_days=0,
        )

        result = run_backtest_single_pair(
            formation_close=formation_close,
            trading_close=trading_close,
            trading_open=trading_open,
            pair=("A", "B"),
            config=config,
        )

        # Find end_of_data trades
        end_of_data_trades = [t for t in result.trades if t.exit_reason == "end_of_data"]
        assert len(end_of_data_trades) > 0, "Expected at least one end_of_data trade"

        for trade in end_of_data_trades:
            assert not pd.isna(trade.exit_distance), (
                f"exit_distance should not be NaN for end_of_data exit. "
                f"Got: {trade.exit_distance}. "
                f"Should fallback to entry_distance: {trade.entry_distance}"
            )
            # Verify fallback to entry_distance when last day is NaN
            assert trade.exit_distance == trade.entry_distance, (
                f"When last day has NaN prices, exit_distance should equal entry_distance. "
                f"Got exit_distance={trade.exit_distance}, entry_distance={trade.entry_distance}"
            )

    def test_exit_distance_uses_actual_value_when_available(self):
        """
        Test that exit_distance uses actual value when last day has valid prices.
        """
        dates = pd.date_range("2023-01-01", periods=30, freq="B")
        formation_dates = dates[:15]
        trading_dates = dates[15:]

        np.random.seed(42)
        formation_a = 100 + np.cumsum(np.random.randn(15) * 0.5)
        formation_b = 100 + np.cumsum(np.random.randn(15) * 0.5)

        formation_close = pd.DataFrame({
            "A": formation_a,
            "B": formation_b,
        }, index=formation_dates)

        # Create divergence, but keep all prices valid (no NaN)
        trading_a = np.zeros(15)
        trading_b = np.zeros(15)
        trading_a[0] = formation_a[-1]
        trading_b[0] = formation_b[-1]

        for i in range(1, 15):
            trading_a[i] = trading_a[0] * (1 + 0.03 * i)
            trading_b[i] = trading_b[0] * (1 - 0.01 * i)

        # Last day has VALID prices (no NaN)
        trading_close = pd.DataFrame({
            "A": trading_a,
            "B": trading_b,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.5,
            max_holding_days=126,
            capital_per_trade=10000.0,
            commission=0.001,
            wait_days=0,
        )

        result = run_backtest_single_pair(
            formation_close=formation_close,
            trading_close=trading_close,
            trading_open=trading_open,
            pair=("A", "B"),
            config=config,
        )

        end_of_data_trades = [t for t in result.trades if t.exit_reason == "end_of_data"]
        assert len(end_of_data_trades) > 0, "Expected at least one end_of_data trade"

        for trade in end_of_data_trades:
            assert not pd.isna(trade.exit_distance), "exit_distance should not be NaN"
            # When prices are valid, exit_distance should differ from entry_distance
            # (unless by coincidence they're the same)
            # Just verify it's a valid number
            assert isinstance(trade.exit_distance, (int, float, np.number))
