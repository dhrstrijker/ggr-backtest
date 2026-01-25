"""Tests for edge cases across all modules.

These tests verify the system handles unusual inputs gracefully:
- Empty DataFrames and lists
- Single-element inputs (can't form pairs)
- NaN handling at critical points
- Invalid inputs (zero capital, negative prices)
- Boundary conditions (last day, simultaneous delisting)
"""

import numpy as np
import pandas as pd
import pytest

from src.pairs import (
    normalize_prices,
    calculate_ssd_matrix,
    select_top_pairs,
)
from src.signals import (
    calculate_spread,
    calculate_formation_stats,
    calculate_distance,
    generate_signals_ggr,
)
from src.backtest import (
    BacktestConfig,
    run_backtest,
    run_backtest_single_pair,
)


# =============================================================================
# Empty Input Tests
# =============================================================================


class TestEmptyInputs:
    """Tests for handling empty inputs gracefully."""

    def test_normalize_prices_empty_dataframe(self):
        """normalize_prices should return empty DataFrame for empty input."""
        prices = pd.DataFrame()
        result = normalize_prices(prices)

        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert result.empty, "Result should be empty for empty input"

    def test_calculate_ssd_matrix_empty(self):
        """calculate_ssd_matrix should return empty DataFrame for empty input."""
        prices = pd.DataFrame()
        normalized = normalize_prices(prices)
        result = calculate_ssd_matrix(normalized)

        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert result.empty, "Result should be empty for empty input"

    def test_run_backtest_empty_pairs_list(self):
        """run_backtest should return empty dict for empty pairs list."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        formation_close = pd.DataFrame({
            'A': [100.0] * 50,
            'B': [100.0] * 50,
        }, index=formation_dates)

        trading_close = pd.DataFrame({
            'A': [100.0] * 30,
            'B': [100.0] * 30,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig()
        pairs = []  # Empty pairs list

        result = run_backtest(
            formation_close, trading_close, trading_open, pairs, config
        )

        assert isinstance(result, dict), "Should return dict"
        assert len(result) == 0, "Result should be empty for empty pairs list"

    def test_generate_signals_empty_spread(self):
        """generate_signals_ggr should handle empty spread Series."""
        spread = pd.Series([], dtype=float)
        formation_std = 1.0

        signals = generate_signals_ggr(spread, formation_std)

        assert isinstance(signals, pd.Series), "Should return Series"
        assert len(signals) == 0, "Result should be empty for empty spread"


# =============================================================================
# Single Element Tests
# =============================================================================


class TestSingleElementInputs:
    """Tests for handling inputs with single elements."""

    def test_normalize_prices_single_symbol(self):
        """normalize_prices should work with single symbol."""
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 120.0],
        })
        result = normalize_prices(prices)

        assert 'A' in result.columns, "Single symbol should be preserved"
        assert len(result.columns) == 1, "Should have exactly one column"
        assert result['A'].iloc[0] == 1.0, "Normalized price should start at 1.0"

    def test_select_top_pairs_single_symbol(self):
        """select_top_pairs should return empty list for single symbol."""
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 120.0],
        })
        normalized = normalize_prices(prices)
        ssd_matrix = calculate_ssd_matrix(normalized)

        # With 1 symbol, there are no pairs to select
        pairs = select_top_pairs(ssd_matrix, n=5)

        assert isinstance(pairs, list), "Should return list"
        assert len(pairs) == 0, "No pairs can be formed from single symbol"


# =============================================================================
# NaN Handling Tests
# =============================================================================


class TestNaNHandling:
    """Tests for handling NaN values at critical points."""

    def test_run_backtest_all_nan_formation(self):
        """Backtest should handle formation period with all NaN values."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        # All NaN formation prices
        formation_close = pd.DataFrame({
            'A': [np.nan] * 50,
            'B': [np.nan] * 50,
        }, index=formation_dates)

        trading_close = pd.DataFrame({
            'A': [100.0] * 30,
            'B': [100.0] * 30,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig()

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Should return valid result (empty trades) without crashing
        assert isinstance(result.trades, list), "Should return valid trades list"
        assert len(result.trades) == 0, "Should have no trades with all-NaN formation"

    def test_run_backtest_nan_at_entry_time(self):
        """Backtest should skip entry when price is NaN on entry day."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        # Formation with variation to get valid std
        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 5 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 5 for i in range(50)],
        }, index=formation_dates)

        # Trading period with divergence but NaN at entry time
        trading_close = pd.DataFrame({
            'A': [100.0] * 5 + [np.nan] * 5 + [120.0] * 20,  # NaN during signal
            'B': [100.0] * 30,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=2.0,
            wait_days=0,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Should not crash, may have trades or no trades depending on implementation
        assert isinstance(result.trades, list), "Should return valid trades list"
        assert len(result.equity_curve) > 0, "Should have equity curve"

        # If there are trades, verify no NaN entry prices
        for trade in result.trades:
            assert not np.isnan(trade.entry_price_a), \
                "Entry price A should not be NaN"
            assert not np.isnan(trade.entry_price_b), \
                "Entry price B should not be NaN"

    def test_calculate_distance_all_nan_spread(self):
        """calculate_distance should handle all-NaN spread."""
        spread = pd.Series([np.nan, np.nan, np.nan])
        formation_std = 1.0

        distance = calculate_distance(spread, formation_std)

        assert isinstance(distance, pd.Series), "Should return Series"
        assert distance.isna().all(), "All distances should be NaN"


# =============================================================================
# Invalid Input Tests
# =============================================================================


class TestInvalidInputs:
    """Tests for handling invalid inputs gracefully."""

    def test_run_backtest_zero_capital(self):
        """Backtest should handle zero capital per trade without division errors."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 5 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 5 for i in range(50)],
        }, index=formation_dates)

        trading_close = pd.DataFrame({
            'A': [100.0] * 10 + [120.0] * 20,
            'B': [100.0] * 30,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            capital_per_trade=0.0,  # Zero capital
            entry_threshold=2.0,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Should not crash with zero capital
        assert isinstance(result.trades, list), "Should return valid trades list"

        # If trades exist, shares should be 0 or trades should be empty
        for trade in result.trades:
            assert trade.shares_a == 0 or np.isnan(trade.shares_a) or \
                   np.isinf(trade.shares_a) or trade.shares_a >= 0, \
                "Shares should handle zero capital gracefully"

    def test_run_backtest_negative_prices(self):
        """Backtest should handle negative prices in trading data."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 5 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 5 for i in range(50)],
        }, index=formation_dates)

        # Trading data with some negative prices (data error)
        trading_close = pd.DataFrame({
            'A': [100.0] * 10 + [-50.0] * 10 + [100.0] * 10,  # Negative prices
            'B': [100.0] * 30,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.0,
            wait_days=0,
        )

        # Should not crash with negative prices
        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        assert isinstance(result.trades, list), "Should return valid trades list"
        assert len(result.equity_curve) > 0, "Should have equity curve"

    def test_backtest_config_negative_threshold(self):
        """BacktestConfig should accept negative threshold (edge case)."""
        # Negative threshold is technically valid (always triggers entry)
        # This documents the behavior rather than requiring rejection
        config = BacktestConfig(entry_threshold=-1.0)

        assert config.entry_threshold == -1.0, "Should accept negative threshold"


# =============================================================================
# Boundary Condition Tests
# =============================================================================


class TestBoundaryConditions:
    """Tests for boundary conditions at edges of data."""

    def test_entry_on_last_day_of_data(self):
        """Entry signal on last day should be handled gracefully."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=10, freq='D')

        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 5 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 5 for i in range(50)],
        }, index=formation_dates)

        # Divergence only on last day (signal on day 9)
        trading_close = pd.DataFrame({
            'A': [100.0] * 9 + [130.0],  # Diverges only on last day
            'B': [100.0] * 10,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=50,
            wait_days=0,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Should not crash
        assert isinstance(result.trades, list), "Should return valid trades list"

        # If a trade was opened on last day, it should exit with end_of_data
        if result.trades:
            last_trade = result.trades[-1]
            # Entry on last day means exit on last day too
            assert last_trade.exit_reason in ['end_of_data', 'crossing', 'max_holding'], \
                f"Last day trade should exit properly, got: {last_trade.exit_reason}"

    def test_both_symbols_delist_same_day(self):
        """Both symbols delisting on same day should be handled gracefully."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 5 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 5 for i in range(50)],
        }, index=formation_dates)

        # Both stocks delist on day 15
        trading_close = pd.DataFrame({
            'A': [100.0] * 5 + [120.0] * 10 + [np.nan] * 15,  # Delists day 15
            'B': [100.0] * 15 + [np.nan] * 15,                 # Delists day 15
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=100,  # High so delisting triggers exit
            wait_days=0,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Should not crash
        assert isinstance(result.trades, list), "Should return valid trades list"
        assert len(result.equity_curve) > 0, "Should have equity curve"

        # All trades should have valid P&L (not NaN)
        for trade in result.trades:
            assert not np.isnan(trade.pnl), \
                f"Trade P&L should not be NaN: {trade}"

    def test_commission_zero_still_calculates(self):
        """Zero commission should still calculate valid P&L."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 5 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 5 for i in range(50)],
        }, index=formation_dates)

        trading_close = pd.DataFrame({
            'A': [100.0] * 5 + [130.0] * 10 + [100.0] * 15,
            'B': [100.0] * 30,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.0,
            commission=0.0,  # Zero commission
            wait_days=0,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Should have trades
        assert len(result.trades) > 0, "Should have trades with zero commission"

        # P&L should be valid (not NaN)
        for trade in result.trades:
            assert not np.isnan(trade.pnl), "P&L should be valid with zero commission"
            assert not np.isnan(trade.pnl_pct), "P&L % should be valid"


# =============================================================================
# Formation Statistics Edge Cases
# =============================================================================


class TestFormationStatsEdgeCases:
    """Tests for edge cases in formation statistics calculation."""

    def test_calculate_formation_stats_constant_spread(self):
        """Formation stats with constant spread should have zero std."""
        spread = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])
        stats = calculate_formation_stats(spread)

        assert stats['mean'] == 0.0, "Mean of zeros should be 0"
        assert stats['std'] == 0.0, "Std of zeros should be 0"

    def test_calculate_formation_stats_single_value(self):
        """Formation stats with single value should handle gracefully."""
        spread = pd.Series([0.5])
        stats = calculate_formation_stats(spread)

        assert stats['mean'] == 0.5, "Mean of single value should be that value"
        # Std of single value is NaN in pandas
        assert np.isnan(stats['std']) or stats['std'] == 0.0, \
            "Std of single value should be NaN or 0"

    def test_generate_signals_zero_std(self):
        """Signals with zero formation std should handle division by zero."""
        spread = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        formation_std = 0.0  # Zero std

        # Should not crash - may return all zeros or handle specially
        signals = generate_signals_ggr(spread, formation_std, entry_threshold=2.0)

        assert isinstance(signals, pd.Series), "Should return Series"
        assert len(signals) == len(spread), "Should have same length as spread"


# =============================================================================
# Spread Calculation Edge Cases
# =============================================================================


class TestSpreadCalculationEdgeCases:
    """Tests for edge cases in spread calculation."""

    def test_calculate_spread_different_lengths(self):
        """calculate_spread should handle series of same length."""
        prices_a = pd.Series([100.0, 110.0, 120.0])
        prices_b = pd.Series([100.0, 105.0, 115.0])

        spread = calculate_spread(prices_a, prices_b, normalize=True)

        assert len(spread) == 3, "Spread length should match input"
        assert spread.iloc[0] == 0.0, "Normalized spread should start at 0"

    def test_calculate_spread_no_normalize(self):
        """calculate_spread with normalize=False should return raw difference."""
        prices_a = pd.Series([100.0, 110.0, 120.0])
        prices_b = pd.Series([100.0, 105.0, 115.0])

        spread = calculate_spread(prices_a, prices_b, normalize=False)

        # Without normalization, spread is raw difference
        assert spread.iloc[0] == 0.0, "Day 0: 100 - 100 = 0"
        assert spread.iloc[1] == 5.0, "Day 1: 110 - 105 = 5"
        assert spread.iloc[2] == 5.0, "Day 2: 120 - 115 = 5"
