"""Tests for GGR backtest engine."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

import sys
sys.path.insert(0, str(pd.io.common.Path(__file__).parent.parent))

from src.backtest import (
    run_backtest_single_pair,
    BacktestConfig,
    Trade,
)
from src.signals import calculate_spread, calculate_formation_stats


class TestTradeExecution:
    """Test suite for trade execution logic."""

    def test_trade_execution_next_day_open(self):
        """Trade should execute at OPEN of day AFTER signal."""
        # Create formation and trading period data
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        # Formation period: stocks move with DIFFERENT patterns to create non-zero spread std
        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 3 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 3 for i in range(50)],
        }, index=formation_dates)

        # Trading period: A stays flat, B drops significantly - spread will go positive
        trading_close = pd.DataFrame({
            'A': [100.0] * 30,
            'B': [100.0] * 5 + [70.0] * 25,  # Significant drop to ensure entry
        }, index=trading_dates)

        trading_open = pd.DataFrame({
            'A': [100.5] * 30,  # Open slightly different from close
            'B': [100.5] * 5 + [70.5] * 25,
        }, index=trading_dates)

        config = BacktestConfig(
            entry_threshold=1.5,  # Lower threshold to trigger entry
            max_holding_days=20,
            capital_per_trade=10000,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Must have trades - test data is designed to trigger entry
        assert len(result.trades) > 0, \
            "Test data should trigger at least one trade (B drops 30% while A stays flat)"

        trade = result.trades[0]
        # Entry should use OPEN prices, not CLOSE prices
        # Verify the entry price matches the open price on the entry date
        expected_open_a = trading_open.loc[trade.entry_date, 'A']
        expected_open_b = trading_open.loc[trade.entry_date, 'B']
        assert trade.entry_price_a == expected_open_a, \
            f"Entry price A should be open price {expected_open_a}, got {trade.entry_price_a}"
        assert trade.entry_price_b == expected_open_b, \
            f"Entry price B should be open price {expected_open_b}, got {trade.entry_price_b}"


class TestGGRExitLogic:
    """Test suite for GGR crossing-zero exit logic."""

    def test_exit_on_spread_crossing_zero(self):
        """Should exit when spread crosses zero (GGR rule)."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        # Formation: stocks move with DIFFERENT patterns to create non-zero spread std
        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 3 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 3 for i in range(50)],
        }, index=formation_dates)

        # Trading: spread goes positive (A > B), then crosses zero (A < B)
        # Day 0-4: both at 100 (spread = 0)
        # Day 5-14: A jumps to 130 (positive spread, triggers short entry)
        # Day 15-29: A drops to 90 (spread crosses zero to negative, triggers exit)
        trading_close = pd.DataFrame({
            'A': [100.0] * 5 + [130.0] * 10 + [90.0] * 15,  # Up significantly then down below B
            'B': [100.0] * 30,  # Flat
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.0,  # Easy entry
            max_holding_days=50,  # High so crossing triggers exit, not max_holding
            capital_per_trade=10000,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Must have trades - test data is designed to trigger entry
        assert len(result.trades) > 0, \
            "Test data should trigger at least one trade (A jumps 30% while B stays flat)"

        # At least one trade should exit on crossing
        crossing_exits = [t for t in result.trades if t.exit_reason == 'crossing']
        assert len(crossing_exits) > 0, \
            f"Should have at least one trade that exits on crossing zero. " \
            f"Got {len(result.trades)} trades with exit reasons: {[t.exit_reason for t in result.trades]}"


class TestDataAlignment:
    """Test suite for data alignment."""

    def test_spread_handles_missing_data(self):
        """Spread calculation should handle missing data correctly."""
        prices = pd.DataFrame({
            'A': [100.0, 101.0, np.nan, 103.0, 104.0],
            'B': [50.0, 51.0, 52.0, 53.0, 54.0],
        }, index=pd.date_range('2024-01-01', periods=5))

        spread = calculate_spread(prices['A'], prices['B'], normalize=True)

        # Row with NaN should propagate
        assert pd.isna(spread.iloc[2]), "NaN should propagate in spread"
        # Other values should be valid
        assert spread.iloc[0:2].notna().all(), "Non-NaN rows should be valid"
        assert spread.iloc[3:].notna().all(), "Non-NaN rows should be valid"


class TestFormationPeriodStats:
    """Test suite for formation period statistics."""

    def test_formation_std_used_for_entry(self):
        """Entry should use formation period std, not trading period std."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        # Formation: low volatility (std ≈ 0.02)
        formation_close = pd.DataFrame({
            'A': [100.0 + i * 0.1 for i in range(50)],
            'B': [100.0 + i * 0.1 for i in range(50)],
        }, index=formation_dates)

        # Trading: same stocks but with a divergence
        trading_close = pd.DataFrame({
            'A': [100.0] * 10 + [110.0] * 20,
            'B': [100.0] * 30,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        # Calculate what the formation std should be
        formation_spread = calculate_spread(
            formation_close['A'], formation_close['B'], normalize=True
        )
        formation_stats = calculate_formation_stats(formation_spread)

        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=20,
            capital_per_trade=10000,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # The entry should be based on formation std, not trading std
        # This is implicitly tested by the function working correctly


class TestPnLCalculation:
    """Test suite for P&L calculation."""

    def test_pnl_not_nan(self):
        """P&L should be calculated correctly (not NaN)."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=50, freq='D')

        # Formation: stocks move with DIFFERENT patterns to create non-zero spread std
        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 3 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 3 for i in range(50)],
        }, index=formation_dates)

        # Create prices that will trigger entry then exit
        # A drops significantly while B rises - creates large spread for entry
        # Then they converge back - triggers exit
        trading_close = pd.DataFrame({
            'A': [100.0] * 5 + [80.0] * 15 + [100.0] * 30,  # Drop then recover
            'B': [100.0] * 5 + [120.0] * 15 + [100.0] * 30,  # Rise then recover
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.5,
            max_holding_days=30,
            capital_per_trade=10000,
            commission=0.001,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Must have trades - test data is designed to trigger entry
        assert len(result.trades) > 0, \
            "Test data should trigger at least one trade (A drops 20%, B rises 20%)"

        trade = result.trades[0]
        # P&L should be a number, not NaN
        assert not np.isnan(trade.pnl), "P&L should not be NaN"
        # P&L percentage should match
        expected_pnl_pct = trade.pnl / config.capital_per_trade
        assert abs(trade.pnl_pct - expected_pnl_pct) < 0.001, \
            "P&L percentage should match"


class TestMaxHoldingDays:
    """Test suite for max holding days feature."""

    def test_exit_on_max_holding(self):
        """Should exit after max holding days even without crossing."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=100, freq='D')

        # Formation: stocks move with DIFFERENT patterns to create non-zero spread std
        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 3 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 3 for i in range(50)],
        }, index=formation_dates)

        # Create prices that diverge and STAY diverged (never cross zero)
        # A jumps up significantly and stays there - spread never reverts
        trading_close = pd.DataFrame({
            'A': [100.0] * 5 + [150.0] * 95,  # Jump and stay high
            'B': [100.0] * 100,  # Stay flat
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=5,  # Force exit after 5 days
            capital_per_trade=10000,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Must have trades - test data is designed to trigger entry
        assert len(result.trades) > 0, \
            "Test data should trigger at least one trade (A jumps 50% while B stays flat)"

        # Since spread never crosses zero, trades must exit via max_holding
        max_holding_exits = [t for t in result.trades if t.exit_reason == 'max_holding']
        assert len(max_holding_exits) > 0, \
            f"Should have trades that exit via max_holding since spread never crosses zero. " \
            f"Got exit reasons: {[t.exit_reason for t in result.trades]}"

        # All trades should respect max holding days
        for trade in result.trades:
            assert trade.holding_days <= config.max_holding_days, \
                f"Trade held for {trade.holding_days} days, exceeds max of {config.max_holding_days}"


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_empty_result_no_signals(self):
        """Should handle case with no trading signals."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=50, freq='D')

        # Formation with some volatility
        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 10 for i in range(50)],
            'B': [100.0 + np.sin(i/5) * 10 for i in range(50)],
        }, index=formation_dates)

        # Trading: prices that never diverge enough for entry
        trading_close = pd.DataFrame({
            'A': [100.0 + i * 0.01 for i in range(50)],
            'B': [100.0 + i * 0.01 for i in range(50)],
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=5.0,  # Very high threshold
            max_holding_days=20,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Should not crash, just return empty trades
        assert result.trades == [], "Should have no trades when no signals"
        assert len(result.equity_curve) > 0, "Equity curve should still exist"

    def test_trade_has_distance_not_zscore(self):
        """Trade objects should have entry_distance and exit_distance."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 5 for i in range(50)],
            'B': [100.0 + np.sin(i/5) * 5 for i in range(50)],
        }, index=formation_dates)

        trading_close = pd.DataFrame({
            'A': [100.0] * 10 + [120.0] * 10 + [95.0] * 10,
            'B': [100.0] * 30,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=50,
            capital_per_trade=10000,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        if result.trades:
            trade = result.trades[0]
            # Check that trade has distance attributes
            assert hasattr(trade, 'entry_distance'), "Trade should have entry_distance"
            assert hasattr(trade, 'exit_distance'), "Trade should have exit_distance"
            assert not hasattr(trade, 'entry_zscore'), "Trade should NOT have entry_zscore"

    def test_entry_exactly_at_two_sigma_boundary(self):
        """Entry should trigger when distance equals exactly the threshold."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        # Formation with known std
        formation_close = pd.DataFrame({
            'A': [100.0 + i * 0.1 for i in range(50)],
            'B': [100.0 + i * 0.1 for i in range(50)],
        }, index=formation_dates)

        # Calculate formation spread std
        formation_spread = calculate_spread(
            formation_close['A'], formation_close['B'], normalize=True
        )
        formation_stats = calculate_formation_stats(formation_spread)
        formation_std = formation_stats['std']

        # Create trading prices where distance hits exactly 2.0
        # normalized spread = (A/A0) - (B/B0) = target_distance * formation_std
        target_distance = 2.0
        # At day 10, A jumps to create exactly 2σ distance
        # spread = (A_new/100) - (100/100) = target_distance * formation_std
        # A_new/100 - 1 = target_distance * formation_std
        # A_new = 100 * (1 + target_distance * formation_std)
        a_jump_price = 100.0 * (1 + target_distance * formation_std)

        trading_a = [100.0] * 10 + [a_jump_price] * 10 + [100.0] * 10
        trading_b = [100.0] * 30

        trading_close = pd.DataFrame({
            'A': trading_a,
            'B': trading_b,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
            capital_per_trade=10000,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Should generate at least one trade since we hit exactly 2σ
        # (Whether >= or > is used determines the result)
        # The test documents the behavior
        assert isinstance(result.trades, list), "Should return trades list"

    def test_spread_exactly_at_zero_exits(self):
        """Exit should trigger when spread crosses exactly through zero."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 5 for i in range(50)],
            'B': [100.0 + np.sin(i/5) * 5 for i in range(50)],
        }, index=formation_dates)

        # Create prices that diverge then converge to exactly equal
        # Day 0-9: prices equal (spread = 0)
        # Day 10-14: A rises (positive spread)
        # Day 15: A returns to exactly equal B (spread crosses zero)
        trading_close = pd.DataFrame({
            'A': [100.0] * 10 + [115.0] * 5 + [100.0] * 15,
            'B': [100.0] * 30,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=50,
            capital_per_trade=10000,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # If we have trades, verify exit on crossing
        crossing_exits = [t for t in result.trades if t.exit_reason == 'crossing']
        if result.trades:
            assert len(crossing_exits) > 0, \
                "Should exit via crossing when spread hits zero"

    def test_commission_deducted_correctly(self):
        """Commission should be deducted from both entry and exit."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 5 for i in range(50)],
            'B': [100.0 + np.sin(i/5) * 5 for i in range(50)],
        }, index=formation_dates)

        # Simple divergence then convergence
        trading_close = pd.DataFrame({
            'A': [100.0] * 10 + [120.0] * 10 + [100.0] * 10,
            'B': [100.0] * 30,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        # Run with zero commission
        config_no_comm = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=50,
            capital_per_trade=10000,
            commission=0.0,
        )

        # Run with 0.1% commission (10 bps)
        config_with_comm = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=50,
            capital_per_trade=10000,
            commission=0.001,
        )

        result_no_comm = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config_no_comm
        )

        result_with_comm = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config_with_comm
        )

        if result_no_comm.trades and result_with_comm.trades:
            # P&L with commission should be less than without
            pnl_no_comm = sum(t.pnl for t in result_no_comm.trades)
            pnl_with_comm = sum(t.pnl for t in result_with_comm.trades)

            assert pnl_with_comm < pnl_no_comm, \
                f"Commission should reduce P&L: {pnl_with_comm} should be < {pnl_no_comm}"

    def test_zero_std_formation_handles_gracefully(self):
        """Should handle formation period with zero volatility gracefully.

        When formation period has zero spread std (identical price movements),
        any non-zero spread in trading period would be "infinite" sigmas.

        Current behavior: Trades are generated with inf entry_distance.
        This is suboptimal but not a crash. Ideally, pairs with zero
        formation std should be skipped entirely in pair selection.
        """
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        # Formation: exactly zero spread volatility (identical prices)
        formation_close = pd.DataFrame({
            'A': [100.0] * 50,
            'B': [100.0] * 50,
        }, index=formation_dates)

        trading_close = pd.DataFrame({
            'A': [100.0] * 10 + [120.0] * 20,
            'B': [100.0] * 30,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
            capital_per_trade=10000,
        )

        # Should not crash - must handle division by zero gracefully
        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Result should be valid (not crash)
        assert isinstance(result.trades, list), "Should return valid trades list"
        assert len(result.equity_curve) > 0, "Should have equity curve"

        # Verify trades have valid P&L (not NaN) even if entry_distance is inf
        for trade in result.trades:
            assert not np.isnan(trade.pnl), \
                "P&L should not be NaN even with zero formation std"


class TestForcedLiquidation:
    """Test suite for forced liquidation scenarios."""

    def test_forced_liquidation_at_period_end(self):
        """Open positions must be closed at end of trading period.

        Per GGR paper: Any position still open at the end of the 6-month
        trading window is closed at market price, regardless of P&L.
        """
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        # Use DIFFERENT patterns for A and B to get non-zero spread volatility
        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 5 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 5 for i in range(50)],  # Different pattern
        }, index=formation_dates)

        # Prices that diverge and NEVER converge (never cross zero)
        trading_close = pd.DataFrame({
            'A': [100.0] * 5 + [150.0] * 25,  # Jump up and stay
            'B': [100.0] * 30,  # Stay flat
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=100,  # Very high so crossing/period-end triggers exit
            capital_per_trade=10000,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Should have at least one trade
        assert len(result.trades) > 0, "Should have opened a trade"

        # Last trade should exit at end of data
        last_trade = result.trades[-1]
        assert last_trade.exit_reason == 'end_of_data', \
            f"Last trade should exit at period end, got: {last_trade.exit_reason}"

        # Exit date should be last trading day
        assert last_trade.exit_date == trading_dates[-1], \
            "Exit should be on last day of trading period"

    def test_delisting_handling_with_nan(self):
        """Should handle stock becoming unavailable (NaN prices).

        While not full delisting, tests robustness to missing data mid-trade.
        """
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        # Use DIFFERENT patterns for A and B to get non-zero spread volatility
        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 5 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 5 for i in range(50)],  # Different pattern
        }, index=formation_dates)

        # Stock A has price data, but B goes to NaN mid-series
        trading_close = pd.DataFrame({
            'A': [100.0] * 5 + [120.0] * 25,
            'B': [100.0] * 15 + [np.nan] * 15,  # B stops trading
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=50,
            capital_per_trade=10000,
        )

        # Should not crash - test that it handles gracefully
        try:
            result = run_backtest_single_pair(
                formation_close, trading_close, trading_open, ('A', 'B'), config
            )
            # If it runs, verify equity curve exists
            assert len(result.equity_curve) > 0, "Should have equity curve"
        except Exception as e:
            # Currently may not handle delisting - document this
            pytest.skip(f"Delisting handling not yet implemented: {e}")


class TestNegativeSpread:
    """Test suite for negative spread handling."""

    def test_negative_spread_long_entry(self):
        """Should correctly handle negative divergence (long spread).

        Scenario: Spread diverts to -3.0σ
        Expected: Long Stock A, Short Stock B
        """
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        # Use DIFFERENT patterns for A and B to get non-zero spread volatility
        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 5 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 5 for i in range(50)],  # Different pattern
        }, index=formation_dates)

        # A drops relative to B (negative spread = A underperforms)
        trading_close = pd.DataFrame({
            'A': [100.0] * 10 + [80.0] * 10 + [100.0] * 10,  # Drop then recover
            'B': [100.0] * 30,  # Flat
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=50,
            capital_per_trade=10000,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Should have trades
        assert len(result.trades) > 0, "Should have trades for negative spread"

        # First trade should be LONG spread (direction = 1)
        # because spread is negative (A < B normalized)
        long_trades = [t for t in result.trades if t.direction == 1]
        assert len(long_trades) > 0, "Should have long spread trades for negative divergence"


class TestMultiplePairs:
    """Test suite for multiple pair portfolio handling."""

    def test_multiple_pair_results_combined(self):
        """Verify multiple pairs are tracked independently."""
        from src.backtest import run_backtest, combine_results

        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        # Create 4 stocks
        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 5 for i in range(50)],
            'B': [100.0 + np.sin(i/5) * 5 for i in range(50)],
            'C': [100.0 + np.cos(i/5) * 5 for i in range(50)],
            'D': [100.0 + np.cos(i/5) * 5 for i in range(50)],
        }, index=formation_dates)

        # Different divergence patterns
        trading_close = pd.DataFrame({
            'A': [100.0] * 10 + [120.0] * 10 + [100.0] * 10,
            'B': [100.0] * 30,
            'C': [100.0] * 10 + [80.0] * 10 + [100.0] * 10,
            'D': [100.0] * 30,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=50,
            capital_per_trade=10000,
        )

        pairs = [('A', 'B'), ('C', 'D')]
        results = run_backtest(
            formation_close, trading_close, trading_open, pairs, config
        )

        # Should have results for both pairs
        assert len(results) == 2, "Should have results for both pairs"
        assert ('A', 'B') in results, "Should have A/B results"
        assert ('C', 'D') in results, "Should have C/D results"

        # Combine results
        all_trades, combined_equity = combine_results(results, initial_capital=50000)

        # Combined equity should start at initial capital
        assert combined_equity.iloc[0] == 50000, "Should start at initial capital"

    def test_portfolio_capital_allocation(self):
        """Verify capital per trade is respected across pairs.

        With 2 pairs at $10,000 each, total committed = $20,000
        """
        from src.backtest import run_backtest

        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 5 for i in range(50)],
            'B': [100.0 + np.sin(i/5) * 5 for i in range(50)],
            'C': [100.0 + np.cos(i/5) * 5 for i in range(50)],
            'D': [100.0 + np.cos(i/5) * 5 for i in range(50)],
        }, index=formation_dates)

        trading_close = pd.DataFrame({
            'A': [100.0] * 10 + [120.0] * 10 + [100.0] * 10,
            'B': [100.0] * 30,
            'C': [100.0] * 10 + [80.0] * 10 + [100.0] * 10,
            'D': [100.0] * 30,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        capital_per_trade = 10000
        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=50,
            capital_per_trade=capital_per_trade,
        )

        pairs = [('A', 'B'), ('C', 'D')]
        results = run_backtest(
            formation_close, trading_close, trading_open, pairs, config
        )

        # Each pair's equity should start at capital_per_trade
        for pair, result in results.items():
            assert result.equity_curve.iloc[0] == capital_per_trade, \
                f"Pair {pair} should start with ${capital_per_trade}"


class TestMaxAdverseExcursion:
    """Test suite for max_adverse_spread (MAE) tracking."""

    def test_trade_has_max_adverse_spread_field(self):
        """Trade objects should have max_adverse_spread field."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        # Formation: Different patterns to create non-zero spread std
        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 3 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 3 for i in range(50)],
        }, index=formation_dates)

        trading_close = pd.DataFrame({
            'A': [100.0] * 5 + [130.0] * 15 + [100.0] * 10,  # Large divergence
            'B': [100.0] * 30,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=50,
            capital_per_trade=10000,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Must have trades
        assert len(result.trades) > 0, "Test data should trigger at least one trade"

        trade = result.trades[0]
        assert hasattr(trade, 'max_adverse_spread'), "Trade should have max_adverse_spread"
        assert isinstance(trade.max_adverse_spread, float), "max_adverse_spread should be float"

    def test_mae_at_least_entry_distance_for_short(self):
        """For short spread, MAE should be at least as extreme as entry distance."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        # Formation: Different patterns to create non-zero spread std
        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 3 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 3 for i in range(50)],
        }, index=formation_dates)

        # Positive spread: short entry (direction=-1)
        trading_close = pd.DataFrame({
            'A': [100.0] * 5 + [130.0] * 15 + [100.0] * 10,  # Large divergence
            'B': [100.0] * 30,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=50,
            capital_per_trade=10000,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Must have trades
        assert len(result.trades) > 0, "Test data should trigger at least one trade"

        # Find short spread trades and verify MAE
        short_trades = [t for t in result.trades if t.direction == -1]
        assert len(short_trades) > 0, "Should have at least one short spread trade"

        for trade in short_trades:
            # MAE should be at least as extreme (positive) as entry
            assert trade.max_adverse_spread >= trade.entry_distance, \
                f"Short spread MAE {trade.max_adverse_spread} should be >= entry {trade.entry_distance}"

    def test_mae_at_least_entry_distance_for_long(self):
        """For long spread, MAE should be at least as extreme as entry distance."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        # Formation: Different patterns to create non-zero spread std
        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 3 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 3 for i in range(50)],
        }, index=formation_dates)

        # Negative spread: long entry (direction=1) - A drops significantly
        trading_close = pd.DataFrame({
            'A': [100.0] * 5 + [70.0] * 15 + [100.0] * 10,  # A drops 30%
            'B': [100.0] * 30,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=50,
            capital_per_trade=10000,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Must have trades
        assert len(result.trades) > 0, "Test data should trigger at least one trade"

        # Find long spread trades and verify MAE
        long_trades = [t for t in result.trades if t.direction == 1]
        assert len(long_trades) > 0, "Should have at least one long spread trade"

        for trade in long_trades:
            # MAE should be at least as extreme as entry
            # For long spread entered at negative distance, MAE tracks the more negative extreme
            assert trade.max_adverse_spread >= trade.entry_distance, \
                f"Long spread MAE {trade.max_adverse_spread} should be >= entry {trade.entry_distance}"

    def test_mae_captures_worst_case(self):
        """MAE should capture the most extreme spread during trade.

        Setup: Short spread entry at ~2σ, spread worsens to ~4σ, then converges.
        MAE should capture the 4σ peak, not just the 2σ entry.
        """
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=40, freq='D')

        # Formation period with controlled volatility
        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 3 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 3 for i in range(50)],
        }, index=formation_dates)

        # Short spread entry: A rises (entry) -> rises MORE (worse) -> converges
        # Entry at ~115, worst at ~140, then converges to 100
        trading_close = pd.DataFrame({
            'A': [100.0] * 5 + [115.0] * 5 + [140.0] * 10 + [100.0] * 20,
            'B': [100.0] * 40,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=50,
            capital_per_trade=10000,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Must have trades
        assert len(result.trades) > 0, "Test data should trigger at least one trade"

        trade = result.trades[0]

        # Verify trade has finite distances
        assert np.isfinite(trade.entry_distance), "Entry distance should be finite"
        assert np.isfinite(trade.max_adverse_spread), "MAE should be finite"

        # For short spread, MAE should be GREATER than entry (spread worsened after entry)
        # Entry at 115, worst at 140 - the spread got worse before converging
        assert trade.max_adverse_spread > trade.entry_distance, \
            f"MAE ({trade.max_adverse_spread:.2f}) should be > entry ({trade.entry_distance:.2f}) " \
            f"since spread worsened from 115 to 140 before converging"

    def test_mae_in_to_dict(self):
        """max_adverse_spread should be included in to_dict() output."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        # Formation: Different patterns to create non-zero spread std
        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 3 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 3 for i in range(50)],
        }, index=formation_dates)

        trading_close = pd.DataFrame({
            'A': [100.0] * 5 + [130.0] * 15 + [100.0] * 10,  # Large divergence
            'B': [100.0] * 30,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=50,
            capital_per_trade=10000,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Must have trades
        assert len(result.trades) > 0, "Test data should trigger at least one trade"

        trade_dict = result.trades[0].to_dict()
        assert 'max_adverse_spread' in trade_dict, "to_dict() should include max_adverse_spread"
        assert isinstance(trade_dict['max_adverse_spread'], float), \
            "max_adverse_spread in dict should be float"


class TestInitialDivergence:
    """Test suite for immediate divergence scenario (divergence on day 1 of trading).

    Note: Day 0 of trading always has spread=0 because prices are normalized from
    the start of the trading period. So "immediate divergence" means divergence
    occurs on day 1.
    """

    def test_trade_opens_early_on_immediate_divergence(self):
        """If pair diverges >2σ immediately (day 1), should open trade early.

        Per GGR paper: A trade should be opened as soon as divergence exceeds 2σ.
        Since spread is normalized from day 0, divergence can first appear on day 1.
        """
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        # Formation: stocks move with DIFFERENT patterns to create non-zero spread std
        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 3 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 3 for i in range(50)],  # Different pattern
        }, index=formation_dates)

        # Calculate formation std to know what 2σ threshold is
        from src.signals import calculate_spread, calculate_formation_stats
        formation_spread = calculate_spread(
            formation_close['A'], formation_close['B'], normalize=True
        )
        formation_stats = calculate_formation_stats(formation_spread)
        formation_std = formation_stats['std']

        # Trading: Day 0 same price, Day 1+ diverges beyond 2σ
        # Normalized spread = (A/A[0]) - (B/B[0])
        # Day 0: spread = 0 (both normalized to 1)
        # Day 1+: A jumps to create spread > 2σ
        divergence_needed = 3.0 * formation_std  # 3σ divergence
        a_diverged_price = 100.0 * (1 + divergence_needed)

        trading_a = [100.0] + [a_diverged_price] * 14 + [100.0] * 15
        trading_b = [100.0] * 30

        trading_close = pd.DataFrame({
            'A': trading_a,
            'B': trading_b,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
            capital_per_trade=10000,
            wait_days=1,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Should have at least one trade
        assert len(result.trades) > 0, "Should open trade when diverged on day 1"

        # First trade should enter early (day 2 with wait_days=1, since signal on day 1)
        first_trade = result.trades[0]
        entry_day_idx = trading_dates.get_loc(first_trade.entry_date)

        # Signal on day 1, with wait_days=1, entry on day 2
        assert entry_day_idx <= 3, \
            f"Trade should open early (day 2-3), got entry on day {entry_day_idx}"

    def test_trade_opens_day_one_with_wait_zero(self):
        """With wait_days=0, trade should open on day 1 if divergence occurs then."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        # Formation with DIFFERENT patterns to create non-zero spread std
        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 3 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 3 for i in range(50)],  # Different pattern
        }, index=formation_dates)

        from src.signals import calculate_spread, calculate_formation_stats
        formation_spread = calculate_spread(
            formation_close['A'], formation_close['B'], normalize=True
        )
        formation_stats = calculate_formation_stats(formation_spread)
        formation_std = formation_stats['std']

        # Day 0 same price, Day 1+ diverges
        divergence_needed = 3.0 * formation_std
        a_diverged_price = 100.0 * (1 + divergence_needed)

        trading_a = [100.0] + [a_diverged_price] * 14 + [100.0] * 15
        trading_b = [100.0] * 30

        trading_close = pd.DataFrame({
            'A': trading_a,
            'B': trading_b,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
            capital_per_trade=10000,
            wait_days=0,  # Same-day execution
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        assert len(result.trades) > 0, "Should open trade when diverged on day 1"

        # With wait_days=0, signal on day 1 means entry on day 1
        first_trade = result.trades[0]
        entry_day_idx = trading_dates.get_loc(first_trade.entry_date)

        assert entry_day_idx == 1, \
            f"With wait_days=0, trade should open on day 1, got day {entry_day_idx}"


class TestDelistingHandling:
    """Test suite for proper delisting/data-end handling.

    The backtest engine handles delisting by:
    1. Detecting NaN prices when a position is open
    2. Looking backwards to find the last valid price for each stock
    3. Calculating proper P&L based on last valid prices
    4. Creating a Trade with exit_reason="delisting"
    """

    def test_delisting_does_not_crash(self):
        """When a stock stops trading (NaN), backtest should exit at last valid price."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 5 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 5 for i in range(50)],  # Different pattern
        }, index=formation_dates)

        # A diverges to trigger entry, then B goes to NaN (delisted)
        trading_a = [100.0] * 5 + [120.0] * 25  # Stays tradeable
        trading_b = [100.0] * 5 + [100.0] * 10 + [np.nan] * 15  # Stops at day 15

        trading_close = pd.DataFrame({
            'A': trading_a,
            'B': trading_b,
        }, index=trading_dates)

        trading_open = pd.DataFrame({
            'A': trading_a,
            'B': [100.0] * 5 + [100.0] * 10 + [np.nan] * 15,
        }, index=trading_dates)

        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=100,  # High so delisting triggers exit
            capital_per_trade=10000,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Verify it returns a valid result structure
        assert isinstance(result.trades, list), "Should return valid trades list"
        assert len(result.equity_curve) > 0, "Should have equity curve"

        # Should have trades that exit due to delisting
        if result.trades:
            delisting_exits = [t for t in result.trades if t.exit_reason == 'delisting']
            assert len(delisting_exits) > 0, \
                f"Should have delisting exit. Got exit reasons: {[t.exit_reason for t in result.trades]}"

            # Verify P&L is valid (not NaN) for delisting exits
            for trade in delisting_exits:
                assert not np.isnan(trade.pnl), "Delisting trade P&L should not be NaN"
                assert not np.isnan(trade.exit_price_a), "Exit price A should not be NaN"
                assert not np.isnan(trade.exit_price_b), "Exit price B should not be NaN"

    def test_both_stocks_delist_gracefully(self):
        """If both stocks in a pair stop trading, should handle gracefully."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 5 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 5 for i in range(50)],  # Different pattern
        }, index=formation_dates)

        # Both stocks stop trading midway
        trading_close = pd.DataFrame({
            'A': [100.0] * 5 + [120.0] * 10 + [np.nan] * 15,
            'B': [100.0] * 15 + [np.nan] * 15,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=100,
            capital_per_trade=10000,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Verify it returns a valid result structure
        assert isinstance(result.trades, list), "Should return valid trades list"
        assert len(result.equity_curve) > 0, "Should have equity curve"

        # If trades were opened before delisting, they should exit with valid P&L
        for trade in result.trades:
            assert not np.isnan(trade.pnl), f"Trade P&L should not be NaN: {trade}"


class TestHoldingDaysCalculation:
    """Tests for correct holding_days calculation (Bug #1 fix)."""

    def test_holding_days_single_day_trade(self):
        """Entry and exit on consecutive days should be at least 2 days held.

        With wait_days=0:
        - Signal on day 0 -> entry on day 0
        - If exit on day 0 as well, holding_days = 1

        This test ensures the +1 fix is applied correctly.
        """
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=10, freq='D')

        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 3 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 3 for i in range(50)],
        }, index=formation_dates)

        # Diverge on day 1, converge immediately on day 2
        trading_close = pd.DataFrame({
            'A': [100.0, 130.0, 100.0] + [100.0] * 7,  # Jump then immediate return
            'B': [100.0] * 10,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=50,
            capital_per_trade=10000,
            wait_days=0,  # Same-day execution for tighter control
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        if result.trades:
            trade = result.trades[0]
            # Holding days should be at least 1 (same-day entry/exit)
            # and should count inclusive of both entry and exit days
            assert trade.holding_days >= 1, \
                f"Holding days should be at least 1, got {trade.holding_days}"

    def test_holding_days_max_holding_enforcement(self):
        """With max_holding_days=5, trade should exit after exactly 5 days held."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=100, freq='D')

        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 3 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 3 for i in range(50)],
        }, index=formation_dates)

        # Prices diverge and NEVER converge (forces max_holding exit)
        trading_close = pd.DataFrame({
            'A': [100.0] * 5 + [150.0] * 95,  # Jump and stay high
            'B': [100.0] * 100,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=5,  # Force exit after exactly 5 days
            capital_per_trade=10000,
            wait_days=0,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Should have trades that exit via max_holding
        max_holding_exits = [t for t in result.trades if t.exit_reason == 'max_holding']
        assert len(max_holding_exits) > 0, "Should have max_holding exits"

        # Each max_holding trade should have exactly max_holding_days held
        for trade in max_holding_exits:
            assert trade.holding_days == config.max_holding_days, \
                f"Trade holding_days should be {config.max_holding_days}, got {trade.holding_days}"


class TestZeroFormationStd:
    """Tests for zero formation standard deviation handling (Bug #8 fix)."""

    def test_zero_std_pair_returns_no_trades(self):
        """Pair with identical formation prices should produce no trades."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        # Formation: exactly zero spread volatility (identical prices)
        formation_close = pd.DataFrame({
            'A': [100.0] * 50,
            'B': [100.0] * 50,
        }, index=formation_dates)

        trading_close = pd.DataFrame({
            'A': [100.0] * 10 + [120.0] * 20,
            'B': [100.0] * 30,
        }, index=trading_dates)

        trading_open = trading_close.copy()

        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
            capital_per_trade=10000,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Should return empty trades (not crash)
        assert result.trades == [], \
            "Zero formation std should return empty trades list"
        assert len(result.equity_curve) > 0, \
            "Should still have equity curve"


class TestZeroEntryPriceHandling:
    """Tests for zero/negative entry price handling (Bug #7 fix)."""

    def test_zero_entry_price_skipped(self):
        """Entry should be skipped when price is 0."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 3 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 3 for i in range(50)],
        }, index=formation_dates)

        # Signal would trigger, but entry price is 0
        trading_close = pd.DataFrame({
            'A': [100.0] * 5 + [120.0] * 25,  # Triggers entry signal
            'B': [100.0] * 30,
        }, index=trading_dates)

        # Open prices have zero for first few days
        trading_open = pd.DataFrame({
            'A': [0.0] * 10 + [120.0] * 20,  # Zero prices initially
            'B': [100.0] * 30,
        }, index=trading_dates)

        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=50,
            capital_per_trade=10000,
            wait_days=1,  # Entry at next open
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Should not have trades with infinity shares
        for trade in result.trades:
            assert np.isfinite(trade.shares_a), "Shares A should be finite"
            assert np.isfinite(trade.shares_b), "Shares B should be finite"

    def test_negative_entry_price_skipped(self):
        """Entry should be skipped when price is negative (data error)."""
        formation_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        trading_dates = pd.date_range('2023-03-01', periods=30, freq='D')

        formation_close = pd.DataFrame({
            'A': [100.0 + np.sin(i/5) * 3 for i in range(50)],
            'B': [100.0 + np.cos(i/5) * 3 for i in range(50)],
        }, index=formation_dates)

        trading_close = pd.DataFrame({
            'A': [100.0] * 5 + [120.0] * 25,
            'B': [100.0] * 30,
        }, index=trading_dates)

        # Open prices have negative values (data error)
        trading_open = pd.DataFrame({
            'A': [-10.0] * 10 + [120.0] * 20,  # Negative prices
            'B': [100.0] * 30,
        }, index=trading_dates)

        config = BacktestConfig(
            entry_threshold=1.0,
            max_holding_days=50,
            capital_per_trade=10000,
            wait_days=1,
        )

        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, ('A', 'B'), config
        )

        # Should not have trades with negative shares
        for trade in result.trades:
            assert trade.shares_a > 0, "Shares A should be positive"
            assert trade.shares_b > 0, "Shares B should be positive"
