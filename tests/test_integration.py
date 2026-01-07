"""Integration tests for end-to-end backtest verification.

These tests verify the complete pipeline from price data to final metrics,
using pre-calculated fixtures with known expected outcomes.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, Trade, run_backtest_single_pair, combine_results
from src.analysis import calculate_metrics
from src.signals import calculate_spread, calculate_formation_stats, calculate_distance

from tests.fixtures.known_outcomes import (
    SIMPLE_CONVERGENCE_FIXTURE,
    KNOWN_METRICS_FIXTURE,
    NO_TRADE_FIXTURE,
)


class TestEndToEndBacktest:
    """End-to-end backtest integration tests."""

    def test_simple_convergence_generates_trades(self):
        """Simple convergence scenario should generate expected number of trades."""
        fixture = SIMPLE_CONVERGENCE_FIXTURE

        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
            capital_per_trade=10000.0,
        )

        result = run_backtest_single_pair(
            formation_close=fixture["formation_close"],
            trading_close=fixture["trading_close"],
            trading_open=fixture["trading_open"],
            pair=fixture["pair"],
            config=config,
        )

        # Should generate trades within expected range
        assert fixture["expected_min_trades"] <= len(result.trades) <= fixture["expected_max_trades"], \
            f"Expected {fixture['expected_min_trades']}-{fixture['expected_max_trades']} trades, got {len(result.trades)}"

    def test_no_trade_scenario(self):
        """Scenario with no divergence should generate no trades."""
        fixture = NO_TRADE_FIXTURE

        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
        )

        result = run_backtest_single_pair(
            formation_close=fixture["formation_close"],
            trading_close=fixture["trading_close"],
            trading_open=fixture["trading_open"],
            pair=fixture["pair"],
            config=config,
        )

        assert len(result.trades) == fixture["expected_trades"], \
            f"Expected {fixture['expected_trades']} trades, got {len(result.trades)}"

    def test_formation_std_stays_constant(self):
        """Formation period σ should not change during trading."""
        fixture = SIMPLE_CONVERGENCE_FIXTURE

        # Calculate formation stats
        formation_spread = calculate_spread(
            fixture["formation_close"]["A"],
            fixture["formation_close"]["B"],
            normalize=True,
        )
        formation_stats = calculate_formation_stats(formation_spread)
        formation_std = formation_stats["std"]

        # Calculate trading spread
        trading_spread = calculate_spread(
            fixture["trading_close"]["A"],
            fixture["trading_close"]["B"],
            normalize=True,
        )

        # Calculate distance using formation std
        distance = calculate_distance(trading_spread, formation_std)

        # Verify distance uses CONSTANT formation std (not rolling)
        # The std used should be a single value, not recalculated per row
        assert isinstance(formation_std, float), "Formation std should be a scalar"
        assert formation_std > 0, "Formation std should be positive"

        # Distance should be spread / formation_std for all points
        expected_distance = trading_spread / formation_std
        pd.testing.assert_series_equal(distance, expected_distance, check_names=False)

    def test_no_lookahead_bias(self):
        """Signals should be based on close, execution on next-day open."""
        fixture = SIMPLE_CONVERGENCE_FIXTURE

        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
            wait_days=1,  # Next-day execution
        )

        result = run_backtest_single_pair(
            formation_close=fixture["formation_close"],
            trading_close=fixture["trading_close"],
            trading_open=fixture["trading_open"],
            pair=fixture["pair"],
            config=config,
        )

        for trade in result.trades:
            # Entry date should be in trading period
            assert trade.entry_date in fixture["trading_open"].index, \
                "Entry date should be a trading date"

            # Entry price should match OPEN price (not close)
            if trade.entry_date in fixture["trading_open"].index:
                expected_a = fixture["trading_open"].loc[trade.entry_date, "A"]
                expected_b = fixture["trading_open"].loc[trade.entry_date, "B"]

                assert trade.entry_price_a == expected_a, \
                    f"Entry price A should be open: {expected_a}, got {trade.entry_price_a}"
                assert trade.entry_price_b == expected_b, \
                    f"Entry price B should be open: {expected_b}, got {trade.entry_price_b}"

    def test_pipeline_produces_consistent_results(self):
        """Running the same backtest twice should produce identical results."""
        fixture = SIMPLE_CONVERGENCE_FIXTURE

        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
        )

        result1 = run_backtest_single_pair(
            formation_close=fixture["formation_close"],
            trading_close=fixture["trading_close"],
            trading_open=fixture["trading_open"],
            pair=fixture["pair"],
            config=config,
        )

        result2 = run_backtest_single_pair(
            formation_close=fixture["formation_close"],
            trading_close=fixture["trading_close"],
            trading_open=fixture["trading_open"],
            pair=fixture["pair"],
            config=config,
        )

        # Same number of trades
        assert len(result1.trades) == len(result2.trades)

        # Same P&L for each trade
        for t1, t2 in zip(result1.trades, result2.trades):
            assert t1.pnl == t2.pnl
            assert t1.entry_date == t2.entry_date
            assert t1.exit_date == t2.exit_date

    def test_equity_curve_starts_at_capital(self):
        """Equity curve should start at configured capital."""
        fixture = SIMPLE_CONVERGENCE_FIXTURE
        capital = 15000.0

        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
            capital_per_trade=capital,
        )

        result = run_backtest_single_pair(
            formation_close=fixture["formation_close"],
            trading_close=fixture["trading_close"],
            trading_open=fixture["trading_open"],
            pair=fixture["pair"],
            config=config,
        )

        assert result.equity_curve.iloc[0] == capital


class TestMetricsIntegration:
    """Integration tests for metrics calculation."""

    def test_metrics_match_manual_calculation(self):
        """Calculated metrics should match manually computed expected values."""
        fixture = KNOWN_METRICS_FIXTURE

        # Create Trade objects from fixture data
        trades = []
        base_date = datetime(2024, 1, 1)
        for i, td in enumerate(fixture["trades_data"]):
            entry_date = base_date
            exit_date = entry_date + timedelta(days=td["holding_days"])
            trade = Trade(
                pair=("A", "B"),
                direction=td["direction"],
                entry_date=entry_date,
                exit_date=exit_date,
                entry_price_a=100.0,
                entry_price_b=100.0,
                exit_price_a=100.0,
                exit_price_b=100.0,
                shares_a=50.0,
                shares_b=50.0,
                pnl=td["pnl"],
                pnl_pct=td["pnl_pct"],
                holding_days=td["holding_days"],
                entry_distance=2.5,
                exit_distance=0.1,
                exit_reason="crossing",
            )
            trades.append(trade)

        # Create simple equity curve
        equity = pd.Series(
            [10000.0, 10000.0 + fixture["expected_total_pnl"]],
            index=[datetime(2024, 1, 1), datetime(2024, 2, 1)],
        )

        metrics = calculate_metrics(trades, equity)

        # Verify against expected values
        assert metrics["total_trades"] == fixture["expected_total_trades"]
        assert metrics["total_return"] == fixture["expected_total_pnl"]
        assert pytest.approx(metrics["win_rate"], rel=0.01) == fixture["expected_win_rate"]
        assert pytest.approx(metrics["avg_win"], rel=0.01) == fixture["expected_avg_win"]
        assert pytest.approx(metrics["avg_loss"], rel=0.01) == fixture["expected_avg_loss"]
        assert pytest.approx(metrics["profit_factor"], rel=0.01) == fixture["expected_profit_factor"]
        assert pytest.approx(metrics["avg_holding_days"], rel=0.01) == fixture["expected_avg_holding_days"]
        assert metrics["long_trades"] == fixture["expected_long_trades"]
        assert metrics["short_trades"] == fixture["expected_short_trades"]
        assert pytest.approx(metrics["long_win_rate"], rel=0.01) == fixture["expected_long_win_rate"]
        assert pytest.approx(metrics["short_win_rate"], rel=0.01) == fixture["expected_short_win_rate"]

    def test_backtest_to_metrics_pipeline(self):
        """Full pipeline from backtest to metrics should work without errors."""
        fixture = SIMPLE_CONVERGENCE_FIXTURE

        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
            capital_per_trade=10000.0,
        )

        result = run_backtest_single_pair(
            formation_close=fixture["formation_close"],
            trading_close=fixture["trading_close"],
            trading_open=fixture["trading_open"],
            pair=fixture["pair"],
            config=config,
        )

        # Should be able to calculate metrics without error
        metrics = calculate_metrics(result.trades, result.equity_curve)

        # Basic sanity checks
        assert isinstance(metrics, dict)
        assert "total_trades" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert not any(pd.isna(v) for v in metrics.values() if isinstance(v, (int, float)))


class TestCombineResultsIntegration:
    """Integration tests for combining multiple pair results."""

    def test_combine_results_aggregates_trades(self):
        """Combined results should include trades from all pairs."""
        fixture = SIMPLE_CONVERGENCE_FIXTURE

        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
        )

        # Run backtest twice for same pair (simulating independent runs)
        result1 = run_backtest_single_pair(
            formation_close=fixture["formation_close"],
            trading_close=fixture["trading_close"],
            trading_open=fixture["trading_open"],
            pair=("A", "B"),
            config=config,
        )

        # Run again - using ("A", "B") relabeled as ("pair2_A", "pair2_B") conceptually
        # In practice, just run the same data under a different key
        result2 = run_backtest_single_pair(
            formation_close=fixture["formation_close"],
            trading_close=fixture["trading_close"],
            trading_open=fixture["trading_open"],
            pair=("A", "B"),  # Same data, different result object
            config=config,
        )

        # Combine under different keys to simulate multi-pair portfolio
        results = {
            ("A", "B"): result1,
            ("A2", "B2"): result2,  # Conceptually a different pair
        }

        all_trades, combined_equity = combine_results(results, initial_capital=20000.0)

        # Should have trades from both "pairs"
        assert len(all_trades) == len(result1.trades) + len(result2.trades)

        # Combined equity should reflect both
        assert combined_equity.iloc[0] == 20000.0
