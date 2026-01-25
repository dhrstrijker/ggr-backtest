"""Test for Issue #2: Sharpe Ratio with Negative/Zero Equity.

This test verifies that Sharpe ratio calculations handle edge cases:
- Zero equity values
- Negative equity values
- Monthly returns with 100%+ loss

BUG STATUS: FIXED - All Sharpe calculations now guard against invalid log inputs.
"""

import pandas as pd
import numpy as np
import pytest

from src.analysis import calculate_metrics, calculate_staggered_metrics
from src.backtest import Trade


class TestSharpeWithNegativeEquity:
    """Test cases for Sharpe ratio edge cases."""

    def test_sharpe_with_positive_equity(self):
        """Test normal case - Sharpe should compute correctly."""
        # Create a simple equity curve with positive values
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        equity = pd.Series(
            10000 + np.cumsum(np.random.randn(100) * 50),
            index=dates
        )
        # Ensure all positive
        equity = equity + abs(equity.min()) + 1000

        # Create a dummy trade
        trades = [
            Trade(
                pair=("A", "B"),
                direction=1,
                entry_date=dates[0],
                exit_date=dates[50],
                entry_price_a=100.0,
                entry_price_b=100.0,
                exit_price_a=105.0,
                exit_price_b=95.0,
                shares_a=50.0,
                shares_b=50.0,
                pnl=500.0,
                pnl_pct=0.05,
                holding_days=50,
                entry_distance=2.5,
                exit_distance=0.1,
                exit_reason="crossing",
                max_adverse_spread=3.0,
            )
        ]

        metrics = calculate_metrics(trades, equity)

        # Sharpe should be a valid number (not NaN, not inf)
        assert not np.isnan(metrics["sharpe_ratio"]), "Sharpe should not be NaN"
        assert not np.isinf(metrics["sharpe_ratio"]), "Sharpe should not be inf"

    def test_sharpe_with_zero_equity(self):
        """Test that Sharpe returns 0 when equity reaches zero."""
        dates = pd.date_range("2023-01-01", periods=100, freq="B")

        # Create equity curve that goes to zero
        equity = pd.Series(index=dates, dtype=float)
        equity.iloc[:50] = np.linspace(10000, 100, 50)
        equity.iloc[50:] = 0  # Zero equity

        trades = [
            Trade(
                pair=("A", "B"),
                direction=1,
                entry_date=dates[0],
                exit_date=dates[50],
                entry_price_a=100.0,
                entry_price_b=100.0,
                exit_price_a=50.0,
                exit_price_b=150.0,
                shares_a=50.0,
                shares_b=50.0,
                pnl=-10000.0,
                pnl_pct=-1.0,
                holding_days=50,
                entry_distance=2.5,
                exit_distance=0.1,
                exit_reason="max_holding",
                max_adverse_spread=5.0,
            )
        ]

        metrics = calculate_metrics(trades, equity)

        # Sharpe should be 0 (not NaN or -inf)
        assert metrics["sharpe_ratio"] == 0, (
            f"Sharpe should be 0 with zero equity, got {metrics['sharpe_ratio']}"
        )

    def test_sharpe_with_negative_equity(self):
        """Test that Sharpe returns 0 when equity goes negative."""
        dates = pd.date_range("2023-01-01", periods=100, freq="B")

        # Create equity curve that goes negative
        equity = pd.Series(index=dates, dtype=float)
        equity.iloc[:50] = np.linspace(10000, 100, 50)
        equity.iloc[50:] = np.linspace(0, -5000, 50)  # Negative equity

        trades = [
            Trade(
                pair=("A", "B"),
                direction=1,
                entry_date=dates[0],
                exit_date=dates[99],
                entry_price_a=100.0,
                entry_price_b=100.0,
                exit_price_a=20.0,
                exit_price_b=180.0,
                shares_a=50.0,
                shares_b=50.0,
                pnl=-15000.0,
                pnl_pct=-1.5,
                holding_days=99,
                entry_distance=2.5,
                exit_distance=0.1,
                exit_reason="max_holding",
                max_adverse_spread=10.0,
            )
        ]

        metrics = calculate_metrics(trades, equity)

        # Sharpe should be 0 (not NaN or -inf)
        assert metrics["sharpe_ratio"] == 0, (
            f"Sharpe should be 0 with negative equity, got {metrics['sharpe_ratio']}"
        )

    def test_sharpe_no_crash_on_edge_cases(self):
        """Test that Sharpe calculation doesn't crash on various edge cases."""
        dates = pd.date_range("2023-01-01", periods=10, freq="B")

        edge_cases = [
            # All zeros
            pd.Series([0] * 10, index=dates),
            # Single negative value
            pd.Series([10000, 10000, 10000, -100, 10000, 10000, 10000, 10000, 10000, 10000], index=dates),
            # Mix of positive and zero
            pd.Series([10000, 10000, 0, 10000, 10000, 0, 10000, 10000, 10000, 10000], index=dates),
            # Very small positive values
            pd.Series([0.0001] * 10, index=dates),
        ]

        for i, equity in enumerate(edge_cases):
            trades = []  # Empty trades is fine
            try:
                metrics = calculate_metrics(trades, equity)
                # Should not crash, should return valid metrics
                assert not np.isnan(metrics["sharpe_ratio"]), f"Case {i}: Sharpe should not be NaN"
            except Exception as e:
                pytest.fail(f"Case {i} crashed: {e}")


class TestStaggeredMetricsSharpe:
    """Test Sharpe ratio in staggered metrics with extreme monthly returns."""

    def test_sharpe_with_100_percent_loss_month(self):
        """Test that Sharpe handles months with 100%+ loss gracefully."""
        # This simulates a month where the portfolio lost more than 100%
        # (e.g., due to leverage or margin call)

        # Create a mock StaggeredResult-like object
        class MockStaggeredResult:
            def __init__(self):
                self.monthly_returns = pd.Series([0.05, 0.02, -1.5, 0.03, 0.01])  # -150% loss
                self.active_portfolios_over_time = pd.Series([6, 6, 6, 6, 6])
                self.all_trades = []
                self.total_portfolios = 5

        result = MockStaggeredResult()
        metrics = calculate_staggered_metrics(result)

        # Sharpe should be 0 (not NaN or -inf)
        assert metrics["sharpe_ratio"] == 0, (
            f"Sharpe should be 0 with 100%+ loss month, got {metrics['sharpe_ratio']}"
        )

    def test_sharpe_with_exactly_100_percent_loss(self):
        """Test boundary case of exactly -100% return."""
        class MockStaggeredResult:
            def __init__(self):
                self.monthly_returns = pd.Series([0.05, 0.02, -1.0, 0.03, 0.01])  # Exactly -100%
                self.active_portfolios_over_time = pd.Series([6, 6, 6, 6, 6])
                self.all_trades = []
                self.total_portfolios = 5

        result = MockStaggeredResult()
        metrics = calculate_staggered_metrics(result)

        # Sharpe should be 0 (log(1 + (-1)) = log(0) = -inf)
        assert metrics["sharpe_ratio"] == 0, (
            f"Sharpe should be 0 with exactly -100% loss, got {metrics['sharpe_ratio']}"
        )

    def test_sharpe_with_normal_returns(self):
        """Test that Sharpe computes normally with reasonable returns."""
        class MockStaggeredResult:
            def __init__(self):
                # Normal monthly returns (no extreme losses)
                self.monthly_returns = pd.Series([0.02, 0.01, -0.03, 0.04, 0.02, -0.01])
                self.active_portfolios_over_time = pd.Series([6, 6, 6, 6, 6, 6])
                self.all_trades = []
                self.total_portfolios = 6

        result = MockStaggeredResult()
        metrics = calculate_staggered_metrics(result)

        # Sharpe should be a valid number
        assert not np.isnan(metrics["sharpe_ratio"]), "Sharpe should not be NaN"
        assert not np.isinf(metrics["sharpe_ratio"]), "Sharpe should not be inf"
