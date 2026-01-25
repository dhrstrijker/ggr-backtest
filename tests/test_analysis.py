"""Tests for src/analysis.py - Performance metrics and analysis functions."""

from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    calculate_metrics,
    trades_to_dataframe,
    calculate_ggr_dollar_metrics,
    calculate_monthly_pnl_series,
    calculate_cumulative_pnl_series,
)
from src.backtest import Trade


def create_trade(
    pnl: float,
    direction: int = 1,
    holding_days: int = 10,
    pair: tuple[str, str] = ("A", "B"),
) -> Trade:
    """Helper to create a Trade object for testing."""
    entry_date = datetime(2024, 1, 1)
    return Trade(
        pair=pair,
        direction=direction,
        entry_date=entry_date,
        exit_date=entry_date + timedelta(days=holding_days),
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
        exit_reason="crossing",
        max_adverse_spread=2.5,  # Same as entry for simple test trades
    )


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_empty_trades_returns_zeros(self):
        """Empty trade list should return dictionary with zero values."""
        equity = pd.Series([10000.0], index=[datetime(2024, 1, 1)])
        metrics = calculate_metrics([], equity)

        assert metrics["total_trades"] == 0
        assert metrics["total_return"] == 0
        assert metrics["win_rate"] == 0
        assert metrics["sharpe_ratio"] == 0
        assert metrics["max_drawdown"] == 0

    def test_single_winning_trade(self):
        """Single winning trade should have 100% win rate."""
        trades = [create_trade(pnl=500.0)]
        equity = pd.Series(
            [10000.0, 10500.0],
            index=[datetime(2024, 1, 1), datetime(2024, 1, 11)],
        )

        metrics = calculate_metrics(trades, equity)

        assert metrics["total_trades"] == 1
        assert metrics["total_return"] == 500.0
        assert metrics["win_rate"] == 1.0
        assert metrics["avg_win"] == 500.0
        assert metrics["avg_loss"] == 0

    def test_single_losing_trade(self):
        """Single losing trade should have 0% win rate."""
        trades = [create_trade(pnl=-300.0)]
        equity = pd.Series(
            [10000.0, 9700.0],
            index=[datetime(2024, 1, 1), datetime(2024, 1, 11)],
        )

        metrics = calculate_metrics(trades, equity)

        assert metrics["total_trades"] == 1
        assert metrics["total_return"] == -300.0
        assert metrics["win_rate"] == 0.0
        assert metrics["avg_win"] == 0
        assert metrics["avg_loss"] == 300.0

    def test_all_winners_100_percent_win_rate(self):
        """All winning trades should yield 100% win rate."""
        trades = [
            create_trade(pnl=100.0),
            create_trade(pnl=200.0),
            create_trade(pnl=50.0),
        ]
        equity = pd.Series(
            [10000.0, 10100.0, 10300.0, 10350.0],
            index=pd.date_range("2024-01-01", periods=4),
        )

        metrics = calculate_metrics(trades, equity)

        assert metrics["total_trades"] == 3
        assert metrics["win_rate"] == 1.0
        assert metrics["total_return"] == 350.0

    def test_all_losers_zero_win_rate(self):
        """All losing trades should yield 0% win rate."""
        trades = [
            create_trade(pnl=-100.0),
            create_trade(pnl=-200.0),
            create_trade(pnl=-50.0),
        ]
        equity = pd.Series(
            [10000.0, 9900.0, 9700.0, 9650.0],
            index=pd.date_range("2024-01-01", periods=4),
        )

        metrics = calculate_metrics(trades, equity)

        assert metrics["total_trades"] == 3
        assert metrics["win_rate"] == 0.0
        assert metrics["total_return"] == -350.0

    def test_mixed_trades_correct_win_rate(self):
        """Mixed wins and losses should calculate correct win rate."""
        trades = [
            create_trade(pnl=100.0),   # Win
            create_trade(pnl=-50.0),   # Loss
            create_trade(pnl=200.0),   # Win
            create_trade(pnl=-100.0),  # Loss
            create_trade(pnl=150.0),   # Win
        ]
        equity = pd.Series(
            [10000.0, 10100.0, 10050.0, 10250.0, 10150.0, 10300.0],
            index=pd.date_range("2024-01-01", periods=6),
        )

        metrics = calculate_metrics(trades, equity)

        assert metrics["total_trades"] == 5
        assert metrics["win_rate"] == 0.6  # 3 wins out of 5

    def test_profit_factor_calculation(self):
        """Profit factor should be gross profit / gross loss."""
        trades = [
            create_trade(pnl=300.0),   # Win
            create_trade(pnl=-100.0),  # Loss
            create_trade(pnl=200.0),   # Win
        ]
        equity = pd.Series(
            [10000.0, 10300.0, 10200.0, 10400.0],
            index=pd.date_range("2024-01-01", periods=4),
        )

        metrics = calculate_metrics(trades, equity)

        # Gross profit = 300 + 200 = 500
        # Gross loss = 100
        # Profit factor = 500 / 100 = 5.0
        assert metrics["profit_factor"] == 5.0

    def test_max_drawdown_calculation(self):
        """Max drawdown should correctly identify largest peak-to-trough decline."""
        # Equity: 10000 -> 10500 -> 10200 -> 10800 -> 10100
        # Drawdown from 10500 peak: -300 (to 10200)
        # Drawdown from 10800 peak: -700 (to 10100) <- This is the max
        equity = pd.Series(
            [10000.0, 10500.0, 10200.0, 10800.0, 10100.0],
            index=pd.date_range("2024-01-01", periods=5),
        )
        trades = [create_trade(pnl=100.0)]  # Dummy trade

        metrics = calculate_metrics(trades, equity)

        assert metrics["max_drawdown"] == -700.0
        assert pytest.approx(metrics["max_drawdown_pct"], rel=0.01) == -700.0 / 10800.0

    def test_avg_holding_days(self):
        """Average holding days should be mean of all trade holding periods."""
        trades = [
            create_trade(pnl=100.0, holding_days=5),
            create_trade(pnl=200.0, holding_days=10),
            create_trade(pnl=-50.0, holding_days=15),
        ]
        equity = pd.Series(
            [10000.0, 10250.0],
            index=[datetime(2024, 1, 1), datetime(2024, 1, 31)],
        )

        metrics = calculate_metrics(trades, equity)

        assert metrics["avg_holding_days"] == 10.0  # (5 + 10 + 15) / 3

    def test_long_short_breakdown(self):
        """Should correctly count and calculate win rates by direction."""
        trades = [
            create_trade(pnl=100.0, direction=1),   # Long win
            create_trade(pnl=-50.0, direction=1),   # Long loss
            create_trade(pnl=200.0, direction=-1),  # Short win
            create_trade(pnl=-100.0, direction=-1), # Short loss
            create_trade(pnl=150.0, direction=-1),  # Short win
        ]
        equity = pd.Series(
            [10000.0, 10300.0],
            index=[datetime(2024, 1, 1), datetime(2024, 2, 1)],
        )

        metrics = calculate_metrics(trades, equity)

        assert metrics["long_trades"] == 2
        assert metrics["short_trades"] == 3
        assert metrics["long_win_rate"] == 0.5   # 1 win out of 2
        assert pytest.approx(metrics["short_win_rate"], rel=0.01) == 2/3  # 2 wins out of 3

    def test_sharpe_ratio_positive_returns(self):
        """Sharpe ratio should be positive for consistently positive returns."""
        # Create equity curve with consistent daily gains
        dates = pd.date_range("2024-01-01", periods=30)
        equity = pd.Series(
            [10000 + i * 10 for i in range(30)],  # +10 per day
            index=dates,
        )
        trades = [create_trade(pnl=290.0)]

        metrics = calculate_metrics(trades, equity)

        assert metrics["sharpe_ratio"] > 0

    def test_sharpe_ratio_zero_volatility(self):
        """Sharpe ratio should be 0 when there's no volatility."""
        # Flat equity curve
        dates = pd.date_range("2024-01-01", periods=10)
        equity = pd.Series([10000.0] * 10, index=dates)
        trades = [create_trade(pnl=0.0)]

        metrics = calculate_metrics(trades, equity)

        assert metrics["sharpe_ratio"] == 0

    def test_total_return_pct_calculation(self):
        """Total return percentage should be (final - initial) / initial."""
        equity = pd.Series(
            [10000.0, 12000.0],
            index=[datetime(2024, 1, 1), datetime(2024, 2, 1)],
        )
        trades = [create_trade(pnl=2000.0)]

        metrics = calculate_metrics(trades, equity)

        assert metrics["total_return_pct"] == 0.2  # (12000 - 10000) / 10000

    def test_profit_factor_all_winners_no_losses(self):
        """Profit factor with no losses should equal infinity (Bug #6 fix)."""
        trades = [
            create_trade(pnl=300.0),  # Win
            create_trade(pnl=200.0),  # Win
            create_trade(pnl=100.0),  # Win
        ]
        equity = pd.Series(
            [10000.0, 10300.0, 10500.0, 10600.0],
            index=pd.date_range("2024-01-01", periods=4),
        )

        metrics = calculate_metrics(trades, equity)

        # With no losses, profit factor should be infinity (dimensionless ratio)
        assert metrics["profit_factor"] == float("inf"), \
            f"Profit factor with all winners should be infinity, got {metrics['profit_factor']}"

    def test_sharpe_ratio_handles_zero_volatility(self):
        """Sharpe ratio with zero return volatility should return 0.

        When returns have zero standard deviation (flat equity), Sharpe ratio
        is undefined (division by zero). The implementation returns 0 in this case.
        """
        # Flat equity curve (zero volatility in returns)
        dates = pd.date_range("2024-01-01", periods=10)
        equity = pd.Series([10000.0] * 10, index=dates)
        trades = []  # No trades

        metrics = calculate_metrics(trades, equity)

        # Implementation returns 0 when std == 0 (see analysis.py line 79)
        assert metrics["sharpe_ratio"] == 0, \
            f"Sharpe ratio with zero volatility should be 0, got {metrics['sharpe_ratio']}"

    def test_avg_loss_with_no_losses(self):
        """Average loss with no losses should be 0."""
        trades = [
            create_trade(pnl=100.0),  # Win
            create_trade(pnl=200.0),  # Win
        ]
        equity = pd.Series(
            [10000.0, 10100.0, 10300.0],
            index=pd.date_range("2024-01-01", periods=3),
        )

        metrics = calculate_metrics(trades, equity)

        assert metrics["avg_loss"] == 0, \
            f"Average loss with no losses should be 0, got {metrics['avg_loss']}"

    def test_avg_win_with_no_wins(self):
        """Average win with no wins should be 0."""
        trades = [
            create_trade(pnl=-100.0),  # Loss
            create_trade(pnl=-200.0),  # Loss
        ]
        equity = pd.Series(
            [10000.0, 9900.0, 9700.0],
            index=pd.date_range("2024-01-01", periods=3),
        )

        metrics = calculate_metrics(trades, equity)

        assert metrics["avg_win"] == 0, \
            f"Average win with no wins should be 0, got {metrics['avg_win']}"


class TestTradesToDataframe:
    """Tests for trades_to_dataframe function."""

    def test_converts_trades_to_dataframe(self):
        """Should convert list of trades to DataFrame."""
        trades = [
            create_trade(pnl=100.0, direction=1),
            create_trade(pnl=-50.0, direction=-1),
        ]

        df = trades_to_dataframe(trades)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_empty_trades_returns_empty_df(self):
        """Empty trade list should return empty DataFrame."""
        df = trades_to_dataframe([])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_dataframe_has_required_columns(self):
        """DataFrame should have all expected columns from Trade.to_dict()."""
        trades = [create_trade(pnl=100.0)]

        df = trades_to_dataframe(trades)

        expected_columns = [
            "pair", "direction", "entry_date", "exit_date",
            "entry_price_a", "entry_price_b", "exit_price_a", "exit_price_b",
            "pnl", "pnl_pct", "holding_days", "entry_distance",
            "exit_distance", "exit_reason", "max_adverse_spread",
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

        # Verify at least some values are populated correctly (not just column existence)
        assert df.iloc[0]["pnl"] == 100.0, "P&L value should match trade"
        assert df.iloc[0]["entry_price_a"] == 100.0, "Entry price A should match trade"
        assert df.iloc[0]["exit_reason"] == "crossing", "Exit reason should match trade"

    def test_dataframe_values_match_trade(self):
        """DataFrame values should match original Trade object."""
        trade = create_trade(pnl=123.45, direction=1, holding_days=7)

        df = trades_to_dataframe([trade])

        assert df.iloc[0]["pnl"] == 123.45
        assert df.iloc[0]["holding_days"] == 7
        assert df.iloc[0]["direction"] == "Long"  # to_dict converts 1 to "Long"


# -----------------------------------------------------------------------------
# Tests for GGR Dollar-Based Metrics
# -----------------------------------------------------------------------------


class MockStaggeredResult:
    """Mock StaggeredResult for testing calculate_ggr_dollar_metrics."""

    def __init__(
        self,
        trades: list[Trade],
        avg_active: float = 5.0,
        total_portfolios: int = 40,
    ):
        self.all_trades = trades
        self.total_portfolios = total_portfolios
        # Create a Series that returns avg_active when .mean() is called
        self.active_portfolios_over_time = pd.Series([avg_active] * 100)


class TestCalculateMonthlyPnlSeries:
    """Tests for calculate_monthly_pnl_series function."""

    def test_empty_trades_returns_empty_series(self):
        """Empty list returns empty Series."""
        result = calculate_monthly_pnl_series([])
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_aggregates_by_exit_month(self):
        """Trades should be grouped by exit month."""
        # Create trades in different months
        trade1 = create_trade(pnl=100.0, holding_days=10)  # Jan
        trade2 = create_trade(pnl=200.0, holding_days=10)  # Jan

        # Modify exit dates to be in different months
        trade1.exit_date = datetime(2024, 1, 15)
        trade2.exit_date = datetime(2024, 1, 20)
        trade3 = create_trade(pnl=300.0, holding_days=10)
        trade3.exit_date = datetime(2024, 2, 15)

        result = calculate_monthly_pnl_series([trade1, trade2, trade3])

        assert len(result) == 2  # Two months
        # January total: 100 + 200 = 300
        jan_period = pd.Period("2024-01", freq="M")
        assert result[jan_period] == 300.0
        # February total: 300
        feb_period = pd.Period("2024-02", freq="M")
        assert result[feb_period] == 300.0

    def test_handles_negative_pnl(self):
        """Should correctly sum negative P&L."""
        trade1 = create_trade(pnl=-500.0)
        trade2 = create_trade(pnl=200.0)
        trade1.exit_date = datetime(2024, 1, 15)
        trade2.exit_date = datetime(2024, 1, 20)

        result = calculate_monthly_pnl_series([trade1, trade2])

        jan_period = pd.Period("2024-01", freq="M")
        assert result[jan_period] == -300.0  # -500 + 200


class TestCalculateGGRDollarMetrics:
    """Tests for calculate_ggr_dollar_metrics function."""

    def test_empty_trades_returns_empty_dict(self):
        """Empty trades should return empty dict."""
        result = MockStaggeredResult(trades=[])
        metrics = calculate_ggr_dollar_metrics(result, capital_per_trade=10000, n_pairs=10)
        assert metrics == {}

    def test_total_pnl_is_sum_of_trade_pnl(self):
        """Total P&L should equal sum of all trade P&L."""
        trades = [
            create_trade(pnl=100.0),
            create_trade(pnl=-50.0),
            create_trade(pnl=200.0),
        ]
        result = MockStaggeredResult(trades=trades)

        metrics = calculate_ggr_dollar_metrics(result, capital_per_trade=10000, n_pairs=10)

        assert metrics["total_pnl"] == 250.0  # 100 + (-50) + 200

    def test_fully_invested_capital_based_on_traded_pairs(self):
        """Fully invested capital should count unique pairs that traded."""
        trades = [
            create_trade(pnl=100.0, pair=("A", "B")),
            create_trade(pnl=100.0, pair=("A", "B")),  # Same pair
            create_trade(pnl=100.0, pair=("C", "D")),  # Different pair
        ]
        result = MockStaggeredResult(trades=trades)

        metrics = calculate_ggr_dollar_metrics(result, capital_per_trade=10000, n_pairs=10)

        # 2 unique pairs: ("A", "B") and ("C", "D")
        assert metrics["pairs_traded"] == 2
        assert metrics["capital_fully_invested"] == 20000.0  # 2 * 10000

    def test_committed_capital_uses_all_pairs_and_avg_active(self):
        """Committed capital should use n_pairs × avg_active × capital_per_trade."""
        trades = [create_trade(pnl=100.0)]
        result = MockStaggeredResult(trades=trades, avg_active=4.5)

        metrics = calculate_ggr_dollar_metrics(result, capital_per_trade=10000, n_pairs=10)

        # 10 pairs × 4.5 avg active × $10k = $450k
        assert metrics["capital_committed"] == 450000.0

    def test_return_fully_invested_calculation(self):
        """Fully invested return = P&L / capital_fully_invested."""
        trades = [create_trade(pnl=1000.0, pair=("A", "B"))]
        result = MockStaggeredResult(trades=trades)

        metrics = calculate_ggr_dollar_metrics(result, capital_per_trade=10000, n_pairs=10)

        # P&L: $1000, Capital: 1 pair × $10k = $10k
        # Return: 1000 / 10000 = 0.10 (10%)
        assert metrics["return_fully_invested"] == pytest.approx(0.10)

    def test_return_committed_calculation(self):
        """Committed return = P&L / capital_committed."""
        trades = [create_trade(pnl=4500.0, pair=("A", "B"))]
        result = MockStaggeredResult(trades=trades, avg_active=4.5)

        metrics = calculate_ggr_dollar_metrics(result, capital_per_trade=10000, n_pairs=10)

        # P&L: $4500, Committed Capital: 10 × 4.5 × $10k = $450k
        # Return: 4500 / 450000 = 0.01 (1%)
        assert metrics["return_committed"] == pytest.approx(0.01)

    def test_negative_pnl_gives_negative_return(self):
        """Negative P&L should produce negative return."""
        trades = [create_trade(pnl=-5000.0)]
        result = MockStaggeredResult(trades=trades, avg_active=1.0)

        metrics = calculate_ggr_dollar_metrics(result, capital_per_trade=10000, n_pairs=10)

        assert metrics["total_pnl"] == -5000.0
        assert metrics["return_committed"] < 0
        assert metrics["ann_return_committed"] < 0

    def test_return_sign_matches_pnl_sign(self):
        """Return sign should always match P&L sign."""
        # Positive P&L
        trades_pos = [create_trade(pnl=1000.0)]
        result_pos = MockStaggeredResult(trades=trades_pos, avg_active=1.0)
        metrics_pos = calculate_ggr_dollar_metrics(result_pos, capital_per_trade=10000, n_pairs=10)

        assert metrics_pos["total_pnl"] > 0
        assert metrics_pos["return_committed"] > 0

        # Negative P&L
        trades_neg = [create_trade(pnl=-1000.0)]
        result_neg = MockStaggeredResult(trades=trades_neg, avg_active=1.0)
        metrics_neg = calculate_ggr_dollar_metrics(result_neg, capital_per_trade=10000, n_pairs=10)

        assert metrics_neg["total_pnl"] < 0
        assert metrics_neg["return_committed"] < 0

    def test_sharpe_ratio_calculated(self):
        """Sharpe ratio should be calculated from monthly returns."""
        # Create trades across multiple months with large enough P&L
        # to exceed the risk-free rate (2% / 12 = 0.17% monthly)
        trades = []
        # With avg_active=1, n_pairs=10, capital=10000 → committed = $100k
        # Need monthly P&L > $100k * 0.17% = $170 to beat risk-free rate
        pnl_values = [5000.0, 6000.0, 4000.0, 7000.0, 5500.0, 6500.0]  # Large varying P&L
        for month, pnl in enumerate(pnl_values, start=1):
            trade = create_trade(pnl=pnl)
            trade.exit_date = datetime(2024, month, 15)
            trades.append(trade)

        result = MockStaggeredResult(trades=trades, avg_active=1.0)  # Lower avg_active

        metrics = calculate_ggr_dollar_metrics(result, capital_per_trade=10000, n_pairs=10)

        # Should have a Sharpe ratio (not testing exact value, just that it exists)
        assert "sharpe_ratio" in metrics
        # With returns exceeding risk-free rate, Sharpe should be positive
        assert metrics["sharpe_ratio"] > 0

    def test_max_drawdown_calculated(self):
        """Max drawdown should be calculated from cumulative P&L."""
        # Create trades: win, win, big loss, small win
        # P&L: 100, 200, -500, 50 -> Cumulative: 100, 300, -200, -150
        # Peak at 300, drawdown to -200 = -500
        trades = []
        pnl_values = [100.0, 200.0, -500.0, 50.0]
        for day, pnl in enumerate(pnl_values, start=1):
            trade = create_trade(pnl=pnl)
            trade.exit_date = datetime(2024, 1, day * 5)
            trades.append(trade)

        result = MockStaggeredResult(trades=trades, avg_active=1.0)
        metrics = calculate_ggr_dollar_metrics(result, capital_per_trade=10000, n_pairs=10)

        assert "max_drawdown" in metrics
        assert "max_drawdown_pct" in metrics
        # Max drawdown should be -500 (from peak of 300 to trough of -200)
        assert metrics["max_drawdown"] == -500.0
        # As pct of committed capital ($100k): -500 / 100000 = -0.5%
        assert metrics["max_drawdown_pct"] == pytest.approx(-0.005)


class TestCalculateCumulativePnlSeries:
    """Tests for calculate_cumulative_pnl_series function."""

    def test_empty_trades_returns_empty_series(self):
        """Empty list returns empty Series."""
        result = calculate_cumulative_pnl_series([])
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_cumulative_sum_correct(self):
        """Cumulative sum should add up correctly."""
        trade1 = create_trade(pnl=100.0)
        trade2 = create_trade(pnl=200.0)
        trade3 = create_trade(pnl=-50.0)

        # Set different exit dates
        trade1.exit_date = datetime(2024, 1, 10)
        trade2.exit_date = datetime(2024, 1, 15)
        trade3.exit_date = datetime(2024, 1, 20)

        result = calculate_cumulative_pnl_series([trade1, trade2, trade3])

        assert len(result) == 3
        # Cumulative: 100, 300, 250
        assert result.iloc[0] == 100.0
        assert result.iloc[1] == 300.0
        assert result.iloc[2] == 250.0

    def test_orders_by_date(self):
        """Series should be ordered by exit date."""
        trade1 = create_trade(pnl=100.0)
        trade2 = create_trade(pnl=200.0)

        # Set dates out of order
        trade1.exit_date = datetime(2024, 2, 1)
        trade2.exit_date = datetime(2024, 1, 1)

        result = calculate_cumulative_pnl_series([trade1, trade2])

        # Should be sorted by date, so trade2 (Jan) comes first
        assert result.index[0] < result.index[1]
        # First cumulative is 200 (trade2), second is 300 (200+100)
        assert result.iloc[0] == 200.0
        assert result.iloc[1] == 300.0

    def test_multiple_trades_same_day(self):
        """Trades on same day should be summed."""
        trade1 = create_trade(pnl=100.0)
        trade2 = create_trade(pnl=150.0)
        trade3 = create_trade(pnl=50.0)

        # All exit on same day
        same_date = datetime(2024, 1, 15)
        trade1.exit_date = same_date
        trade2.exit_date = same_date
        trade3.exit_date = same_date

        result = calculate_cumulative_pnl_series([trade1, trade2, trade3])

        # Should have only one entry (all same day)
        assert len(result) == 1
        # Total P&L: 100 + 150 + 50 = 300
        assert result.iloc[0] == 300.0


# -----------------------------------------------------------------------------
# Bug Fix Tests - Win/Loss Classification (Bug #4)
# -----------------------------------------------------------------------------


class TestWinLossClassification:
    """Tests for correct win/loss/breakeven classification (Bug #4 fix)."""

    def test_breakeven_not_counted_as_loss(self):
        """Trade with P&L = 0 should not be in losses list."""
        trades = [
            create_trade(pnl=100.0),  # Win
            create_trade(pnl=-50.0),  # Loss
            create_trade(pnl=0.0),    # Break-even
            create_trade(pnl=200.0),  # Win
            create_trade(pnl=-100.0), # Loss
        ]
        equity = pd.Series(
            [10000.0, 10150.0],
            index=[datetime(2024, 1, 1), datetime(2024, 2, 1)],
        )

        metrics = calculate_metrics(trades, equity)

        # Win rate should be 2 / 4 = 50% (2 wins, 2 losses, 1 breakeven excluded)
        assert metrics["win_rate"] == pytest.approx(0.5), \
            f"Win rate should be 50% with 2 wins and 2 losses, got {metrics['win_rate']}"

    def test_win_rate_excludes_breakeven(self):
        """Win rate should be wins / (wins + losses), excluding breakeven."""
        trades = [
            create_trade(pnl=100.0),  # Win
            create_trade(pnl=-50.0),  # Loss
            create_trade(pnl=0.0),    # Break-even
        ]
        equity = pd.Series(
            [10000.0, 10050.0],
            index=[datetime(2024, 1, 1), datetime(2024, 2, 1)],
        )

        metrics = calculate_metrics(trades, equity)

        # 1 win, 1 loss, 1 breakeven -> win rate = 1/2 = 50%
        assert metrics["win_rate"] == pytest.approx(0.5), \
            f"Win rate should be 50% (1 win / 2 decided), got {metrics['win_rate']}"

    def test_all_breakeven_trades(self):
        """All breakeven trades should result in 0 win rate."""
        trades = [
            create_trade(pnl=0.0),
            create_trade(pnl=0.0),
            create_trade(pnl=0.0),
        ]
        equity = pd.Series(
            [10000.0, 10000.0],
            index=[datetime(2024, 1, 1), datetime(2024, 2, 1)],
        )

        metrics = calculate_metrics(trades, equity)

        # No decided trades (all breakeven), win rate should be 0
        assert metrics["win_rate"] == 0, \
            f"Win rate should be 0 with all breakeven trades, got {metrics['win_rate']}"

    def test_avg_loss_excludes_breakeven(self):
        """Average loss should only consider actual losses."""
        trades = [
            create_trade(pnl=100.0),   # Win
            create_trade(pnl=-100.0),  # Loss
            create_trade(pnl=0.0),     # Break-even
            create_trade(pnl=-200.0),  # Loss
        ]
        equity = pd.Series(
            [10000.0, 9800.0],
            index=[datetime(2024, 1, 1), datetime(2024, 2, 1)],
        )

        metrics = calculate_metrics(trades, equity)

        # avg_loss = (100 + 200) / 2 = 150 (positive magnitude)
        assert metrics["avg_loss"] == pytest.approx(150.0), \
            f"avg_loss should be 150 (avg of |100| and |200|), got {metrics['avg_loss']}"


# -----------------------------------------------------------------------------
# Bug Fix Tests - Profit Factor Edge Cases (Bug #6)
# -----------------------------------------------------------------------------


class TestProfitFactorEdgeCases:
    """Tests for profit factor edge cases (Bug #6 fix)."""

    def test_profit_factor_all_winners_returns_infinity(self):
        """All winning trades should return profit_factor = inf."""
        trades = [
            create_trade(pnl=100.0),
            create_trade(pnl=200.0),
            create_trade(pnl=300.0),
        ]
        equity = pd.Series(
            [10000.0, 10600.0],
            index=[datetime(2024, 1, 1), datetime(2024, 2, 1)],
        )

        metrics = calculate_metrics(trades, equity)

        # With no losses, profit factor should be infinity
        assert metrics["profit_factor"] == float("inf"), \
            f"Profit factor with all winners should be inf, got {metrics['profit_factor']}"

    def test_profit_factor_all_losers_returns_zero(self):
        """All losing trades should return profit_factor = 0."""
        trades = [
            create_trade(pnl=-100.0),
            create_trade(pnl=-200.0),
            create_trade(pnl=-300.0),
        ]
        equity = pd.Series(
            [10000.0, 9400.0],
            index=[datetime(2024, 1, 1), datetime(2024, 2, 1)],
        )

        metrics = calculate_metrics(trades, equity)

        # With no wins, profit factor should be 0
        assert metrics["profit_factor"] == 0, \
            f"Profit factor with all losers should be 0, got {metrics['profit_factor']}"

    def test_profit_factor_is_dimensionless_ratio(self):
        """Profit factor should be ratio, not dollar amount."""
        trades = [
            create_trade(pnl=100.0),  # Win
            create_trade(pnl=-50.0),  # Loss
        ]
        equity = pd.Series(
            [10000.0, 10050.0],
            index=[datetime(2024, 1, 1), datetime(2024, 2, 1)],
        )

        metrics = calculate_metrics(trades, equity)

        # Profit factor = 100 / 50 = 2.0 (ratio)
        assert metrics["profit_factor"] == 2.0, \
            f"Profit factor should be 2.0 (100/50 ratio), got {metrics['profit_factor']}"


# =============================================================================
# Tests for calculate_staggered_metrics
# =============================================================================


class MockStaggeredResultForMetrics:
    """Mock StaggeredResult for testing calculate_staggered_metrics."""

    def __init__(
        self,
        monthly_returns: pd.Series,
        cumulative_returns: pd.Series,
        all_trades: list[Trade] = None,
        active_portfolios: pd.Series = None,
        total_cycles: int = 10,
    ):
        self.monthly_returns = monthly_returns
        self.cumulative_returns = cumulative_returns
        self.all_trades = all_trades or []
        self.active_portfolios_over_time = active_portfolios or pd.Series([5] * len(monthly_returns))
        self.cycles = [None] * total_cycles  # Dummy cycles for total_portfolios property

    @property
    def total_portfolios(self) -> int:
        return len(self.cycles)


class TestCalculateStaggeredMetrics:
    """Tests for calculate_staggered_metrics function."""

    def test_returns_expected_structure(self):
        """calculate_staggered_metrics should return dict with expected keys."""
        from src.analysis import calculate_staggered_metrics

        # Create mock staggered result with data
        monthly_returns = pd.Series(
            [0.02, 0.01, -0.01, 0.03, 0.02],
            index=pd.date_range("2021-01-31", periods=5, freq="ME"),
        )
        cumulative_returns = (1 + monthly_returns).cumprod() - 1

        mock_result = MockStaggeredResultForMetrics(
            monthly_returns=monthly_returns,
            cumulative_returns=cumulative_returns,
        )

        metrics = calculate_staggered_metrics(mock_result)

        # Should have expected keys (based on actual implementation)
        expected_keys = [
            "total_months", "annualized_return", "sharpe_ratio",
            "max_drawdown", "total_portfolios", "total_trades",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_handles_empty_result(self):
        """Should handle result with no monthly returns."""
        from src.analysis import calculate_staggered_metrics

        empty_returns = pd.Series([], dtype=float)
        empty_cumulative = pd.Series([], dtype=float)

        mock_result = MockStaggeredResultForMetrics(
            monthly_returns=empty_returns,
            cumulative_returns=empty_cumulative,
        )

        metrics = calculate_staggered_metrics(mock_result)

        # Should return metrics with zeros
        assert metrics["total_months"] == 0
        assert metrics["annualized_return"] == 0
        assert metrics["sharpe_ratio"] == 0

    def test_sharpe_ratio_calculation(self):
        """Sharpe ratio should be calculated correctly from monthly returns."""
        from src.analysis import calculate_staggered_metrics

        # Create returns with known values
        monthly_returns = pd.Series(
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            index=pd.date_range("2021-01-31", periods=12, freq="ME"),
        )
        cumulative_returns = (1 + monthly_returns).cumprod() - 1

        mock_result = MockStaggeredResultForMetrics(
            monthly_returns=monthly_returns,
            cumulative_returns=cumulative_returns,
        )

        # With 2% annual risk-free rate = 0.167% monthly
        metrics = calculate_staggered_metrics(mock_result, risk_free_rate=0.02)

        # Sharpe ratio should be positive for consistent positive returns
        assert metrics["sharpe_ratio"] > 0, \
            f"Sharpe should be positive for consistent gains, got {metrics['sharpe_ratio']}"

    def test_max_drawdown_calculation(self):
        """Max drawdown should be calculated correctly."""
        from src.analysis import calculate_staggered_metrics

        # Returns that create a drawdown
        # 0.10, -0.20, -0.10, 0.05 -> cumulative: 1.1, 0.88, 0.792, 0.832
        # Peak at 1.1, trough at 0.792, drawdown = (0.792 - 1.1) / 1.1 = -0.28
        monthly_returns = pd.Series(
            [0.10, -0.20, -0.10, 0.05],
            index=pd.date_range("2021-01-31", periods=4, freq="ME"),
        )
        cumulative_returns = (1 + monthly_returns).cumprod() - 1

        mock_result = MockStaggeredResultForMetrics(
            monthly_returns=monthly_returns,
            cumulative_returns=cumulative_returns,
        )

        metrics = calculate_staggered_metrics(mock_result)

        # Max drawdown should be negative (function returns max_drawdown, not max_drawdown_pct)
        assert metrics["max_drawdown"] < 0, \
            f"Max drawdown should be negative, got {metrics['max_drawdown']}"

    def test_annualized_return_positive_for_gains(self):
        """Annualized return should be positive for net positive monthly returns."""
        from src.analysis import calculate_staggered_metrics

        # Net positive monthly returns
        monthly_returns = pd.Series(
            [0.05, 0.03, -0.02, 0.04],  # Net positive
            index=pd.date_range("2021-01-31", periods=4, freq="ME"),
        )
        cumulative_returns = (1 + monthly_returns).cumprod() - 1

        mock_result = MockStaggeredResultForMetrics(
            monthly_returns=monthly_returns,
            cumulative_returns=cumulative_returns,
        )

        metrics = calculate_staggered_metrics(mock_result)

        # Annualized return should be positive for net gains
        assert metrics["annualized_return"] > 0, \
            f"Annualized return should be positive for net gains, got {metrics['annualized_return']}"

        # Avg monthly return should also be positive
        assert metrics["avg_monthly_return"] > 0, \
            f"Avg monthly return should be positive, got {metrics['avg_monthly_return']}"
