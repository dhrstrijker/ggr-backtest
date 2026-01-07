"""Tests for src/analysis.py - Performance metrics and analysis functions."""

from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.analysis import calculate_metrics, trades_to_dataframe
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
            "exit_distance", "exit_reason",
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_dataframe_values_match_trade(self):
        """DataFrame values should match original Trade object."""
        trade = create_trade(pnl=123.45, direction=1, holding_days=7)

        df = trades_to_dataframe([trade])

        assert df.iloc[0]["pnl"] == 123.45
        assert df.iloc[0]["holding_days"] == 7
        assert df.iloc[0]["direction"] == "Long"  # to_dict converts 1 to "Long"
