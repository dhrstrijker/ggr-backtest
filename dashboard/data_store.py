"""Data store for dashboard - manages pre-computed backtest results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.data import fetch_or_load, fetch_benchmark, get_close_prices, get_open_prices
from src.pairs import normalize_prices, calculate_ssd_matrix, select_top_pairs, get_pair_stats
from src.signals import calculate_spread, calculate_formation_stats, calculate_distance
from src.backtest import BacktestConfig, BacktestResult, Trade, run_backtest, combine_results
from src.analysis import calculate_metrics


# Default configuration matching the notebook
DEFAULT_CONFIG = {
    "symbols": ["DHT", "FRO", "ASC", "ECO", "NAT", "TNK", "INSW", "TRMD", "TOPS", "TORO", "PSHG"],
    "start_date": "2024-01-01",
    "end_date": "2026-01-01",
    "formation_days": 252,
    "entry_threshold": 2.0,
    "max_holding_days": 126,
    "top_n_pairs": 5,
    "capital_per_trade": 10000.0,
    "commission": 0.001,
}


@dataclass
class DataStore:
    """
    Centralized data storage for the dashboard.

    Pre-computes and stores backtest results for both Wait-0-Day and Wait-1-Day
    modes, enabling instant toggling in the dashboard.
    """

    # Configuration
    config: dict = field(default_factory=lambda: DEFAULT_CONFIG.copy())

    # Price data
    formation_prices: pd.DataFrame = field(default=None)
    trading_prices: pd.DataFrame = field(default=None)
    trading_open_prices: pd.DataFrame = field(default=None)
    spy_prices: pd.DataFrame = field(default=None)

    # Pre-computed backtest results (both variants)
    results_wait_0: dict = field(default=None)
    results_wait_1: dict = field(default=None)

    # Combined results
    all_trades_wait_0: list = field(default=None)
    all_trades_wait_1: list = field(default=None)
    equity_wait_0: pd.Series = field(default=None)
    equity_wait_1: pd.Series = field(default=None)

    # Metrics
    metrics_wait_0: dict = field(default=None)
    metrics_wait_1: dict = field(default=None)

    # Pair metadata
    pairs: list = field(default=None)
    ssd_matrix: pd.DataFrame = field(default=None)
    pair_correlations: dict = field(default_factory=dict)
    formation_stats: dict = field(default_factory=dict)

    # Distance series for each pair (for Live Monitor)
    distance_series: dict = field(default_factory=dict)

    def load_or_compute(self, config: dict | None = None) -> None:
        """
        Load data and compute backtest results for both wait modes.

        Args:
            config: Optional configuration dict (uses DEFAULT_CONFIG if not provided)
        """
        if config is not None:
            self.config = config

        print("=" * 60)
        print("Loading data and computing backtest results...")
        print("=" * 60)

        # Step 1: Fetch price data
        self._load_price_data()

        # Step 2: Calculate SSD matrix and select pairs
        self._select_pairs()

        # Step 3: Calculate formation stats for each pair
        self._calculate_formation_stats()

        # Step 4: Run backtests for both wait modes
        self._run_backtests()

        # Step 5: Calculate distance series for each pair
        self._calculate_distance_series()

        print("=" * 60)
        print("Data loading complete!")
        print(f"  - Pairs: {len(self.pairs)}")
        print(f"  - Trading days: {len(self.trading_prices)}")
        print(f"  - Wait-1 trades: {len(self.all_trades_wait_1)}")
        print(f"  - Wait-0 trades: {len(self.all_trades_wait_0)}")
        print("=" * 60)

    def _load_price_data(self) -> None:
        """Load price data from Polygon.io or cache."""
        print("\n[1/5] Loading price data...")

        # Fetch universe prices
        prices_df = fetch_or_load(
            symbols=self.config["symbols"],
            start_date=self.config["start_date"],
            end_date=self.config["end_date"],
        )

        close_prices = get_close_prices(prices_df)
        open_prices = get_open_prices(prices_df)

        # Split into formation and trading periods
        formation_days = self.config["formation_days"]
        self.formation_prices = close_prices.iloc[:formation_days]
        self.trading_prices = close_prices.iloc[formation_days:]
        self.trading_open_prices = open_prices.iloc[formation_days:]

        print(f"  Formation period: {self.formation_prices.index[0].date()} to {self.formation_prices.index[-1].date()}")
        print(f"  Trading period: {self.trading_prices.index[0].date()} to {self.trading_prices.index[-1].date()}")

        # Fetch SPY benchmark
        self.spy_prices = fetch_benchmark(
            start_date=self.config["start_date"],
            end_date=self.config["end_date"],
        )
        print(f"  SPY data loaded: {len(self.spy_prices)} days")

    def _select_pairs(self) -> None:
        """Calculate SSD matrix and select top pairs."""
        print("\n[2/5] Selecting pairs...")

        # Normalize formation prices
        normalized = normalize_prices(self.formation_prices)

        # Calculate SSD matrix
        self.ssd_matrix = calculate_ssd_matrix(normalized)

        # Select top N pairs
        self.pairs = select_top_pairs(
            self.ssd_matrix,
            n=self.config["top_n_pairs"]
        )

        # Calculate correlations
        for pair in self.pairs:
            stats = get_pair_stats(normalized, pair)
            self.pair_correlations[pair] = stats["correlation"]

        print(f"  Selected {len(self.pairs)} pairs:")
        for i, pair in enumerate(self.pairs, 1):
            ssd = self.ssd_matrix.loc[pair[0], pair[1]]
            corr = self.pair_correlations[pair]
            print(f"    {i}. {pair[0]}/{pair[1]} (SSD: {ssd:.4f}, Corr: {corr:.4f})")

    def _calculate_formation_stats(self) -> None:
        """Calculate formation period statistics for each pair."""
        print("\n[3/5] Calculating formation statistics...")

        for pair in self.pairs:
            spread = calculate_spread(
                self.formation_prices[pair[0]],
                self.formation_prices[pair[1]],
                normalize=True
            )
            self.formation_stats[pair] = calculate_formation_stats(spread)

        print(f"  Calculated stats for {len(self.formation_stats)} pairs")

    def _run_backtests(self) -> None:
        """Run backtests for both wait modes."""
        print("\n[4/5] Running backtests...")

        initial_capital = self.config["capital_per_trade"] * self.config["top_n_pairs"]

        # Wait-1-Day backtest (default)
        config_wait_1 = BacktestConfig(
            entry_threshold=self.config["entry_threshold"],
            max_holding_days=self.config["max_holding_days"],
            capital_per_trade=self.config["capital_per_trade"],
            commission=self.config["commission"],
            wait_days=1,
        )

        self.results_wait_1 = run_backtest(
            formation_close=self.formation_prices,
            trading_close=self.trading_prices,
            trading_open=self.trading_open_prices,
            pairs=self.pairs,
            config=config_wait_1,
        )

        self.all_trades_wait_1, self.equity_wait_1 = combine_results(
            self.results_wait_1,
            initial_capital=initial_capital,
        )

        self.metrics_wait_1 = calculate_metrics(
            self.all_trades_wait_1,
            self.equity_wait_1,
        )

        print(f"  Wait-1-Day: {len(self.all_trades_wait_1)} trades")

        # Wait-0-Day backtest
        config_wait_0 = BacktestConfig(
            entry_threshold=self.config["entry_threshold"],
            max_holding_days=self.config["max_holding_days"],
            capital_per_trade=self.config["capital_per_trade"],
            commission=self.config["commission"],
            wait_days=0,
        )

        self.results_wait_0 = run_backtest(
            formation_close=self.formation_prices,
            trading_close=self.trading_prices,
            trading_open=self.trading_open_prices,
            pairs=self.pairs,
            config=config_wait_0,
        )

        self.all_trades_wait_0, self.equity_wait_0 = combine_results(
            self.results_wait_0,
            initial_capital=initial_capital,
        )

        self.metrics_wait_0 = calculate_metrics(
            self.all_trades_wait_0,
            self.equity_wait_0,
        )

        print(f"  Wait-0-Day: {len(self.all_trades_wait_0)} trades")

    def _calculate_distance_series(self) -> None:
        """Calculate distance series for each pair during trading period."""
        print("\n[5/5] Calculating distance series...")

        for pair in self.pairs:
            # Calculate trading spread (normalized from start of trading period)
            spread = calculate_spread(
                self.trading_prices[pair[0]],
                self.trading_prices[pair[1]],
                normalize=True
            )

            # Calculate distance using formation-period std
            formation_std = self.formation_stats[pair]["std"]
            self.distance_series[pair] = calculate_distance(spread, formation_std)

        print(f"  Calculated distance series for {len(self.distance_series)} pairs")

    # -------------------------------------------------------------------------
    # Helper methods for dashboard
    # -------------------------------------------------------------------------

    def get_results(self, wait_mode: int = 1) -> dict:
        """Get backtest results for specified wait mode."""
        return self.results_wait_1 if wait_mode == 1 else self.results_wait_0

    def get_equity_curve(self, wait_mode: int = 1) -> pd.Series:
        """Get equity curve for specified wait mode."""
        return self.equity_wait_1 if wait_mode == 1 else self.equity_wait_0

    def get_all_trades(self, wait_mode: int = 1) -> list[Trade]:
        """Get all trades for specified wait mode."""
        return self.all_trades_wait_1 if wait_mode == 1 else self.all_trades_wait_0

    def get_metrics(self, wait_mode: int = 1) -> dict:
        """Get metrics for specified wait mode."""
        return self.metrics_wait_1 if wait_mode == 1 else self.metrics_wait_0

    def get_position_at_date(self, date: datetime, wait_mode: int = 1) -> list[dict]:
        """
        Get all open positions at a specific date.

        Args:
            date: The date to check
            wait_mode: 0 or 1 for wait mode selection

        Returns:
            List of position dictionaries with trade details
        """
        results = self.get_results(wait_mode)
        positions = []

        for pair, result in results.items():
            for trade in result.trades:
                # Position is open if we're between entry and exit
                if trade.entry_date <= date < trade.exit_date:
                    positions.append({
                        "pair": pair,
                        "direction": trade.direction,
                        "entry_date": trade.entry_date,
                        "entry_price_a": trade.entry_price_a,
                        "entry_price_b": trade.entry_price_b,
                        "shares_a": trade.shares_a,
                        "shares_b": trade.shares_b,
                        "entry_distance": trade.entry_distance,
                    })

        return positions

    def get_current_distance(self, pair: tuple, date: datetime) -> float | None:
        """Get the distance value for a pair at a specific date."""
        if pair not in self.distance_series:
            return None

        distance = self.distance_series[pair]
        if date in distance.index:
            return float(distance.loc[date])

        # Find nearest date
        mask = distance.index <= date
        if mask.any():
            return float(distance.loc[distance.index[mask][-1]])

        return None

    def calculate_unrealized_pnl(
        self,
        position: dict,
        date: datetime,
    ) -> float:
        """
        Calculate unrealized P&L for an open position at a given date.

        Args:
            position: Position dict from get_position_at_date
            date: Current date for P&L calculation

        Returns:
            Unrealized P&L in dollars
        """
        pair = position["pair"]

        # Get current prices
        if date not in self.trading_prices.index:
            # Find nearest date
            mask = self.trading_prices.index <= date
            if not mask.any():
                return 0.0
            date = self.trading_prices.index[mask][-1]

        current_price_a = self.trading_prices.loc[date, pair[0]]
        current_price_b = self.trading_prices.loc[date, pair[1]]

        # Calculate P&L based on direction
        if position["direction"] == 1:
            # Long spread: bought A, sold B
            pnl_a = (current_price_a - position["entry_price_a"]) * position["shares_a"]
            pnl_b = (position["entry_price_b"] - current_price_b) * position["shares_b"]
        else:
            # Short spread: sold A, bought B
            pnl_a = (position["entry_price_a"] - current_price_a) * position["shares_a"]
            pnl_b = (current_price_b - position["entry_price_b"]) * position["shares_b"]

        return pnl_a + pnl_b

    def get_spy_returns(self) -> pd.Series:
        """
        Get SPY returns normalized to trading period start.

        Returns:
            Series of SPY returns (percentage)
        """
        trading_start = self.trading_prices.index[0]

        # Align SPY to trading period
        spy = self.spy_prices["SPY"]
        spy_aligned = spy[spy.index >= trading_start]

        if len(spy_aligned) == 0:
            return pd.Series()

        # Normalize to start at 0%
        returns = (spy_aligned / spy_aligned.iloc[0] - 1) * 100
        return returns

    def get_strategy_returns(self, wait_mode: int = 1, calc_method: str = "committed") -> pd.Series:
        """
        Get strategy returns for the specified wait mode.

        Args:
            wait_mode: 0 or 1
            calc_method: "committed" (total pool) or "fully_invested" (deployed only)

        Returns:
            Series of strategy returns (percentage)
        """
        equity = self.get_equity_curve(wait_mode)
        initial = equity.iloc[0]

        if calc_method == "committed":
            # Return based on total capital pool
            total_capital = self.config["capital_per_trade"] * self.config["top_n_pairs"]
            returns = (equity - total_capital) / total_capital * 100
        else:
            # Simple return based on starting equity
            returns = (equity / initial - 1) * 100

        return returns

    def get_realized_pnl_at_date(self, date: datetime, wait_mode: int = 1) -> float:
        """Get total realized P&L from closed trades up to a date."""
        trades = self.get_all_trades(wait_mode)
        return sum(t.pnl for t in trades if t.exit_date <= date)

    def get_trading_dates(self) -> list:
        """Get list of trading dates."""
        return self.trading_prices.index.tolist()

    def get_formation_period_info(self) -> dict:
        """Get formation period date range info."""
        return {
            "start": self.formation_prices.index[0],
            "end": self.formation_prices.index[-1],
            "days": len(self.formation_prices),
        }

    def get_trading_period_info(self) -> dict:
        """Get trading period date range info."""
        return {
            "start": self.trading_prices.index[0],
            "end": self.trading_prices.index[-1],
            "days": len(self.trading_prices),
        }
