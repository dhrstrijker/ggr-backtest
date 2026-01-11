"""Data store for dashboard - manages pre-computed staggered backtest results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import numpy as np

from src.data import fetch_or_load, fetch_benchmark, get_close_prices, get_open_prices
from src.backtest import BacktestConfig, Trade
from src.staggered import (
    StaggeredConfig,
    StaggeredResult,
    run_staggered_backtest,
    precompute_formations,
    run_backtest_only,
)
from src.analysis import (
    calculate_staggered_metrics,
    calculate_ggr_dollar_metrics,
    calculate_monthly_pnl_series,
    calculate_cumulative_pnl_series,
)


@dataclass
class DataStore:
    """
    Centralized data storage for the dashboard.

    Uses the GGR staggered portfolio methodology with overlapping portfolios.
    Pre-computes results for both Wait-0-Day and Wait-1-Day modes.
    """

    # Configuration (loaded from configs/sectors/*.json via dashboard.py)
    config: dict = field(default_factory=dict)

    # Price data
    close_prices: pd.DataFrame = field(default=None)
    open_prices: pd.DataFrame = field(default=None)
    spy_prices: pd.DataFrame = field(default=None)

    # Staggered backtest results (both wait modes)
    staggered_result_wait_0: StaggeredResult = field(default=None)
    staggered_result_wait_1: StaggeredResult = field(default=None)

    # Metrics (old equal-weighted style)
    metrics_wait_0: dict = field(default=None)
    metrics_wait_1: dict = field(default=None)

    # GGR Dollar-based metrics (fair metrics)
    ggr_metrics_wait_0: dict = field(default=None)
    ggr_metrics_wait_1: dict = field(default=None)

    # Aggregated pair statistics (across all cycles)
    pair_stats: dict = field(default_factory=dict)

    def load_or_compute(self, config: dict) -> None:
        """
        Load data and compute staggered backtest results for both wait modes.

        Args:
            config: Configuration dict loaded from configs/sectors/*.json
        """
        self.config = config

        print("=" * 60)
        print("GGR Staggered Portfolio Methodology")
        print("Loading data and computing backtest results...")
        print("=" * 60)

        # Step 1: Fetch price data
        self._load_price_data()

        # Step 2: Run staggered backtests for both wait modes
        self._run_staggered_backtests()

        # Step 3: Aggregate pair statistics
        self._aggregate_pair_stats()

        result = self.staggered_result_wait_1
        print("=" * 60)
        print("Data loading complete!")
        print(f"  - Portfolio cycles: {result.total_portfolios}")
        print(f"  - Avg active portfolios: {result.active_portfolios_over_time.mean():.1f}")
        print(f"  - Total months: {len(result.monthly_returns)}")
        print(f"  - Wait-1 trades: {len(result.all_trades)}")
        print(f"  - Wait-0 trades: {len(self.staggered_result_wait_0.all_trades)}")
        print("=" * 60)

    def _load_price_data(self) -> None:
        """Load price data from Polygon.io or cache."""
        print("\n[1/3] Loading price data...")

        # Fetch universe prices
        prices_df = fetch_or_load(
            symbols=self.config["symbols"],
            start_date=self.config["start_date"],
            end_date=self.config["end_date"],
        )

        self.close_prices = get_close_prices(prices_df)
        self.open_prices = get_open_prices(prices_df)

        print(f"  Date range: {self.close_prices.index[0].date()} to {self.close_prices.index[-1].date()}")
        print(f"  Trading days: {len(self.close_prices)}")
        print(f"  Symbols: {len(self.close_prices.columns)}")

        # Fetch SPY benchmark
        self.spy_prices = fetch_benchmark(
            start_date=self.config["start_date"],
            end_date=self.config["end_date"],
        )
        print(f"  SPY data loaded: {len(self.spy_prices)} days")

    def _run_staggered_backtests(self) -> None:
        """Run staggered backtests for both wait modes efficiently.

        Uses precompute_formations() to compute SSD matrices and pair selection
        once, then runs backtests for both wait modes using run_backtest_only().
        This eliminates redundant computation (~50% speedup).
        """
        print("\n[2/3] Running staggered backtests...")

        # Create base staggered config (wait-mode independent)
        base_config = StaggeredConfig(
            formation_days=self.config["formation_days"],
            trading_days=self.config["trading_days"],
            overlap_days=self.config["overlap_days"],
            n_pairs=self.config["n_pairs"],
        )

        # Phase 1: Precompute formations (SSD matrices, pair selection) - ONCE
        print("  Precomputing formations (shared between wait modes)...")
        precomputed = precompute_formations(
            self.close_prices,
            base_config,
            progress_callback=lambda i, n: print(f"    Cycle {i}/{n}") if i % 10 == 0 else None,
        )
        print(f"  Precomputed {len(precomputed.cycles)} cycles")

        # Phase 2a: Run Wait-1-Day backtest
        backtest_config_wait_1 = BacktestConfig(
            entry_threshold=self.config["entry_threshold"],
            max_holding_days=self.config["max_holding_days"],
            capital_per_trade=self.config["capital_per_trade"],
            commission=self.config["commission"],
            wait_days=1,
        )

        print("  Running Wait-1-Day backtest...")
        self.staggered_result_wait_1 = run_backtest_only(
            precomputed,
            self.close_prices,
            self.open_prices,
            backtest_config_wait_1,
            progress_callback=lambda i, n: print(f"    Cycle {i}/{n}") if i % 10 == 0 else None,
        )
        self.metrics_wait_1 = calculate_staggered_metrics(self.staggered_result_wait_1)
        print(f"  Wait-1-Day: {len(self.staggered_result_wait_1.all_trades)} trades across {self.staggered_result_wait_1.total_portfolios} cycles")

        # Phase 2b: Run Wait-0-Day backtest
        backtest_config_wait_0 = BacktestConfig(
            entry_threshold=self.config["entry_threshold"],
            max_holding_days=self.config["max_holding_days"],
            capital_per_trade=self.config["capital_per_trade"],
            commission=self.config["commission"],
            wait_days=0,
        )

        print("  Running Wait-0-Day backtest...")
        self.staggered_result_wait_0 = run_backtest_only(
            precomputed,
            self.close_prices,
            self.open_prices,
            backtest_config_wait_0,
            progress_callback=lambda i, n: print(f"    Cycle {i}/{n}") if i % 10 == 0 else None,
        )
        self.metrics_wait_0 = calculate_staggered_metrics(self.staggered_result_wait_0)
        print(f"  Wait-0-Day: {len(self.staggered_result_wait_0.all_trades)} trades across {self.staggered_result_wait_0.total_portfolios} cycles")

        # Calculate dollar-based GGR metrics
        print("  Calculating dollar-based GGR metrics...")
        self.ggr_metrics_wait_1 = calculate_ggr_dollar_metrics(
            self.staggered_result_wait_1,
            self.config["capital_per_trade"],
            self.config["n_pairs"],
        )
        self.ggr_metrics_wait_0 = calculate_ggr_dollar_metrics(
            self.staggered_result_wait_0,
            self.config["capital_per_trade"],
            self.config["n_pairs"],
        )

    def _aggregate_pair_stats(self) -> None:
        """Aggregate statistics for each pair across all cycles."""
        print("\n[3/3] Aggregating pair statistics...")

        result = self.staggered_result_wait_1

        # Collect all pairs that appear in any cycle
        all_pairs = set()
        for cycle in result.cycles:
            if cycle.pairs:
                all_pairs.update(cycle.pairs)

        # Aggregate stats for each pair
        for pair in all_pairs:
            pair_trades = [t for t in result.all_trades if t.pair == pair]
            cycles_with_pair = sum(1 for c in result.cycles if c.pairs and pair in c.pairs)

            if pair_trades:
                total_pnl = sum(t.pnl for t in pair_trades)
                wins = [t for t in pair_trades if t.pnl > 0]
                win_rate = len(wins) / len(pair_trades) if pair_trades else 0
            else:
                total_pnl = 0
                win_rate = 0

            self.pair_stats[pair] = {
                "pair": pair,
                "cycles_traded": cycles_with_pair,
                "total_trades": len(pair_trades),
                "total_pnl": total_pnl,
                "win_rate": win_rate,
            }

        print(f"  Aggregated stats for {len(self.pair_stats)} unique pairs")

    # -------------------------------------------------------------------------
    # Helper methods for dashboard
    # -------------------------------------------------------------------------

    def get_staggered_result(self, wait_mode: int = 1) -> StaggeredResult:
        """Get staggered result for specified wait mode."""
        return self.staggered_result_wait_1 if wait_mode == 1 else self.staggered_result_wait_0

    def get_metrics(self, wait_mode: int = 1) -> dict:
        """Get metrics for specified wait mode."""
        return self.metrics_wait_1 if wait_mode == 1 else self.metrics_wait_0

    def get_ggr_metrics(self, wait_mode: int = 1) -> dict:
        """Get dollar-based GGR metrics for specified wait mode."""
        return self.ggr_metrics_wait_1 if wait_mode == 1 else self.ggr_metrics_wait_0

    def get_all_trades(self, wait_mode: int = 1) -> list[Trade]:
        """Get all trades for specified wait mode."""
        result = self.get_staggered_result(wait_mode)
        return result.all_trades if result else []

    def get_monthly_returns(self, wait_mode: int = 1) -> pd.Series:
        """Get monthly returns (averaged across active portfolios)."""
        result = self.get_staggered_result(wait_mode)
        return result.monthly_returns if result else pd.Series()

    def get_cumulative_returns(self, wait_mode: int = 1) -> pd.Series:
        """Get cumulative returns."""
        result = self.get_staggered_result(wait_mode)
        return result.cumulative_returns if result else pd.Series()

    def get_active_portfolios(self, wait_mode: int = 1) -> pd.Series:
        """Get active portfolio count over time."""
        result = self.get_staggered_result(wait_mode)
        return result.active_portfolios_over_time if result else pd.Series()

    def get_spy_returns(self) -> pd.Series:
        """
        Get SPY returns aligned to the backtest period.

        Returns:
            Series of SPY cumulative returns (as decimal, e.g., 0.10 = 10%)
        """
        result = self.staggered_result_wait_1
        if not result or result.monthly_returns is None:
            return pd.Series()

        # Get the trading period from the first cycle's trading start
        first_trading_start = min(c.trading_start for c in result.cycles)

        # Align SPY to trading period
        spy = self.spy_prices["SPY"]
        spy_aligned = spy[spy.index >= first_trading_start]

        if len(spy_aligned) == 0:
            return pd.Series()

        # Calculate cumulative returns
        returns = spy_aligned / spy_aligned.iloc[0] - 1
        return returns

    def get_monthly_pnl(self, wait_mode: int = 1) -> pd.Series:
        """Get monthly P&L in dollars (not percentage returns)."""
        trades = self.get_all_trades(wait_mode)
        return calculate_monthly_pnl_series(trades)

    def get_cumulative_pnl(self, wait_mode: int = 1) -> pd.Series:
        """Get cumulative P&L in dollars over time."""
        trades = self.get_all_trades(wait_mode)
        return calculate_cumulative_pnl_series(trades)

    def get_trades_for_pair(self, pair: tuple, wait_mode: int = 1) -> list[Trade]:
        """Get all trades for a specific pair across all cycles."""
        all_trades = self.get_all_trades(wait_mode)
        return [t for t in all_trades if t.pair == pair]

    def get_pair_stats_list(self) -> list[dict]:
        """Get list of pair statistics sorted by total P&L."""
        stats_list = list(self.pair_stats.values())
        stats_list.sort(key=lambda x: x["total_pnl"], reverse=True)
        return stats_list

    def get_all_pairs(self) -> list[tuple]:
        """Get all unique pairs that appear in any cycle."""
        return list(self.pair_stats.keys())

    def get_cycles_for_pair(self, pair: tuple, wait_mode: int = 1) -> list:
        """Get list of cycle IDs where this pair was traded."""
        result = self.get_staggered_result(wait_mode)
        cycles = []
        for cycle in result.cycles:
            if cycle.pairs and pair in cycle.pairs:
                cycles.append(cycle.cycle_id)
        return cycles

    def get_date_range_info(self) -> dict:
        """Get date range info for the backtest."""
        return {
            "start": self.close_prices.index[0],
            "end": self.close_prices.index[-1],
            "days": len(self.close_prices),
        }

    def get_backtest_info(self, wait_mode: int = 1) -> dict:
        """Get summary info about the backtest."""
        result = self.get_staggered_result(wait_mode)
        metrics = self.get_metrics(wait_mode)

        return {
            "total_cycles": result.total_portfolios,
            "avg_active_portfolios": result.active_portfolios_over_time.mean(),
            "total_trades": len(result.all_trades),
            "total_months": len(result.monthly_returns),
            "annualized_return": metrics.get("annualized_return", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
        }
