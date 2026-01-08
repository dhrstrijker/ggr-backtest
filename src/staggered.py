"""Staggered portfolio methodology implementing GGR paper's overlapping portfolios.

Implements the Gatev, Goetzmann, and Rouwenhorst (2006) approach where:
- A new portfolio starts every month (overlap_days ~21 trading days)
- Each portfolio has 12-month formation + 6-month trading
- At steady state, 6 portfolios are active simultaneously
- Final monthly return = average across all active portfolios
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, BacktestResult, run_backtest, combine_results
from .pairs import normalize_prices, calculate_ssd_matrix, select_top_pairs


@dataclass
class StaggeredConfig:
    """Configuration for GGR staggered portfolio methodology."""

    formation_days: int = 252  # 12 months
    trading_days: int = 126  # 6 months
    overlap_days: int = 21  # ~1 month between portfolio starts
    n_pairs: int = 20  # Top pairs per portfolio
    min_data_pct: float = 0.95  # Min % of valid data required per symbol
    backtest_config: BacktestConfig = field(default_factory=BacktestConfig)


@dataclass
class PortfolioCycle:
    """Represents a single portfolio in the staggered methodology."""

    cycle_id: int
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    trading_start: pd.Timestamp
    trading_end: pd.Timestamp
    pairs: list[tuple[str, str]] | None = None
    results: dict[tuple[str, str], BacktestResult] | None = None
    valid_symbols: list[str] | None = None
    monthly_returns: pd.Series | None = None

    def is_active_on(self, date: pd.Timestamp) -> bool:
        """Check if portfolio is in its trading window on a given date."""
        return self.trading_start <= date <= self.trading_end


@dataclass
class StaggeredResult:
    """Results from staggered portfolio backtest."""

    cycles: list[PortfolioCycle]
    monthly_returns: pd.Series  # Averaged across active portfolios
    cumulative_returns: pd.Series
    active_portfolios_over_time: pd.Series
    config: StaggeredConfig
    all_trades: list | None = None

    @property
    def total_portfolios(self) -> int:
        """Total number of portfolio cycles."""
        return len(self.cycles)


def generate_portfolio_cycles(
    all_dates: pd.DatetimeIndex,
    config: StaggeredConfig,
) -> list[PortfolioCycle]:
    """
    Generate the schedule of portfolio cycles.

    Creates overlapping portfolios where each new portfolio starts
    every `overlap_days` trading days.

    Args:
        all_dates: Full date range of available price data
        config: Staggered configuration

    Returns:
        List of PortfolioCycle objects with formation/trading windows
    """
    cycles = []
    cycle_id = 0
    required_days = config.formation_days + config.trading_days
    start_idx = 0

    while start_idx + required_days <= len(all_dates):
        formation_start = all_dates[start_idx]
        formation_end_idx = start_idx + config.formation_days - 1
        formation_end = all_dates[formation_end_idx]

        trading_start_idx = formation_end_idx + 1
        trading_start = all_dates[trading_start_idx]
        trading_end_idx = trading_start_idx + config.trading_days - 1
        trading_end = all_dates[trading_end_idx]

        cycle = PortfolioCycle(
            cycle_id=cycle_id,
            formation_start=formation_start,
            formation_end=formation_end,
            trading_start=trading_start,
            trading_end=trading_end,
        )
        cycles.append(cycle)

        cycle_id += 1
        start_idx += config.overlap_days

    return cycles


def filter_valid_symbols(
    close_prices: pd.DataFrame,
    formation_start: pd.Timestamp,
    trading_end: pd.Timestamp,
    min_data_pct: float = 0.95,
) -> list[str]:
    """
    Filter symbols that have sufficient data for the full cycle.

    Args:
        close_prices: Full close price DataFrame
        formation_start: Start of formation period
        trading_end: End of trading period
        min_data_pct: Minimum percentage of valid data required (0-1)

    Returns:
        List of symbols with sufficient data
    """
    # Slice to the cycle's full range
    cycle_data = close_prices.loc[formation_start:trading_end]
    total_days = len(cycle_data)

    if total_days == 0:
        return []

    valid_symbols = []
    for symbol in close_prices.columns:
        valid_count = cycle_data[symbol].notna().sum()
        pct_valid = valid_count / total_days
        if pct_valid >= min_data_pct:
            valid_symbols.append(symbol)

    return valid_symbols


def run_portfolio_cycle(
    cycle: PortfolioCycle,
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    config: StaggeredConfig,
) -> PortfolioCycle:
    """
    Run backtest for a single portfolio cycle.

    Args:
        cycle: PortfolioCycle with dates set
        close_prices: Full close price DataFrame
        open_prices: Full open price DataFrame
        config: Configuration

    Returns:
        Updated PortfolioCycle with pairs selected and results populated
    """
    # Filter valid symbols for this cycle
    valid_symbols = filter_valid_symbols(
        close_prices,
        cycle.formation_start,
        cycle.trading_end,
        config.min_data_pct,
    )
    cycle.valid_symbols = valid_symbols

    if len(valid_symbols) < 2:
        # Not enough symbols to form pairs
        cycle.pairs = []
        cycle.results = {}
        return cycle

    # Slice formation period data (only valid symbols)
    formation_close = close_prices[valid_symbols].loc[
        cycle.formation_start : cycle.formation_end
    ]

    # Select pairs using SSD on formation period
    normalized = normalize_prices(formation_close)
    ssd_matrix = calculate_ssd_matrix(normalized)

    # Select top pairs (but not more than available)
    max_pairs = len(valid_symbols) * (len(valid_symbols) - 1) // 2
    n_pairs = min(config.n_pairs, max_pairs)
    pairs = select_top_pairs(ssd_matrix, n=n_pairs)
    cycle.pairs = pairs

    if not pairs:
        cycle.results = {}
        return cycle

    # Slice trading period data
    trading_close = close_prices[valid_symbols].loc[
        cycle.trading_start : cycle.trading_end
    ]
    trading_open = open_prices[valid_symbols].loc[
        cycle.trading_start : cycle.trading_end
    ]

    # Run backtest using existing infrastructure
    results = run_backtest(
        formation_close=formation_close,
        trading_close=trading_close,
        trading_open=trading_open,
        pairs=pairs,
        config=config.backtest_config,
    )
    cycle.results = results

    return cycle


def calculate_cycle_monthly_returns(
    cycle: PortfolioCycle,
    capital_per_pair: float,
) -> pd.Series:
    """
    Calculate monthly returns for a single completed cycle.

    Args:
        cycle: Completed PortfolioCycle with results
        capital_per_pair: Capital allocated per pair

    Returns:
        Series with monthly returns indexed by month-end date
    """
    if not cycle.results or not cycle.pairs:
        return pd.Series(dtype=float)

    # Combine results into equity curve
    initial_capital = capital_per_pair * len(cycle.pairs)
    _, combined_equity = combine_results(cycle.results, initial_capital)

    if len(combined_equity) < 2:
        return pd.Series(dtype=float)

    # Resample to month-end
    monthly_equity = combined_equity.resample("ME").last()

    # Forward fill any missing months within the trading period
    monthly_equity = monthly_equity.ffill()

    # Calculate returns
    monthly_returns = monthly_equity.pct_change().dropna()

    return monthly_returns


def aggregate_monthly_returns(
    cycles: list[PortfolioCycle],
) -> tuple[pd.Series, pd.Series]:
    """
    Average monthly returns across all active portfolios.

    Per GGR paper: At any month, the return is the average of
    returns from all portfolios currently in their trading window.

    Args:
        cycles: List of completed PortfolioCycles with monthly_returns set

    Returns:
        Tuple of (averaged_monthly_returns, active_portfolio_counts)
    """
    # Collect all monthly returns into a DataFrame
    returns_dict = {}
    for cycle in cycles:
        if cycle.monthly_returns is not None and len(cycle.monthly_returns) > 0:
            returns_dict[cycle.cycle_id] = cycle.monthly_returns

    if not returns_dict:
        return pd.Series(dtype=float), pd.Series(dtype=int)

    # Create DataFrame with all cycles
    returns_df = pd.DataFrame(returns_dict)

    # For each month, we only want to average cycles that are active
    # A cycle is active if the month falls within its trading window
    all_months = returns_df.index.tolist()
    averaged_returns = []
    active_counts = []

    for month in all_months:
        active_returns = []
        for cycle in cycles:
            if cycle.cycle_id in returns_df.columns:
                if cycle.is_active_on(month):
                    ret = returns_df.loc[month, cycle.cycle_id]
                    if pd.notna(ret):
                        active_returns.append(ret)

        if active_returns:
            averaged_returns.append(np.mean(active_returns))
            active_counts.append(len(active_returns))
        else:
            averaged_returns.append(np.nan)
            active_counts.append(0)

    return (
        pd.Series(averaged_returns, index=all_months),
        pd.Series(active_counts, index=all_months),
    )


def run_staggered_backtest(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    config: StaggeredConfig | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> StaggeredResult:
    """
    Run the full GGR staggered portfolio backtest.

    This is the main entry point for the methodology.

    Args:
        close_prices: Full history of close prices
        open_prices: Full history of open prices
        config: Staggered configuration (uses defaults if None)
        progress_callback: Optional callback for progress reporting (current, total)

    Returns:
        StaggeredResult with all portfolio cycles and aggregated returns
    """
    if config is None:
        config = StaggeredConfig()

    # Generate portfolio cycles
    all_dates = close_prices.index
    cycles = generate_portfolio_cycles(all_dates, config)

    if not cycles:
        raise ValueError(
            f"Not enough data for even one portfolio cycle. "
            f"Need at least {config.formation_days + config.trading_days} trading days."
        )

    # Run each cycle
    total_cycles = len(cycles)
    for i, cycle in enumerate(cycles):
        run_portfolio_cycle(cycle, close_prices, open_prices, config)

        # Calculate monthly returns for this cycle
        capital_per_pair = config.backtest_config.capital_per_trade
        cycle.monthly_returns = calculate_cycle_monthly_returns(cycle, capital_per_pair)

        if progress_callback:
            progress_callback(i + 1, total_cycles)

    # Aggregate monthly returns across active portfolios
    monthly_returns, active_counts = aggregate_monthly_returns(cycles)

    # Calculate cumulative returns
    cumulative_returns = (1 + monthly_returns.fillna(0)).cumprod() - 1

    # Collect all trades (optional)
    all_trades = []
    for cycle in cycles:
        if cycle.results:
            for result in cycle.results.values():
                all_trades.extend(result.trades)
    all_trades.sort(key=lambda t: t.exit_date)

    return StaggeredResult(
        cycles=cycles,
        monthly_returns=monthly_returns,
        cumulative_returns=cumulative_returns,
        active_portfolios_over_time=active_counts,
        config=config,
        all_trades=all_trades,
    )
