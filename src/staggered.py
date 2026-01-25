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

from .backtest import BacktestConfig, BacktestResult, run_backtest, run_backtest_parallel, combine_results
from .pairs import normalize_prices, calculate_ssd_matrix, select_top_pairs


@dataclass
class FormationCache:
    """Cache for formation period computations to avoid redundant SSD calculations.

    Caches SSD matrices and valid symbols for each unique formation period.
    Since consecutive cycles have ~99% overlapping formation data (with 21-day
    overlap vs 252-day formation), caching provides significant speedup.
    """

    ssd_matrices: dict[tuple[pd.Timestamp, pd.Timestamp], pd.DataFrame] = field(
        default_factory=dict
    )
    valid_symbols: dict[tuple[pd.Timestamp, pd.Timestamp], list[str]] = field(
        default_factory=dict
    )

    def get_or_compute(
        self,
        close_prices: pd.DataFrame,
        formation_start: pd.Timestamp,
        formation_end: pd.Timestamp,
    ) -> tuple[pd.DataFrame | None, list[str]]:
        """Get cached SSD matrix or compute and cache it.

        Args:
            close_prices: Full close price DataFrame
            formation_start: Start of formation period
            formation_end: End of formation period

        Returns:
            Tuple of (ssd_matrix, valid_symbols). ssd_matrix is None if
            insufficient symbols for pair formation.
        """
        key = (formation_start, formation_end)

        if key not in self.ssd_matrices:
            # Filter valid symbols
            valid = filter_valid_symbols(close_prices, formation_start, formation_end)
            self.valid_symbols[key] = valid

            if len(valid) >= 2:
                formation_close = close_prices[valid].loc[formation_start:formation_end]
                normalized = normalize_prices(formation_close)
                ssd_matrix = calculate_ssd_matrix(normalized)
                self.ssd_matrices[key] = ssd_matrix
            else:
                self.ssd_matrices[key] = None

        return self.ssd_matrices[key], self.valid_symbols[key]


@dataclass
class StaggeredConfig:
    """Configuration for GGR staggered portfolio methodology."""

    formation_days: int = 252  # 12 months
    trading_days: int = 126  # 6 months
    overlap_days: int = 21  # ~1 month between portfolio starts
    n_pairs: int = 10  # Top pairs per portfolio
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

        # Defensive bounds check for formation period
        assert formation_end_idx < len(all_dates), (
            f"Formation end index {formation_end_idx} exceeds data length {len(all_dates)}"
        )
        formation_end = all_dates[formation_end_idx]

        trading_start_idx = formation_end_idx + 1
        trading_end_idx = trading_start_idx + config.trading_days - 1

        # Defensive bounds check for trading period
        assert trading_end_idx < len(all_dates), (
            f"Trading end index {trading_end_idx} exceeds data length {len(all_dates)}"
        )
        trading_start = all_dates[trading_start_idx]
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
    formation_end: pd.Timestamp,
) -> list[str]:
    """
    Filter symbols that have complete data for the formation period.

    Per GGR paper methodology: Stock must have traded every single day during
    the 12-month formation period. No filtering based on trading period data
    as that would introduce look-ahead bias.

    If a stock stops trading during the trading period, it's handled by
    the delisting logic in backtest.py (close position at last available price).

    Args:
        close_prices: Full close price DataFrame
        formation_start: Start of formation period
        formation_end: End of formation period

    Returns:
        List of symbols with 100% data coverage during formation period
    """
    # Slice to formation period only - no look-ahead to trading period
    formation_data = close_prices.loc[formation_start:formation_end]
    formation_days = len(formation_data)

    if formation_days == 0:
        return []

    valid_symbols = []
    for symbol in close_prices.columns:
        # Formation: require 100% coverage (critical for sigma calculation)
        formation_valid = formation_data[symbol].notna().sum()
        if formation_valid == formation_days:
            valid_symbols.append(symbol)

    return valid_symbols


def run_portfolio_cycle(
    cycle: PortfolioCycle,
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    config: StaggeredConfig,
    cache: FormationCache | None = None,
    parallel: bool = False,
    n_jobs: int = -1,
) -> PortfolioCycle:
    """
    Run backtest for a single portfolio cycle.

    Args:
        cycle: PortfolioCycle with dates set
        close_prices: Full close price DataFrame
        open_prices: Full open price DataFrame
        config: Configuration
        cache: Optional FormationCache for caching SSD matrices
        parallel: If True, run pair backtests in parallel
        n_jobs: Number of parallel workers (-1 for all cores)

    Returns:
        Updated PortfolioCycle with pairs selected and results populated
    """
    # Use cache if provided, otherwise compute directly
    if cache is not None:
        ssd_matrix, valid_symbols = cache.get_or_compute(
            close_prices, cycle.formation_start, cycle.formation_end
        )
    else:
        # Filter valid symbols for this cycle (formation period only - no look-ahead)
        valid_symbols = filter_valid_symbols(
            close_prices,
            cycle.formation_start,
            cycle.formation_end,
        )

        if len(valid_symbols) >= 2:
            # Slice formation period data (only valid symbols)
            formation_close = close_prices[valid_symbols].loc[
                cycle.formation_start : cycle.formation_end
            ]
            # Select pairs using SSD on formation period
            normalized = normalize_prices(formation_close)
            ssd_matrix = calculate_ssd_matrix(normalized)
        else:
            ssd_matrix = None

    cycle.valid_symbols = valid_symbols

    if ssd_matrix is None or len(valid_symbols) < 2:
        # Not enough symbols to form pairs
        cycle.pairs = []
        cycle.results = {}
        return cycle

    # Select top pairs (but not more than available)
    max_pairs = len(valid_symbols) * (len(valid_symbols) - 1) // 2
    n_pairs = min(config.n_pairs, max_pairs)
    pairs = select_top_pairs(ssd_matrix, n=n_pairs)
    cycle.pairs = pairs

    if not pairs:
        cycle.results = {}
        return cycle

    # Slice formation period data for backtest (needed for static sigma calculation)
    formation_close = close_prices[valid_symbols].loc[
        cycle.formation_start : cycle.formation_end
    ]

    # Slice trading period data
    trading_close = close_prices[valid_symbols].loc[
        cycle.trading_start : cycle.trading_end
    ]
    trading_open = open_prices[valid_symbols].loc[
        cycle.trading_start : cycle.trading_end
    ]

    # Run backtest using existing infrastructure
    backtest_fn = run_backtest_parallel if parallel else run_backtest
    if parallel:
        results = backtest_fn(
            formation_close=formation_close,
            trading_close=trading_close,
            trading_open=trading_open,
            pairs=pairs,
            config=config.backtest_config,
            cycle_id=cycle.cycle_id,
            n_jobs=n_jobs,
        )
    else:
        results = backtest_fn(
            formation_close=formation_close,
            trading_close=trading_close,
            trading_open=trading_open,
            pairs=pairs,
            config=config.backtest_config,
            cycle_id=cycle.cycle_id,
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
    use_cache: bool = True,
    parallel: bool = False,
    n_jobs: int = -1,
) -> StaggeredResult:
    """
    Run the full GGR staggered portfolio backtest.

    This is the main entry point for the methodology.

    Args:
        close_prices: Full history of close prices
        open_prices: Full history of open prices
        config: Staggered configuration (uses defaults if None)
        progress_callback: Optional callback for progress reporting (current, total)
        use_cache: If True, cache SSD matrices to avoid redundant calculations
        parallel: If True, run pair backtests in parallel within each cycle
        n_jobs: Number of parallel workers (-1 for all cores)

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

    # Create cache for formation period computations (SSD matrices, valid symbols)
    # This significantly speeds up processing when many cycles share overlapping data
    cache = FormationCache() if use_cache else None

    # Run each cycle
    total_cycles = len(cycles)
    for i, cycle in enumerate(cycles):
        run_portfolio_cycle(
            cycle, close_prices, open_prices, config,
            cache=cache, parallel=parallel, n_jobs=n_jobs
        )

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


@dataclass
class PrecomputedFormations:
    """Precomputed formation period results that can be shared across wait modes.

    Contains cycles with pairs and valid_symbols set, but no backtest results.
    This allows running backtests with different wait_days without recomputing
    the expensive SSD matrices and pair selection.
    """

    cycles: list[PortfolioCycle]
    config: StaggeredConfig


def precompute_formations(
    close_prices: pd.DataFrame,
    config: StaggeredConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> PrecomputedFormations:
    """
    Precompute all formation period work (SSD matrices, pair selection).

    This is wait-mode independent and can be reused for multiple backtest runs
    with different wait_days settings. Use run_backtest_only() to run the
    backtest phase.

    Args:
        close_prices: Full history of close prices
        config: Staggered configuration
        progress_callback: Optional callback for progress reporting (current, total)

    Returns:
        PrecomputedFormations with cycles containing pairs but no results
    """
    # Generate portfolio cycles
    all_dates = close_prices.index
    cycles = generate_portfolio_cycles(all_dates, config)

    if not cycles:
        raise ValueError(
            f"Not enough data for even one portfolio cycle. "
            f"Need at least {config.formation_days + config.trading_days} trading days."
        )

    # Create cache for SSD computations
    cache = FormationCache()

    # Precompute formations for each cycle
    total_cycles = len(cycles)
    for i, cycle in enumerate(cycles):
        # Get SSD matrix and valid symbols from cache
        ssd_matrix, valid_symbols = cache.get_or_compute(
            close_prices, cycle.formation_start, cycle.formation_end
        )
        cycle.valid_symbols = valid_symbols

        if ssd_matrix is None or len(valid_symbols) < 2:
            cycle.pairs = []
        else:
            # Select top pairs
            max_pairs = len(valid_symbols) * (len(valid_symbols) - 1) // 2
            n_pairs = min(config.n_pairs, max_pairs)
            cycle.pairs = select_top_pairs(ssd_matrix, n=n_pairs)

        if progress_callback:
            progress_callback(i + 1, total_cycles)

    return PrecomputedFormations(cycles=cycles, config=config)


def run_backtest_only(
    precomputed: PrecomputedFormations,
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    backtest_config: BacktestConfig,
    progress_callback: Callable[[int, int], None] | None = None,
    parallel: bool = False,
    n_jobs: int = -1,
) -> StaggeredResult:
    """
    Run only the backtest phase using precomputed formations.

    This is the wait-mode specific part. Use with precompute_formations()
    to run multiple backtests with different wait_days settings without
    recomputing the expensive SSD matrices.

    Args:
        precomputed: PrecomputedFormations from precompute_formations()
        close_prices: Full history of close prices
        open_prices: Full history of open prices
        backtest_config: Backtest configuration (includes wait_days)
        progress_callback: Optional callback for progress reporting
        parallel: If True, run pair backtests in parallel
        n_jobs: Number of parallel workers

    Returns:
        StaggeredResult with backtest results
    """
    import copy

    # Deep copy cycles to avoid mutation issues (allow multiple runs)
    cycles = [copy.deepcopy(c) for c in precomputed.cycles]
    config = precomputed.config

    # Run backtest for each cycle
    total_cycles = len(cycles)
    for i, cycle in enumerate(cycles):
        if not cycle.pairs or not cycle.valid_symbols:
            cycle.results = {}
            cycle.monthly_returns = pd.Series(dtype=float)
            continue

        # Slice formation period data for backtest
        formation_close = close_prices[cycle.valid_symbols].loc[
            cycle.formation_start : cycle.formation_end
        ]

        # Slice trading period data
        trading_close = close_prices[cycle.valid_symbols].loc[
            cycle.trading_start : cycle.trading_end
        ]
        trading_open = open_prices[cycle.valid_symbols].loc[
            cycle.trading_start : cycle.trading_end
        ]

        # Run backtest
        backtest_fn = run_backtest_parallel if parallel else run_backtest
        if parallel:
            results = backtest_fn(
                formation_close=formation_close,
                trading_close=trading_close,
                trading_open=trading_open,
                pairs=cycle.pairs,
                config=backtest_config,
                cycle_id=cycle.cycle_id,
                n_jobs=n_jobs,
            )
        else:
            results = backtest_fn(
                formation_close=formation_close,
                trading_close=trading_close,
                trading_open=trading_open,
                pairs=cycle.pairs,
                config=backtest_config,
                cycle_id=cycle.cycle_id,
            )
        cycle.results = results

        # Calculate monthly returns for this cycle
        capital_per_pair = backtest_config.capital_per_trade
        cycle.monthly_returns = calculate_cycle_monthly_returns(cycle, capital_per_pair)

        if progress_callback:
            progress_callback(i + 1, total_cycles)

    # Aggregate monthly returns across active portfolios
    monthly_returns, active_counts = aggregate_monthly_returns(cycles)

    # Calculate cumulative returns
    cumulative_returns = (1 + monthly_returns.fillna(0)).cumprod() - 1

    # Collect all trades
    all_trades = []
    for cycle in cycles:
        if cycle.results:
            for result in cycle.results.values():
                all_trades.extend(result.trades)
    all_trades.sort(key=lambda t: t.exit_date)

    # Create config with the specific backtest_config
    result_config = StaggeredConfig(
        formation_days=config.formation_days,
        trading_days=config.trading_days,
        overlap_days=config.overlap_days,
        n_pairs=config.n_pairs,
        backtest_config=backtest_config,
    )

    return StaggeredResult(
        cycles=cycles,
        monthly_returns=monthly_returns,
        cumulative_returns=cumulative_returns,
        active_portfolios_over_time=active_counts,
        config=result_config,
        all_trades=all_trades,
    )
