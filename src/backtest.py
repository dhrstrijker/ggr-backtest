"""Backtest engine for GGR Distance Method pair trading strategy.

Implements the original Gatev, Goetzmann, and Rouwenhorst (2006) methodology:
- Static σ calculated from formation period
- Entry when |spread| > 2σ
- Exit when spread crosses 0 (prices converge)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .signals import (
    calculate_spread,
    calculate_formation_stats,
    calculate_distance,
    generate_signals_ggr,
)


@dataclass
class Trade:
    """Represents a single completed trade."""

    pair: tuple[str, str]
    direction: int  # 1 = long spread, -1 = short spread
    entry_date: datetime
    exit_date: datetime
    entry_price_a: float
    entry_price_b: float
    exit_price_a: float
    exit_price_b: float
    shares_a: float
    shares_b: float
    pnl: float
    pnl_pct: float
    holding_days: int
    entry_distance: float  # Distance in σ at entry
    exit_distance: float  # Distance in σ at exit
    exit_reason: str  # 'crossing', 'max_holding', 'end_of_data'
    max_adverse_spread: float  # Maximum adverse spread (in sigma) during trade
    cycle_id: int | None = None  # Which cycle generated this trade (for staggered)

    def to_dict(self) -> dict:
        """Convert trade to dictionary."""
        return {
            "pair": f"{self.pair[0]}/{self.pair[1]}",
            "direction": "Long" if self.direction == 1 else "Short",
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "entry_price_a": self.entry_price_a,
            "entry_price_b": self.entry_price_b,
            "exit_price_a": self.exit_price_a,
            "exit_price_b": self.exit_price_b,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "holding_days": self.holding_days,
            "entry_distance": self.entry_distance,
            "exit_distance": self.exit_distance,
            "exit_reason": self.exit_reason,
            "max_adverse_spread": self.max_adverse_spread,
            "cycle_id": self.cycle_id,
        }


@dataclass
class BacktestConfig:
    """Configuration for GGR backtest.

    Per GGR paper methodology:
    - entry_threshold: Number of formation-period σ for entry (default 2.0)
    - Exit occurs when spread crosses 0 (no exit_threshold needed)
    - wait_days: Days to wait after signal before executing (GGR paper tests both)
        - wait_days=1 (default): Execute at next-day OPEN (reduces bid-ask bounce)
        - wait_days=0: Execute at same-day CLOSE (original signal day)
    """

    entry_threshold: float = 2.0  # Entry at N sigma (from formation period)
    max_holding_days: int = 20  # Maximum days to hold a position
    capital_per_trade: float = 10000.0  # Capital allocated per pair trade
    commission: float = 0.001  # Commission rate (0.1%)
    wait_days: int = 1  # Days to wait after signal (0 = same-day, 1 = next-day)


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    trades: list[Trade]
    equity_curve: pd.Series
    config: BacktestConfig
    pair: tuple[str, str]


def run_backtest_single_pair(
    formation_close: pd.DataFrame,
    trading_close: pd.DataFrame,
    trading_open: pd.DataFrame,
    pair: tuple[str, str],
    config: BacktestConfig,
    cycle_id: int | None = None,
) -> BacktestResult:
    """
    Run GGR backtest for a single pair.

    Key GGR methodology:
    1. Calculate σ from formation period (static, not rolling)
    2. Enter when |spread| > entry_threshold * σ
    3. Exit when spread crosses 0 (prices converge)

    Execution timing (controlled by config.wait_days):
    - wait_days=1 (default): Execute at OPEN of day AFTER signal
    - wait_days=0: Execute at CLOSE of signal day (same day)

    Args:
        formation_close: Close prices for formation period (for calculating σ)
        trading_close: Close prices for trading period (for signals)
        trading_open: Open prices for trading period (for execution)
        pair: Tuple of (symbol_a, symbol_b)
        config: Backtest configuration

    Returns:
        BacktestResult with trades and equity curve
    """
    sym_a, sym_b = pair

    # Calculate formation period statistics (STATIC - per GGR paper)
    formation_spread = calculate_spread(
        formation_close[sym_a],
        formation_close[sym_b],
        normalize=True,
    )
    formation_stats = calculate_formation_stats(formation_spread)
    formation_std = formation_stats['std']

    # Calculate trading period spread (normalized from START of trading period)
    trading_spread = calculate_spread(
        trading_close[sym_a],
        trading_close[sym_b],
        normalize=True,
    )

    # Calculate distance using STATIC formation σ
    distance = calculate_distance(trading_spread, formation_std)

    # Track trades and equity
    trades = []
    equity = [config.capital_per_trade]
    equity_dates = [trading_close.index[0]]

    # Current position state
    position = None
    position_entry_idx = None

    dates = trading_close.index.tolist()
    entry_level = config.entry_threshold  # In σ units

    for i in range(len(dates) - 1):  # -1 because we execute next day
        current_date = dates[i]
        next_date = dates[i + 1]
        current_spread = trading_spread.iloc[i]
        current_distance = distance.iloc[i]

        # Skip if no valid spread
        if pd.isna(current_spread):
            equity.append(equity[-1])
            equity_dates.append(next_date)
            continue

        # Check for exit conditions
        if position is not None:
            days_held = i - position_entry_idx
            should_exit = False
            exit_reason = None

            # Track maximum adverse excursion (MAE)
            if position["direction"] == 1:  # Long spread (entered when spread too low/negative)
                # Adverse = spread moving even more negative (further from 0)
                if current_distance < position["max_adverse_distance"]:
                    position["max_adverse_distance"] = current_distance
            else:  # Short spread (entered when spread too high/positive)
                # Adverse = spread moving even more positive (further from 0)
                if current_distance > position["max_adverse_distance"]:
                    position["max_adverse_distance"] = current_distance

            # Check max holding days (fallback)
            if days_held >= config.max_holding_days:
                should_exit = True
                exit_reason = "max_holding"

            # GGR exit: Check for spread crossing zero (sign change)
            if not should_exit and i > 0:
                prev_spread = trading_spread.iloc[i - 1]
                crossed_zero = (
                    (prev_spread > 0 and current_spread <= 0) or
                    (prev_spread < 0 and current_spread >= 0)
                )
                if crossed_zero:
                    should_exit = True
                    exit_reason = "crossing"

            if should_exit:
                # Exit timing depends on wait_days setting
                if config.wait_days == 0:
                    # Same-day execution at CLOSE
                    exit_price_a = trading_close[sym_a].iloc[i]
                    exit_price_b = trading_close[sym_b].iloc[i]
                    exit_date = current_date
                else:
                    # Next-day execution at OPEN (default)
                    exit_price_a = trading_open[sym_a].iloc[i + 1]
                    exit_price_b = trading_open[sym_b].iloc[i + 1]
                    exit_date = next_date

                # Calculate P&L
                if position["direction"] == 1:
                    # Long spread: bought A, sold B
                    pnl_a = (exit_price_a - position["entry_price_a"]) * position["shares_a"]
                    pnl_b = (position["entry_price_b"] - exit_price_b) * position["shares_b"]
                else:
                    # Short spread: sold A, bought B
                    pnl_a = (position["entry_price_a"] - exit_price_a) * position["shares_a"]
                    pnl_b = (exit_price_b - position["entry_price_b"]) * position["shares_b"]

                gross_pnl = pnl_a + pnl_b

                # Commission on exit
                exit_value = (exit_price_a * position["shares_a"] +
                              exit_price_b * position["shares_b"])
                commission = exit_value * config.commission
                net_pnl = gross_pnl - commission

                pnl_pct = net_pnl / config.capital_per_trade

                trade = Trade(
                    pair=pair,
                    direction=position["direction"],
                    entry_date=position["entry_date"],
                    exit_date=exit_date,
                    entry_price_a=position["entry_price_a"],
                    entry_price_b=position["entry_price_b"],
                    exit_price_a=exit_price_a,
                    exit_price_b=exit_price_b,
                    shares_a=position["shares_a"],
                    shares_b=position["shares_b"],
                    pnl=net_pnl,
                    pnl_pct=pnl_pct,
                    holding_days=days_held,
                    entry_distance=position["entry_distance"],
                    exit_distance=current_distance,
                    exit_reason=exit_reason,
                    max_adverse_spread=position["max_adverse_distance"],
                    cycle_id=cycle_id,
                )
                trades.append(trade)

                # Update equity
                new_equity = equity[-1] + net_pnl
                equity.append(new_equity)
                equity_dates.append(exit_date)

                position = None
                position_entry_idx = None
                continue

        # Check for entry signals (only if flat) - GGR uses static σ
        if position is None:
            signal = 0
            if current_distance > entry_level:
                # Spread too high - short the spread (sell A, buy B)
                signal = -1
            elif current_distance < -entry_level:
                # Spread too low - long the spread (buy A, sell B)
                signal = 1

            if signal != 0:
                # Entry timing depends on wait_days setting
                if config.wait_days == 0:
                    # Same-day execution at CLOSE
                    entry_price_a = trading_close[sym_a].iloc[i]
                    entry_price_b = trading_close[sym_b].iloc[i]
                    entry_date = current_date
                    entry_idx = i
                else:
                    # Next-day execution at OPEN (default)
                    entry_price_a = trading_open[sym_a].iloc[i + 1]
                    entry_price_b = trading_open[sym_b].iloc[i + 1]
                    entry_date = next_date
                    entry_idx = i + 1

                # Calculate shares (equal dollar allocation to each leg)
                half_capital = config.capital_per_trade / 2
                shares_a = half_capital / entry_price_a
                shares_b = half_capital / entry_price_b

                # Commission on entry
                entry_value = half_capital * 2
                commission = entry_value * config.commission

                position = {
                    "direction": signal,
                    "entry_date": entry_date,
                    "entry_price_a": entry_price_a,
                    "entry_price_b": entry_price_b,
                    "shares_a": shares_a,
                    "shares_b": shares_b,
                    "entry_distance": current_distance,
                    "entry_commission": commission,
                    "max_adverse_distance": current_distance,  # Track MAE
                }
                position_entry_idx = entry_idx

                # Deduct commission from equity
                equity.append(equity[-1] - commission)
                equity_dates.append(entry_date)
                continue

        # No action - equity stays same
        equity.append(equity[-1])
        equity_dates.append(next_date)

    # Close any open position at end of data
    if position is not None:
        exit_price_a = trading_close[sym_a].iloc[-1]
        exit_price_b = trading_close[sym_b].iloc[-1]

        if position["direction"] == 1:
            pnl_a = (exit_price_a - position["entry_price_a"]) * position["shares_a"]
            pnl_b = (position["entry_price_b"] - exit_price_b) * position["shares_b"]
        else:
            pnl_a = (position["entry_price_a"] - exit_price_a) * position["shares_a"]
            pnl_b = (exit_price_b - position["entry_price_b"]) * position["shares_b"]

        gross_pnl = pnl_a + pnl_b
        exit_value = (exit_price_a * position["shares_a"] +
                      exit_price_b * position["shares_b"])
        commission = exit_value * config.commission
        net_pnl = gross_pnl - commission

        days_held = len(dates) - 1 - position_entry_idx

        trade = Trade(
            pair=pair,
            direction=position["direction"],
            entry_date=position["entry_date"],
            exit_date=dates[-1],
            entry_price_a=position["entry_price_a"],
            entry_price_b=position["entry_price_b"],
            exit_price_a=exit_price_a,
            exit_price_b=exit_price_b,
            shares_a=position["shares_a"],
            shares_b=position["shares_b"],
            pnl=net_pnl,
            pnl_pct=net_pnl / config.capital_per_trade,
            holding_days=days_held,
            entry_distance=position["entry_distance"],
            exit_distance=distance.iloc[-1],
            exit_reason="end_of_data",
            max_adverse_spread=position["max_adverse_distance"],
            cycle_id=cycle_id,
        )
        trades.append(trade)
        equity[-1] += net_pnl

    equity_series = pd.Series(equity, index=equity_dates)

    return BacktestResult(
        trades=trades,
        equity_curve=equity_series,
        config=config,
        pair=pair,
    )


def run_backtest(
    formation_close: pd.DataFrame,
    trading_close: pd.DataFrame,
    trading_open: pd.DataFrame,
    pairs: list[tuple[str, str]],
    config: BacktestConfig | None = None,
    cycle_id: int | None = None,
) -> dict[tuple[str, str], BacktestResult]:
    """
    Run GGR backtest for multiple pairs.

    Args:
        formation_close: Close prices for formation period (for calculating σ)
        trading_close: Close prices for trading period (for signals)
        trading_open: Open prices for trading period (for execution)
        pairs: List of pairs to backtest
        config: Backtest configuration (optional)
        cycle_id: Cycle identifier for staggered backtests (optional)

    Returns:
        Dictionary mapping pair to BacktestResult
    """
    if config is None:
        config = BacktestConfig()

    results = {}
    for pair in pairs:
        result = run_backtest_single_pair(
            formation_close, trading_close, trading_open, pair, config, cycle_id
        )
        results[pair] = result

    return results


def combine_results(
    results: dict[tuple[str, str], BacktestResult],
    initial_capital: float = 50000.0,
) -> tuple[list[Trade], pd.Series]:
    """
    Combine results from multiple pairs into aggregate metrics.

    Args:
        results: Dictionary of BacktestResults from run_backtest
        initial_capital: Total initial capital

    Returns:
        Tuple of (all_trades, combined_equity_curve)
    """
    all_trades = []
    for pair, result in results.items():
        all_trades.extend(result.trades)

    # Sort trades by exit date
    all_trades.sort(key=lambda t: t.exit_date)

    # Build combined equity curve
    all_dates = set()
    for result in results.values():
        all_dates.update(result.equity_curve.index)
    all_dates = sorted(all_dates)

    combined_equity = pd.Series(index=all_dates, data=initial_capital)

    for trade in all_trades:
        mask = combined_equity.index >= trade.exit_date
        combined_equity[mask] += trade.pnl

    return all_trades, combined_equity
