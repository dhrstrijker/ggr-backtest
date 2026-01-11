"""GGR Distance Backtester - A pair trading backtester using GGR methodology."""

from .data import fetch_prices, cache_prices, load_cached, fetch_benchmark
from .pairs import normalize_prices, calculate_ssd_matrix, select_top_pairs
from .signals import (
    calculate_spread,
    calculate_formation_stats,
    calculate_distance,
    generate_signals_ggr,
)
from .backtest import Trade, run_backtest
from .analysis import calculate_metrics, plot_equity_curve

__all__ = [
    "fetch_prices",
    "cache_prices",
    "load_cached",
    "fetch_benchmark",
    "normalize_prices",
    "calculate_ssd_matrix",
    "select_top_pairs",
    "calculate_spread",
    "calculate_formation_stats",
    "calculate_distance",
    "generate_signals_ggr",
    "Trade",
    "run_backtest",
    "calculate_metrics",
    "plot_equity_curve",
]
