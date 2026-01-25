# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GGR Distance Method backtester implementing the Gatev, Goetzmann, and Rouwenhorst (2006) statistical pair trading strategy with overlapping portfolio methodology.

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_signals.py -v

# Run a specific test
pytest tests/test_signals.py::TestGGRSignalGeneration::test_entry_on_divergence -v

# Run the notebook
jupyter notebook ggr_backtest.ipynb

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### Data Flow
```
Polygon.io API → src/data.py (fetch + cache) → src/pairs.py (SSD matrix)
    → src/staggered.py (portfolio cycles) → src/backtest.py (uses signals.py internally)
    → src/analysis.py (metrics + charts)
```

### Key Modules

- **`src/data.py`**: Polygon.io data fetching with CSV caching, price extraction, data quality validation
- **`src/pairs.py`**: Price normalization, SSD matrix calculation, pair selection by lowest SSD
- **`src/signals.py`**: Formation stats (static σ), distance calculation, GGR signal generation
- **`src/backtest.py`**: `BacktestConfig`, `Trade`, `BacktestResult` dataclasses; `run_backtest()` orchestration
- **`src/staggered.py`**: Overlapping portfolio methodology with monthly rotation (6 active portfolios at steady state)
- **`src/analysis.py`**: Performance metrics, equity curves, trade visualizations

### GGR Methodology (Critical Implementation Details)

1. **Static σ**: Standard deviation calculated **once** during formation period and remains fixed during trading. Do not use rolling statistics.

2. **Entry**: When `|distance| > 2σ` where `distance = spread / σ_formation`

3. **Exit**: When spread **crosses zero** (sign change), indicating full mean reversion.

4. **No Lookahead**: Signals generated at close, trades execute at **next-day open**

5. **Formation Filtering**: Symbols must have **100% data coverage** during the 12-month formation period. No filtering based on trading period (that would be look-ahead bias).

6. **Delisting Handling**: If a stock delists during trading, close position at last available price with `exit_reason='delisting'`.

### NaN Handling (UNION Data Alignment)

Data uses UNION of dates (all trading days) rather than INTERSECTION. This allows symbols with different listing periods to coexist:

- **`normalize_prices()`**: Drops columns where first price is NaN
- **`calculate_ssd()`**: Requires 50% overlap; returns `inf` for pairs with insufficient data
- **`filter_valid_symbols()`**: 100% formation coverage required (no `min_data_pct` threshold)
- **Entry validation**: Skips trade entry if prices are NaN
- **Exit safety**: Uses last valid prices when end-of-data prices are NaN

### Function Signatures

```python
# signals.py - GGR functions
calculate_formation_stats(spread: pd.Series) -> dict  # Returns {'mean': float, 'std': float}
calculate_distance(spread: pd.Series, formation_std: float) -> pd.Series
generate_signals_ggr(spread: pd.Series, formation_std: float, entry_threshold: float = 2.0) -> pd.Series

# backtest.py - Single/multi-pair backtest
run_backtest(
    formation_close: pd.DataFrame,  # For calculating static σ
    trading_close: pd.DataFrame,
    trading_open: pd.DataFrame,
    pairs: list[tuple[str, str]],
    config: BacktestConfig,
) -> dict[tuple[str, str], BacktestResult]

# staggered.py - Full GGR methodology with portfolio rotation
run_staggered_backtest(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    config: StaggeredConfig,
) -> StaggeredResult
```

### Trade Object

`Trade` dataclass uses `entry_distance` and `exit_distance` (distance in σ units). Exit reasons: `'crossing'`, `'max_holding'`, `'end_of_data'`, `'delisting'`.

### Metrics Calculation

- **Win Rate**: Excludes break-even trades (pnl == 0) from both numerator and denominator. Formula: `wins / (wins + losses)`
- **Monthly P&L**: Aggregated by EXIT month (when P&L is realized), not entry month

### Dashboard

Interactive Dash web app in `dashboard/` directory:
- **`dashboard.py`**: Main entry point with sector CLI flags
- **`dashboard/data_store.py`**: Centralized data management, runs full backtest pipeline for both wait modes
- **`dashboard/layouts/`**: Page layouts (page1_fund_overview, page2_pair_inspector, page3_pairs_summary)
- **`dashboard/callbacks/`**: Interactive callbacks for each page

#### DataStore Wait-Mode Awareness

The `DataStore` maintains separate data for Wait-0-Day and Wait-1-Day modes:
- `pair_stats_wait_0` / `pair_stats_wait_1`: Aggregated pair statistics per mode
- All getter methods accept `wait_mode: int = 1` parameter (e.g., `get_all_pairs(wait_mode)`, `get_spy_returns(wait_mode)`)
- Win rate calculation excludes break-even trades (matching `src/analysis.py`)
- Use `is_initialized()` to check if data loaded successfully

#### Sector CLI Flags

```bash
python dashboard.py --utilities     # Utilities sector (default, ~34 stocks)
python dashboard.py --tech          # Technology sector (~50 stocks)
python dashboard.py --shipping      # Shipping sector (~20 stocks)
python dashboard.py --us-market     # S&P 500 universe (~500 stocks)
python dashboard.py --config path/to/config.json  # Custom config
```

### Configuration

All configuration is centralized in `configs/sectors/` as JSON files (no hardcoded defaults in code):

```
configs/
└── sectors/
    ├── utilities.json    # Default sector (~34 stocks)
    ├── tech.json         # Technology (~50 stocks)
    ├── shipping.json     # Shipping (~20 stocks)
    └── us_market.json    # S&P 500 (~500 stocks)
```

Config files specify: `symbols`, `start_date`, `end_date`, `formation_days`, `trading_days`, `overlap_days`, `n_pairs`, `entry_threshold`, `max_holding_days`, `capital_per_trade`, `commission`.

## Environment

Requires `POLYGON_API_KEY` in `.env` file for data fetching. Copy from `.env.example`.

## Working Guidelines

**Never assume function signatures, API parameters, or implementation details.** Before using any function or API:

1. **Read the source code** to verify the actual signature and parameters
2. **Use AskUserQuestion** if something is unclear or ambiguous
3. **Check existing tests** for usage examples

This prevents bugs like passing non-existent parameters (e.g., `initial_capital` to a function that doesn't accept it) or using deprecated API methods.
