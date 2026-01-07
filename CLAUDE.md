# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GGR Distance Method backtester implementing the Gatev, Goetzmann, and Rouwenhorst (2006) statistical pair trading strategy. This is **not** a Bollinger-style rolling Z-score approach.

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
    → src/signals.py (formation stats + distance) → src/backtest.py (trades)
    → src/analysis.py (metrics + charts)
```

### Key Modules

- **`src/data.py`**: Polygon.io data fetching with CSV caching, price extraction, data quality validation
- **`src/pairs.py`**: Price normalization, SSD matrix calculation, pair selection by lowest SSD
- **`src/signals.py`**: Formation stats (static σ), distance calculation, GGR signal generation
- **`src/backtest.py`**: `BacktestConfig`, `Trade`, `BacktestResult` dataclasses; `run_backtest()` orchestration
- **`src/analysis.py`**: Performance metrics, equity curves, trade visualizations

### GGR Methodology (Critical Implementation Details)

1. **Static σ**: Standard deviation calculated **once** during formation period and remains fixed during trading. Do not use rolling statistics.

2. **Entry**: When `|distance| > 2σ` where `distance = spread / σ_formation`

3. **Exit**: When spread **crosses zero** (sign change). Not at an arbitrary threshold like |Z| < 0.5.

4. **No Lookahead**: Signals generated at close, trades execute at **next-day open**

### Function Signatures

```python
# signals.py - GGR functions
calculate_formation_stats(spread: pd.Series) -> dict  # Returns {'mean': float, 'std': float}
calculate_distance(spread: pd.Series, formation_std: float) -> pd.Series
generate_signals_ggr(spread: pd.Series, formation_std: float, entry_threshold: float = 2.0) -> pd.Series

# backtest.py - Main entry point
run_backtest(
    formation_close: pd.DataFrame,  # For calculating static σ
    trading_close: pd.DataFrame,
    trading_open: pd.DataFrame,
    pairs: list[tuple[str, str]],
    config: BacktestConfig,
) -> dict[tuple[str, str], BacktestResult]
```

### Trade Object

`Trade` dataclass uses `entry_distance` and `exit_distance` (not `entry_zscore`/`exit_zscore`). Exit reasons: `'crossing'`, `'max_holding'`, `'end_of_data'`.

## Environment

Requires `POLYGON_API_KEY` in `.env` file for data fetching. Copy from `.env.example`.

## Working Guidelines

**Never assume function signatures, API parameters, or implementation details.** Before using any function or API:

1. **Read the source code** to verify the actual signature and parameters
2. **Use AskUserQuestion** if something is unclear or ambiguous
3. **Check existing tests** for usage examples

This prevents bugs like passing non-existent parameters (e.g., `initial_capital` to a function that doesn't accept it) or using deprecated API methods.
