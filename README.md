# GGR Distance Method - Pair Trading Backtester

A simple, self-contained backtester for the **Gatev, Goetzmann, and Rouwenhorst (GGR) Distance Method** of statistical pair trading.

## Table of Contents
- [The GGR Distance Method](#the-ggr-distance-method)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Testing](#testing)

---

## The GGR Distance Method

The GGR Distance Method is a foundational statistical arbitrage strategy first published by Gatev, Goetzmann, and Rouwenhorst in their 2006 paper *"Pairs Trading: Performance of a Relative-Value Arbitrage Rule"* (Review of Financial Studies).

### Core Concept

The strategy exploits **mean reversion** in the price relationship between similar stocks. When two historically correlated stocks diverge in price, the strategy bets they will converge back to their historical relationship.

### The Two-Phase Approach

#### Phase 1: Formation Period (Pair Selection)

During the formation period, we identify pairs of stocks that have moved together historically:

1. **Normalize Prices**: Convert all price series to start at 1.0
   ```
   P_normalized(t) = P(t) / P(0)
   ```

2. **Calculate Sum of Squared Differences (SSD)**: For each pair of stocks, measure how closely their normalized prices track each other
   ```
   SSD(A, B) = Σ (P_A_norm(t) - P_B_norm(t))²
   ```

3. **Select Top Pairs**: Choose pairs with the lowest SSD (most similar historical behavior)

**Why SSD?** Unlike correlation, SSD captures both the direction AND magnitude of price movements. Two stocks can be highly correlated but trade at very different levels - SSD penalizes this divergence.

#### Phase 2: Trading Period (Signal Generation & Execution)

For each selected pair, we monitor the spread and trade when it diverges:

1. **Calculate Spread**: The difference between normalized prices
   ```
   Spread(t) = P_A_norm(t) - P_B_norm(t)
   ```

2. **Calculate Distance**: Use the STATIC standard deviation from the formation period
   ```
   Distance(t) = Spread(t) / σ_formation
   ```

   **Important**: Unlike rolling Z-score methods, GGR uses a **fixed** σ calculated once during the formation period. This σ does not change during trading.

3. **Trading Signals**:
   - **Entry (Long Spread)**: When Distance < -2.0 (spread is unusually low)
     - Buy Stock A, Sell Stock B
     - Bet: A will outperform B as spread reverts to parity

   - **Entry (Short Spread)**: When Distance > 2.0 (spread is unusually high)
     - Sell Stock A, Buy Stock B
     - Bet: B will outperform A as spread reverts to parity

   - **Exit**: When spread **crosses zero** (prices converge/cross)
     - Per GGR paper: Exit occurs when normalized prices intersect
     - NOT at an arbitrary threshold like |Z| < 0.5

### Visual Example

```
Price (Normalized)
    │
1.2 │      Stock A ──────╮
    │                     ╲    ← Spread widens (Z > 2)
1.0 │─────────────────────────── Entry: Short spread
    │                     ╱
0.8 │      Stock B ──────╯
    │
    └────────────────────────── Time
                         │
                    Spread reverts, exit when Z < 0.5
```

### Why It Works (Theory)

1. **Mean Reversion**: Similar stocks in the same sector are driven by common factors. Short-term divergences are often noise that corrects.

2. **Market Neutrality**: By going long one stock and short another, you're hedged against broad market moves. Profit comes from the *relative* performance.

3. **Statistical Edge**: The 2σ entry threshold means we only trade when divergence is statistically significant (< 5% probability under normal distribution).

### Risks and Limitations

- **Regime Changes**: Pairs can permanently diverge (e.g., one company loses market share)
- **Convergence Timing**: The spread may take longer to converge than your holding period
- **Transaction Costs**: Frequent trading erodes profits
- **Crowding**: Popular pairs may have reduced alpha due to competition

---

## Quick Start

### 1. Setup

```bash
cd ggr-backtesting

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up your API key
cp .env.example .env
# Edit .env and add your Polygon.io API key
```

### 2. Run the Notebook

```bash
jupyter notebook ggr_backtest.ipynb
```

### 3. Run Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
ggr-backtesting/
├── README.md
├── requirements.txt
├── .env.example
├── ggr_backtest.ipynb      # Main notebook (the deliverable)
├── src/
│   ├── __init__.py
│   ├── data.py             # Polygon data fetching + caching + validation
│   ├── pairs.py            # Pair formation (SSD calculation)
│   ├── signals.py          # Z-score signals, entry/exit logic
│   ├── backtest.py         # Backtest engine
│   └── analysis.py         # Performance metrics + charts
└── tests/
    ├── __init__.py
    ├── test_data_integrity.py  # Data validation tests
    ├── test_zscore.py          # Z-score calculation tests
    ├── test_signals.py         # Signal generation tests
    ├── test_ssd.py             # SSD calculation tests
    └── test_backtest.py        # Backtest logic tests
```

---

## Configuration

Default parameters in the notebook:

```python
CONFIG = {
    # Universe - stocks to consider for pair formation
    "symbols": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM'],

    # Date range for backtest
    "start_date": "2022-01-01",
    "end_date": "2024-12-31",

    # Formation period - how much history to use for pair selection AND σ calculation
    "formation_days": 252,        # ~1 year of trading days

    # Signal parameters (GGR methodology)
    "entry_threshold": 2.0,       # Enter when |distance| > 2.0σ (from formation)
    "max_holding_days": 20,       # Force exit after N days (fallback)
    # Note: Exit occurs when spread crosses zero (GGR rule)

    # Portfolio parameters
    "top_n_pairs": 5,             # Number of pairs to trade
    "capital_per_trade": 10000,   # $ allocated per pair trade
    "commission": 0.001,          # 0.1% commission per trade
}
```

### Parameter Tuning Guide

| Parameter | Lower Value | Higher Value |
|-----------|-------------|--------------|
| `entry_threshold` | More trades, lower quality signals | Fewer trades, higher conviction |
| `max_holding_days` | Cut losses faster | Give trades more time to converge |
| `formation_days` | Recent behavior only, smaller σ | Longer-term relationships, larger σ |

**Note**: Per GGR methodology, exits occur when the spread crosses zero (prices converge), not at an arbitrary threshold.

---

## How It Works

### Data Flow

```
Polygon.io API → fetch_prices() → cache (CSV) → get_close_prices()
                                              → get_open_prices()
                                                      ↓
                                              normalize_prices()
                                                      ↓
                                            calculate_ssd_matrix()
                                                      ↓
                                             select_top_pairs()
                                                      ↓
                    ┌─────────────────────────────────────────────────────┐
                    │  For each pair:                                     │
                    │    Formation Period:                                │
                    │      calculate_spread() → calculate_formation_stats()│
                    │                            (static σ)               │
                    │    Trading Period:                                  │
                    │      calculate_spread() → calculate_distance()      │
                    │      run_backtest_single_pair()                     │
                    │        - Entry: |distance| > 2σ                     │
                    │        - Exit: spread crosses zero                  │
                    └─────────────────────────────────────────────────────┘
                                                      ↓
                                            combine_results()
                                                      ↓
                                           calculate_metrics()
```

### Key Implementation Details

1. **No Lookahead Bias**: Trades execute at the OPEN of the day AFTER the signal is generated. The signal uses the closing price, but execution happens at next-day open.

2. **Static σ (GGR Methodology)**: Unlike Bollinger-style approaches that use rolling statistics, the GGR method calculates standard deviation **once** during the formation period. This σ remains fixed throughout the trading period.

3. **Crossing Zero Exit**: Per the original GGR paper, positions are closed when normalized prices **cross** (spread = 0), not when they reach an arbitrary threshold like |Z| < 0.5.

4. **Data Alignment**: All symbols are aligned to common dates. If one symbol is missing a day, that day is dropped for all symbols.

5. **Commission Model**: Flat percentage commission applied on entry and exit.

---

## Testing

The test suite validates critical assumptions:

### Data Integrity Tests (`test_data_integrity.py`)
- Gap detection in trading data
- NaN value detection
- Zero/negative price detection
- Extreme price jump detection

### Signal Tests (`test_signals.py`)
- Entry triggers when |distance| > 2σ (static from formation)
- **Crossing-zero exit** - exit only when spread crosses zero (GGR rule)
- No early exit just because spread decreased
- Formation statistics calculated correctly (static, not rolling)
- Distance calculation uses fixed σ
- **Static volatility** - σ doesn't adapt to trading period volatility
- **Reopening after convergence** - multiple trades per pair allowed

### SSD Tests (`test_ssd.py`)
- Identical series have SSD = 0
- SSD is symmetric: SSD(A,B) = SSD(B,A)
- Different series have SSD > 0
- **Detailed manual calculation verification**
- **Formation period isolation** - SSD only uses formation data

### Backtest Tests (`test_backtest.py`)
- Trade execution happens at next-day open
- P&L calculation is correct
- Max holding days is enforced
- Exit reason correctly identifies "crossing" vs "max_holding"
- Formation period stats used correctly
- **Forced liquidation at period end**
- **Negative spread handling** (long entries)
- **Multiple pair portfolio tracking**

Run all tests:
```bash
pytest tests/ -v
```

---

## References

- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). *Pairs Trading: Performance of a Relative-Value Arbitrage Rule*. Review of Financial Studies, 19(3), 797-827.

---

## License

MIT
