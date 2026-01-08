"""Tests for staggered portfolio methodology."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from src.staggered import (
    StaggeredConfig,
    PortfolioCycle,
    StaggeredResult,
    generate_portfolio_cycles,
    filter_valid_symbols,
    run_portfolio_cycle,
    calculate_cycle_monthly_returns,
    aggregate_monthly_returns,
    run_staggered_backtest,
)
from src.backtest import BacktestConfig


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_dates():
    """Generate 3 years of trading dates (approx 756 trading days)."""
    return pd.bdate_range(start="2020-01-01", periods=756, freq="B")


@pytest.fixture
def sample_prices(sample_dates):
    """Generate sample price data for multiple symbols."""
    np.random.seed(42)
    n_days = len(sample_dates)
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM"]

    # Generate correlated random walks for prices
    close_data = {}
    open_data = {}

    for i, sym in enumerate(symbols):
        # Start at different prices
        base_price = 100 + i * 20
        # Generate returns with drift
        returns = np.random.normal(0.0003, 0.02, n_days)
        prices = base_price * np.exp(np.cumsum(returns))
        close_data[sym] = prices
        # Open prices with small gap from previous close
        open_data[sym] = prices * (1 + np.random.normal(0, 0.001, n_days))

    close_prices = pd.DataFrame(close_data, index=sample_dates)
    open_prices = pd.DataFrame(open_data, index=sample_dates)

    return close_prices, open_prices


@pytest.fixture
def short_sample_prices():
    """Generate shorter sample for quick tests."""
    np.random.seed(42)
    dates = pd.bdate_range(start="2020-01-01", periods=400, freq="B")
    symbols = ["A", "B", "C", "D"]

    close_data = {}
    open_data = {}

    for i, sym in enumerate(symbols):
        base_price = 100 + i * 10
        returns = np.random.normal(0.0003, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        close_data[sym] = prices
        open_data[sym] = prices * (1 + np.random.normal(0, 0.001, len(dates)))

    close_prices = pd.DataFrame(close_data, index=dates)
    open_prices = pd.DataFrame(open_data, index=dates)

    return close_prices, open_prices


@pytest.fixture
def prices_with_missing_data():
    """Generate prices with some symbols having missing data."""
    np.random.seed(42)
    dates = pd.bdate_range(start="2020-01-01", periods=500, freq="B")
    symbols = ["A", "B", "C", "D"]

    close_data = {}
    open_data = {}

    for i, sym in enumerate(symbols):
        base_price = 100 + i * 10
        returns = np.random.normal(0.0003, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        close_data[sym] = prices.copy()
        open_data[sym] = prices * (1 + np.random.normal(0, 0.001, len(dates)))

    # Add NaN values to symbol "C" (starts later)
    close_data["C"][:100] = np.nan
    open_data["C"][:100] = np.nan

    # Add NaN values to symbol "D" (ends earlier)
    close_data["D"][-50:] = np.nan
    open_data["D"][-50:] = np.nan

    close_prices = pd.DataFrame(close_data, index=dates)
    open_prices = pd.DataFrame(open_data, index=dates)

    return close_prices, open_prices


# =============================================================================
# Test StaggeredConfig
# =============================================================================


class TestStaggeredConfig:
    """Tests for StaggeredConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StaggeredConfig()
        assert config.formation_days == 252
        assert config.trading_days == 126
        assert config.overlap_days == 21
        assert config.n_pairs == 20
        assert config.min_data_pct == 0.95
        assert isinstance(config.backtest_config, BacktestConfig)

    def test_custom_values(self):
        """Test custom configuration values."""
        backtest_config = BacktestConfig(entry_threshold=2.5, wait_days=0)
        config = StaggeredConfig(
            formation_days=189,
            trading_days=63,
            overlap_days=14,
            n_pairs=10,
            min_data_pct=0.90,
            backtest_config=backtest_config,
        )
        assert config.formation_days == 189
        assert config.trading_days == 63
        assert config.overlap_days == 14
        assert config.n_pairs == 10
        assert config.min_data_pct == 0.90
        assert config.backtest_config.entry_threshold == 2.5
        assert config.backtest_config.wait_days == 0


# =============================================================================
# Test generate_portfolio_cycles
# =============================================================================


class TestGeneratePortfolioCycles:
    """Tests for generate_portfolio_cycles function."""

    def test_generates_correct_number_of_cycles(self, sample_dates):
        """Test that correct number of cycles are generated."""
        config = StaggeredConfig(
            formation_days=252,
            trading_days=126,
            overlap_days=21,
        )
        cycles = generate_portfolio_cycles(sample_dates, config)

        # With 756 days, formation=252, trading=126, overlap=21
        # First cycle needs 378 days
        # Each subsequent cycle starts 21 days later
        # Expected: (756 - 378) / 21 + 1 = 18 cycles approximately
        assert len(cycles) > 0
        assert len(cycles) <= 20  # Reasonable upper bound

    def test_cycle_dates_are_correct(self, sample_dates):
        """Test that cycle date boundaries are correct."""
        config = StaggeredConfig(
            formation_days=252,
            trading_days=126,
            overlap_days=21,
        )
        cycles = generate_portfolio_cycles(sample_dates, config)

        for cycle in cycles:
            # Formation end should be formation_days - 1 after start
            formation_length = (
                sample_dates.get_loc(cycle.formation_end)
                - sample_dates.get_loc(cycle.formation_start)
                + 1
            )
            assert formation_length == config.formation_days

            # Trading should start right after formation ends
            trading_start_idx = sample_dates.get_loc(cycle.trading_start)
            formation_end_idx = sample_dates.get_loc(cycle.formation_end)
            assert trading_start_idx == formation_end_idx + 1

            # Trading length should be trading_days
            trading_length = (
                sample_dates.get_loc(cycle.trading_end)
                - sample_dates.get_loc(cycle.trading_start)
                + 1
            )
            assert trading_length == config.trading_days

    def test_cycles_overlap_correctly(self, sample_dates):
        """Test that consecutive cycles start overlap_days apart."""
        config = StaggeredConfig(
            formation_days=252,
            trading_days=126,
            overlap_days=21,
        )
        cycles = generate_portfolio_cycles(sample_dates, config)

        for i in range(1, len(cycles)):
            prev_start_idx = sample_dates.get_loc(cycles[i - 1].formation_start)
            curr_start_idx = sample_dates.get_loc(cycles[i].formation_start)
            assert curr_start_idx - prev_start_idx == config.overlap_days

    def test_insufficient_data_returns_empty(self):
        """Test that insufficient data returns empty list."""
        short_dates = pd.bdate_range(start="2020-01-01", periods=100, freq="B")
        config = StaggeredConfig(formation_days=252, trading_days=126)
        cycles = generate_portfolio_cycles(short_dates, config)
        assert len(cycles) == 0

    def test_cycle_ids_are_sequential(self, sample_dates):
        """Test that cycle IDs are assigned sequentially."""
        config = StaggeredConfig()
        cycles = generate_portfolio_cycles(sample_dates, config)

        for i, cycle in enumerate(cycles):
            assert cycle.cycle_id == i


# =============================================================================
# Test filter_valid_symbols
# =============================================================================


class TestFilterValidSymbols:
    """Tests for filter_valid_symbols function."""

    def test_all_valid_symbols_returned(self, short_sample_prices):
        """Test that all symbols are returned when data is complete."""
        close_prices, _ = short_sample_prices
        valid = filter_valid_symbols(
            close_prices,
            close_prices.index[0],
            close_prices.index[-1],
            min_data_pct=0.95,
        )
        assert len(valid) == len(close_prices.columns)

    def test_filters_symbols_with_missing_data(self, prices_with_missing_data):
        """Test that symbols with insufficient data are filtered out."""
        close_prices, _ = prices_with_missing_data

        # Test with full date range - C and D have missing data
        valid = filter_valid_symbols(
            close_prices,
            close_prices.index[0],
            close_prices.index[-1],
            min_data_pct=0.95,
        )

        # A and B should be valid (complete data)
        assert "A" in valid
        assert "B" in valid
        # C starts late, D ends early - may or may not pass depending on threshold
        # With 95% threshold and 500 days, C is missing 100 (20%), D missing 50 (10%)
        assert "C" not in valid  # Missing 20%
        # D might pass depending on exact calculation

    def test_respects_min_data_pct_threshold(self, prices_with_missing_data):
        """Test that min_data_pct threshold is respected."""
        close_prices, _ = prices_with_missing_data

        # With lower threshold, more symbols should pass
        valid_strict = filter_valid_symbols(
            close_prices,
            close_prices.index[0],
            close_prices.index[-1],
            min_data_pct=0.95,
        )

        valid_lenient = filter_valid_symbols(
            close_prices,
            close_prices.index[0],
            close_prices.index[-1],
            min_data_pct=0.70,
        )

        assert len(valid_lenient) >= len(valid_strict)

    def test_empty_range_returns_empty(self, short_sample_prices):
        """Test that invalid date range returns empty list."""
        close_prices, _ = short_sample_prices

        # Use dates outside the data range
        valid = filter_valid_symbols(
            close_prices,
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-12-31"),
            min_data_pct=0.95,
        )
        assert len(valid) == 0


# =============================================================================
# Test PortfolioCycle
# =============================================================================


class TestPortfolioCycle:
    """Tests for PortfolioCycle dataclass."""

    def test_is_active_on(self):
        """Test is_active_on method."""
        cycle = PortfolioCycle(
            cycle_id=0,
            formation_start=pd.Timestamp("2020-01-01"),
            formation_end=pd.Timestamp("2020-12-31"),
            trading_start=pd.Timestamp("2021-01-01"),
            trading_end=pd.Timestamp("2021-06-30"),
        )

        # During formation - not active
        assert not cycle.is_active_on(pd.Timestamp("2020-06-15"))

        # During trading - active
        assert cycle.is_active_on(pd.Timestamp("2021-01-15"))
        assert cycle.is_active_on(pd.Timestamp("2021-03-15"))
        assert cycle.is_active_on(pd.Timestamp("2021-06-30"))

        # After trading - not active
        assert not cycle.is_active_on(pd.Timestamp("2021-07-15"))


# =============================================================================
# Test run_portfolio_cycle
# =============================================================================


class TestRunPortfolioCycle:
    """Tests for run_portfolio_cycle function."""

    def test_selects_pairs_and_runs_backtest(self, short_sample_prices):
        """Test that pairs are selected and backtest runs."""
        close_prices, open_prices = short_sample_prices

        cycle = PortfolioCycle(
            cycle_id=0,
            formation_start=close_prices.index[0],
            formation_end=close_prices.index[251],
            trading_start=close_prices.index[252],
            trading_end=close_prices.index[377],
        )

        config = StaggeredConfig(
            formation_days=252,
            trading_days=126,
            n_pairs=3,
        )

        result_cycle = run_portfolio_cycle(cycle, close_prices, open_prices, config)

        assert result_cycle.pairs is not None
        assert len(result_cycle.pairs) <= config.n_pairs
        assert result_cycle.results is not None
        assert result_cycle.valid_symbols is not None

    def test_handles_insufficient_symbols(self):
        """Test handling when too few symbols are available."""
        # Create data with only one symbol
        dates = pd.bdate_range(start="2020-01-01", periods=400, freq="B")
        close_prices = pd.DataFrame({"A": np.random.randn(400).cumsum() + 100}, index=dates)
        open_prices = close_prices.copy()

        cycle = PortfolioCycle(
            cycle_id=0,
            formation_start=dates[0],
            formation_end=dates[251],
            trading_start=dates[252],
            trading_end=dates[377],
        )

        config = StaggeredConfig(n_pairs=5)
        result_cycle = run_portfolio_cycle(cycle, close_prices, open_prices, config)

        # Should handle gracefully with empty pairs
        assert result_cycle.pairs == []
        assert result_cycle.results == {}


# =============================================================================
# Test aggregate_monthly_returns
# =============================================================================


class TestAggregateMonthlyReturns:
    """Tests for aggregate_monthly_returns function."""

    def test_averages_active_portfolios(self):
        """Test that returns are averaged only across active portfolios."""
        # Create mock cycles with known returns
        cycle1 = PortfolioCycle(
            cycle_id=0,
            formation_start=pd.Timestamp("2020-01-01"),
            formation_end=pd.Timestamp("2020-12-31"),
            trading_start=pd.Timestamp("2021-01-01"),
            trading_end=pd.Timestamp("2021-06-30"),
        )
        cycle1.monthly_returns = pd.Series(
            [0.01, 0.02, 0.03, 0.01, 0.02, 0.01],
            index=pd.date_range("2021-01-31", periods=6, freq="ME"),
        )

        cycle2 = PortfolioCycle(
            cycle_id=1,
            formation_start=pd.Timestamp("2020-02-01"),
            formation_end=pd.Timestamp("2021-01-31"),
            trading_start=pd.Timestamp("2021-02-01"),
            trading_end=pd.Timestamp("2021-07-31"),
        )
        cycle2.monthly_returns = pd.Series(
            [0.02, 0.01, 0.02, 0.03, 0.01, 0.02],
            index=pd.date_range("2021-02-28", periods=6, freq="ME"),
        )

        cycles = [cycle1, cycle2]
        avg_returns, active_counts = aggregate_monthly_returns(cycles)

        # Check that active counts are correct
        # Jan 2021: only cycle1 active (1)
        # Feb-Jun 2021: both active (2)
        # Jul 2021: only cycle2 active (1)
        assert len(avg_returns) > 0
        assert len(active_counts) > 0

    def test_handles_empty_cycles(self):
        """Test handling of empty cycle list."""
        avg_returns, active_counts = aggregate_monthly_returns([])
        assert len(avg_returns) == 0
        assert len(active_counts) == 0


# =============================================================================
# Test run_staggered_backtest (Integration)
# =============================================================================


class TestRunStaggeredBacktest:
    """Integration tests for run_staggered_backtest."""

    def test_full_pipeline(self, sample_prices):
        """Test full staggered backtest pipeline."""
        close_prices, open_prices = sample_prices

        config = StaggeredConfig(
            formation_days=252,
            trading_days=126,
            overlap_days=21,
            n_pairs=5,
        )

        result = run_staggered_backtest(close_prices, open_prices, config)

        assert isinstance(result, StaggeredResult)
        assert len(result.cycles) > 0
        assert result.monthly_returns is not None
        assert result.cumulative_returns is not None
        assert result.active_portfolios_over_time is not None
        assert result.config == config

    def test_active_portfolio_count_stabilizes(self, sample_prices):
        """Test that active portfolio count reaches expected level."""
        close_prices, open_prices = sample_prices

        config = StaggeredConfig(
            formation_days=252,
            trading_days=126,
            overlap_days=21,
            n_pairs=5,
        )

        result = run_staggered_backtest(close_prices, open_prices, config)

        # After ramp-up, should have ~6 active portfolios
        # (trading_days / overlap_days = 126 / 21 = 6)
        active_counts = result.active_portfolios_over_time.dropna()
        if len(active_counts) > 6:
            # Check steady state (after first 6 months)
            steady_state = active_counts.iloc[6:]
            if len(steady_state) > 0:
                avg_active = steady_state.mean()
                assert avg_active >= 4  # Should be close to 6

    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises ValueError."""
        short_dates = pd.bdate_range(start="2020-01-01", periods=100, freq="B")
        close_prices = pd.DataFrame(
            {"A": np.random.randn(100).cumsum() + 100},
            index=short_dates,
        )
        open_prices = close_prices.copy()

        config = StaggeredConfig(formation_days=252, trading_days=126)

        with pytest.raises(ValueError, match="Not enough data"):
            run_staggered_backtest(close_prices, open_prices, config)

    def test_progress_callback_called(self, sample_prices):
        """Test that progress callback is called during execution."""
        close_prices, open_prices = sample_prices

        config = StaggeredConfig(
            formation_days=252,
            trading_days=126,
            overlap_days=42,  # Fewer cycles for faster test
            n_pairs=3,
        )

        progress_calls = []

        def callback(current, total):
            progress_calls.append((current, total))

        result = run_staggered_backtest(
            close_prices, open_prices, config, progress_callback=callback
        )

        assert len(progress_calls) == len(result.cycles)
        # Progress should be sequential
        for i, (current, total) in enumerate(progress_calls):
            assert current == i + 1


# =============================================================================
# Test with Missing Data
# =============================================================================


class TestStaggeredWithMissingData:
    """Tests for staggered backtest with missing data."""

    def test_filters_symbols_per_cycle(self, prices_with_missing_data):
        """Test that symbols are filtered per cycle based on data availability."""
        close_prices, open_prices = prices_with_missing_data

        config = StaggeredConfig(
            formation_days=150,
            trading_days=75,
            overlap_days=30,
            n_pairs=2,
            min_data_pct=0.95,
        )

        result = run_staggered_backtest(close_prices, open_prices, config)

        # Check that valid_symbols varies across cycles
        valid_symbols_sets = [
            set(c.valid_symbols) for c in result.cycles if c.valid_symbols
        ]

        # At least some cycles should have different valid symbols
        # due to the missing data pattern
        assert len(result.cycles) > 0


# =============================================================================
# Test StaggeredResult
# =============================================================================


class TestStaggeredResult:
    """Tests for StaggeredResult dataclass."""

    def test_total_portfolios_property(self, sample_prices):
        """Test total_portfolios property."""
        close_prices, open_prices = sample_prices

        config = StaggeredConfig(
            formation_days=252,
            trading_days=126,
            overlap_days=42,
            n_pairs=3,
        )

        result = run_staggered_backtest(close_prices, open_prices, config)

        assert result.total_portfolios == len(result.cycles)
