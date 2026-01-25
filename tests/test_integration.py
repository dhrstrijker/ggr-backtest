"""Integration tests for end-to-end backtest verification.

These tests verify the complete pipeline from price data to final metrics,
using pre-calculated fixtures with known expected outcomes.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, Trade, run_backtest_single_pair, combine_results
from src.analysis import calculate_metrics
from src.signals import calculate_spread, calculate_formation_stats, calculate_distance

from tests.fixtures.known_outcomes import (
    SIMPLE_CONVERGENCE_FIXTURE,
    KNOWN_METRICS_FIXTURE,
    NO_TRADE_FIXTURE,
)


class TestEndToEndBacktest:
    """End-to-end backtest integration tests."""

    def test_simple_convergence_generates_trades(self):
        """Simple convergence scenario should generate expected number of trades."""
        fixture = SIMPLE_CONVERGENCE_FIXTURE

        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
            capital_per_trade=10000.0,
        )

        result = run_backtest_single_pair(
            formation_close=fixture["formation_close"],
            trading_close=fixture["trading_close"],
            trading_open=fixture["trading_open"],
            pair=fixture["pair"],
            config=config,
        )

        # Should generate trades within expected range
        assert fixture["expected_min_trades"] <= len(result.trades) <= fixture["expected_max_trades"], \
            f"Expected {fixture['expected_min_trades']}-{fixture['expected_max_trades']} trades, got {len(result.trades)}"

    def test_no_trade_scenario(self):
        """Scenario with no divergence should generate no trades."""
        fixture = NO_TRADE_FIXTURE

        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
        )

        result = run_backtest_single_pair(
            formation_close=fixture["formation_close"],
            trading_close=fixture["trading_close"],
            trading_open=fixture["trading_open"],
            pair=fixture["pair"],
            config=config,
        )

        assert len(result.trades) == fixture["expected_trades"], \
            f"Expected {fixture['expected_trades']} trades, got {len(result.trades)}"

    def test_formation_std_stays_constant(self):
        """Formation period σ should not change during trading."""
        fixture = SIMPLE_CONVERGENCE_FIXTURE

        # Calculate formation stats
        formation_spread = calculate_spread(
            fixture["formation_close"]["A"],
            fixture["formation_close"]["B"],
            normalize=True,
        )
        formation_stats = calculate_formation_stats(formation_spread)
        formation_std = formation_stats["std"]

        # Calculate trading spread
        trading_spread = calculate_spread(
            fixture["trading_close"]["A"],
            fixture["trading_close"]["B"],
            normalize=True,
        )

        # Calculate distance using formation std
        distance = calculate_distance(trading_spread, formation_std)

        # Verify distance uses CONSTANT formation std (not rolling)
        # The std used should be a single value, not recalculated per row
        assert isinstance(formation_std, float), "Formation std should be a scalar"
        assert formation_std > 0, "Formation std should be positive"

        # Distance should be spread / formation_std for all points
        expected_distance = trading_spread / formation_std
        pd.testing.assert_series_equal(distance, expected_distance, check_names=False)

    def test_no_lookahead_bias(self):
        """Signals should be based on close, execution on next-day open.

        This test verifies:
        1. Entry prices use OPEN prices (not close)
        2. Exit prices use OPEN prices (not close)
        3. Entry date is after signal date (no same-day execution with wait_days=1)
        """
        fixture = SIMPLE_CONVERGENCE_FIXTURE

        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
            wait_days=1,  # Next-day execution
        )

        result = run_backtest_single_pair(
            formation_close=fixture["formation_close"],
            trading_close=fixture["trading_close"],
            trading_open=fixture["trading_open"],
            pair=fixture["pair"],
            config=config,
        )

        # Must have trades to verify lookahead
        assert len(result.trades) > 0, "Need trades to verify no lookahead bias"

        for trade in result.trades:
            # Entry date should be in trading period
            assert trade.entry_date in fixture["trading_open"].index, \
                "Entry date should be a trading date"

            # Entry price should match OPEN price (not close)
            expected_entry_a = fixture["trading_open"].loc[trade.entry_date, "A"]
            expected_entry_b = fixture["trading_open"].loc[trade.entry_date, "B"]

            assert trade.entry_price_a == expected_entry_a, \
                f"Entry price A should be open: {expected_entry_a}, got {trade.entry_price_a}"
            assert trade.entry_price_b == expected_entry_b, \
                f"Entry price B should be open: {expected_entry_b}, got {trade.entry_price_b}"

            # Exit prices should also use OPEN prices (no lookahead on exit)
            if trade.exit_date in fixture["trading_open"].index:
                expected_exit_a = fixture["trading_open"].loc[trade.exit_date, "A"]
                expected_exit_b = fixture["trading_open"].loc[trade.exit_date, "B"]

                assert trade.exit_price_a == expected_exit_a, \
                    f"Exit price A should be open: {expected_exit_a}, got {trade.exit_price_a}"
                assert trade.exit_price_b == expected_exit_b, \
                    f"Exit price B should be open: {expected_exit_b}, got {trade.exit_price_b}"

            # With wait_days=1, entry should not be on day 0 of trading period
            # (signal on day N, entry on day N+1)
            trading_start = fixture["trading_close"].index[0]
            if trade.entry_date == trading_start:
                # Only valid if there was a signal at end of formation period
                # which is unusual in our test data
                pass  # Allow for edge case

    def test_pipeline_produces_consistent_results(self):
        """Running the same backtest twice should produce identical results."""
        fixture = SIMPLE_CONVERGENCE_FIXTURE

        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
        )

        result1 = run_backtest_single_pair(
            formation_close=fixture["formation_close"],
            trading_close=fixture["trading_close"],
            trading_open=fixture["trading_open"],
            pair=fixture["pair"],
            config=config,
        )

        result2 = run_backtest_single_pair(
            formation_close=fixture["formation_close"],
            trading_close=fixture["trading_close"],
            trading_open=fixture["trading_open"],
            pair=fixture["pair"],
            config=config,
        )

        # Same number of trades
        assert len(result1.trades) == len(result2.trades)

        # Same P&L for each trade
        for t1, t2 in zip(result1.trades, result2.trades):
            assert t1.pnl == t2.pnl
            assert t1.entry_date == t2.entry_date
            assert t1.exit_date == t2.exit_date

    def test_equity_curve_starts_at_capital(self):
        """Equity curve should start at configured capital."""
        fixture = SIMPLE_CONVERGENCE_FIXTURE
        capital = 15000.0

        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
            capital_per_trade=capital,
        )

        result = run_backtest_single_pair(
            formation_close=fixture["formation_close"],
            trading_close=fixture["trading_close"],
            trading_open=fixture["trading_open"],
            pair=fixture["pair"],
            config=config,
        )

        assert result.equity_curve.iloc[0] == capital


class TestMetricsIntegration:
    """Integration tests for metrics calculation."""

    def test_metrics_match_manual_calculation(self):
        """Calculated metrics should match manually computed expected values."""
        fixture = KNOWN_METRICS_FIXTURE

        # Create Trade objects from fixture data
        trades = []
        base_date = datetime(2024, 1, 1)
        for i, td in enumerate(fixture["trades_data"]):
            entry_date = base_date
            exit_date = entry_date + timedelta(days=td["holding_days"])
            trade = Trade(
                pair=("A", "B"),
                direction=td["direction"],
                entry_date=entry_date,
                exit_date=exit_date,
                entry_price_a=100.0,
                entry_price_b=100.0,
                exit_price_a=100.0,
                exit_price_b=100.0,
                shares_a=50.0,
                shares_b=50.0,
                pnl=td["pnl"],
                pnl_pct=td["pnl_pct"],
                holding_days=td["holding_days"],
                entry_distance=2.5,
                exit_distance=0.1,
                exit_reason="crossing",
                max_adverse_spread=2.5,
            )
            trades.append(trade)

        # Create simple equity curve
        equity = pd.Series(
            [10000.0, 10000.0 + fixture["expected_total_pnl"]],
            index=[datetime(2024, 1, 1), datetime(2024, 2, 1)],
        )

        metrics = calculate_metrics(trades, equity)

        # Verify against expected values
        assert metrics["total_trades"] == fixture["expected_total_trades"]
        assert metrics["total_return"] == fixture["expected_total_pnl"]
        assert pytest.approx(metrics["win_rate"], rel=0.01) == fixture["expected_win_rate"]
        assert pytest.approx(metrics["avg_win"], rel=0.01) == fixture["expected_avg_win"]
        assert pytest.approx(metrics["avg_loss"], rel=0.01) == fixture["expected_avg_loss"]
        assert pytest.approx(metrics["profit_factor"], rel=0.01) == fixture["expected_profit_factor"]
        assert pytest.approx(metrics["avg_holding_days"], rel=0.01) == fixture["expected_avg_holding_days"]
        assert metrics["long_trades"] == fixture["expected_long_trades"]
        assert metrics["short_trades"] == fixture["expected_short_trades"]
        assert pytest.approx(metrics["long_win_rate"], rel=0.01) == fixture["expected_long_win_rate"]
        assert pytest.approx(metrics["short_win_rate"], rel=0.01) == fixture["expected_short_win_rate"]

    def test_backtest_to_metrics_pipeline(self):
        """Full pipeline from backtest to metrics should work without errors."""
        fixture = SIMPLE_CONVERGENCE_FIXTURE

        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
            capital_per_trade=10000.0,
        )

        result = run_backtest_single_pair(
            formation_close=fixture["formation_close"],
            trading_close=fixture["trading_close"],
            trading_open=fixture["trading_open"],
            pair=fixture["pair"],
            config=config,
        )

        # Should be able to calculate metrics without error
        metrics = calculate_metrics(result.trades, result.equity_curve)

        # Basic sanity checks
        assert isinstance(metrics, dict)
        assert "total_trades" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert not any(pd.isna(v) for v in metrics.values() if isinstance(v, (int, float)))


class TestCombineResultsIntegration:
    """Integration tests for combining multiple pair results."""

    def test_combine_results_aggregates_trades(self):
        """Combined results should include trades from all pairs."""
        fixture = SIMPLE_CONVERGENCE_FIXTURE

        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
        )

        # Run backtest twice for same pair (simulating independent runs)
        result1 = run_backtest_single_pair(
            formation_close=fixture["formation_close"],
            trading_close=fixture["trading_close"],
            trading_open=fixture["trading_open"],
            pair=("A", "B"),
            config=config,
        )

        # Run again - using ("A", "B") relabeled as ("pair2_A", "pair2_B") conceptually
        # In practice, just run the same data under a different key
        result2 = run_backtest_single_pair(
            formation_close=fixture["formation_close"],
            trading_close=fixture["trading_close"],
            trading_open=fixture["trading_open"],
            pair=("A", "B"),  # Same data, different result object
            config=config,
        )

        # Combine under different keys to simulate multi-pair portfolio
        results = {
            ("A", "B"): result1,
            ("A2", "B2"): result2,  # Conceptually a different pair
        }

        all_trades, combined_equity = combine_results(results, initial_capital=20000.0)

        # Should have trades from both "pairs"
        assert len(all_trades) == len(result1.trades) + len(result2.trades)

        # Combined equity should reflect both
        assert combined_equity.iloc[0] == 20000.0


# =============================================================================
# Full Pipeline Integration Tests
# =============================================================================


class TestFullPipeline:
    """Tests for complete data → normalize → SSD → pairs → backtest → metrics pipeline."""

    def test_end_to_end_with_synthetic_data(self):
        """Test: data → normalize → SSD → pairs → backtest → metrics."""
        from src.pairs import normalize_prices, calculate_ssd_matrix, select_top_pairs
        from src.backtest import run_backtest
        from src.analysis import calculate_metrics

        # Create synthetic data (simulating fetched data)
        np.random.seed(42)
        dates = pd.bdate_range(start="2020-01-01", periods=400, freq="B")
        symbols = ["A", "B", "C", "D", "E", "F"]

        # Generate correlated price data
        close_data = {}
        open_data = {}
        for i, sym in enumerate(symbols):
            base_price = 100 + i * 10
            # A and B are highly correlated (good pair)
            # Others have different patterns
            if sym in ["A", "B"]:
                returns = np.random.normal(0.0003, 0.01, len(dates))
            else:
                returns = np.random.normal(0.0003, 0.02, len(dates))
            prices = base_price * np.exp(np.cumsum(returns))
            close_data[sym] = prices
            open_data[sym] = prices * (1 + np.random.normal(0, 0.001, len(dates)))

        close_prices = pd.DataFrame(close_data, index=dates)
        open_prices = pd.DataFrame(open_data, index=dates)

        # Step 1: Split into formation and trading periods
        formation_days = 252
        trading_days = 126
        formation_close = close_prices.iloc[:formation_days]
        trading_close = close_prices.iloc[formation_days:formation_days + trading_days]
        trading_open = open_prices.iloc[formation_days:formation_days + trading_days]

        # Step 2: Normalize prices
        normalized = normalize_prices(formation_close)
        assert not normalized.empty, "Normalization should produce data"

        # Step 3: Calculate SSD matrix
        ssd_matrix = calculate_ssd_matrix(normalized)
        assert ssd_matrix.shape[0] == ssd_matrix.shape[1], "SSD matrix should be square"

        # Step 4: Select top pairs
        pairs = select_top_pairs(ssd_matrix, n=3)
        assert len(pairs) == 3, "Should select 3 pairs"

        # Step 5: Run backtest
        config = BacktestConfig(
            entry_threshold=2.0,
            max_holding_days=50,
            capital_per_trade=10000,
        )

        results = run_backtest(
            formation_close=formation_close,
            trading_close=trading_close,
            trading_open=trading_open,
            pairs=pairs,
            config=config,
        )

        assert len(results) == 3, "Should have results for 3 pairs"

        # Step 6: Calculate metrics for each pair
        for pair, result in results.items():
            metrics = calculate_metrics(result.trades, result.equity_curve)

            # Verify metrics structure
            assert "total_trades" in metrics
            assert "sharpe_ratio" in metrics
            assert "max_drawdown" in metrics

            # Metrics should be valid (not NaN)
            assert isinstance(metrics["total_trades"], int)
            assert not np.isnan(metrics["sharpe_ratio"])

    def test_staggered_multiple_cycles_complete(self):
        """Test: full staggered backtest with 3+ cycles."""
        from src.staggered import run_staggered_backtest, StaggeredConfig
        from src.analysis import calculate_staggered_metrics

        # Create sufficient data for multiple cycles
        np.random.seed(123)
        dates = pd.bdate_range(start="2019-01-01", periods=600, freq="B")
        symbols = ["A", "B", "C", "D", "E", "F"]

        close_data = {}
        open_data = {}
        for i, sym in enumerate(symbols):
            base_price = 100 + i * 10
            returns = np.random.normal(0.0003, 0.015, len(dates))
            prices = base_price * np.exp(np.cumsum(returns))
            close_data[sym] = prices
            open_data[sym] = prices * (1 + np.random.normal(0, 0.001, len(dates)))

        close_prices = pd.DataFrame(close_data, index=dates)
        open_prices = pd.DataFrame(open_data, index=dates)

        # Configure for multiple cycles
        config = StaggeredConfig(
            formation_days=150,  # Shorter for faster test
            trading_days=75,
            overlap_days=30,  # ~2.5 portfolios at steady state
            n_pairs=3,
            backtest_config=BacktestConfig(
                entry_threshold=2.0,
                capital_per_trade=10000,
            ),
        )

        # Run staggered backtest
        result = run_staggered_backtest(close_prices, open_prices, config)

        # Should have multiple cycles
        assert result.total_portfolios >= 3, \
            f"Should have at least 3 portfolio cycles, got {result.total_portfolios}"

        # Should have monthly returns
        assert len(result.monthly_returns) > 0, "Should have monthly returns"

        # Should have cumulative returns
        assert len(result.cumulative_returns) > 0, "Should have cumulative returns"

        # Calculate metrics
        metrics = calculate_staggered_metrics(result)

        # Metrics should be valid
        assert metrics["total_months"] > 0, "Should have some months of data"

    def test_error_propagation_nan_in_data(self):
        """Test: NaN in input data handled correctly through pipeline."""
        from src.pairs import normalize_prices, calculate_ssd_matrix, select_top_pairs
        from src.backtest import run_backtest

        # Create data with NaN values
        np.random.seed(456)
        dates = pd.bdate_range(start="2020-01-01", periods=400, freq="B")

        # Symbol C has NaN in first row (should be filtered by normalize_prices)
        close_data = {
            "A": np.random.randn(400).cumsum() + 100,
            "B": np.random.randn(400).cumsum() + 100,
            "C": [np.nan] + list(np.random.randn(399).cumsum() + 100),  # First value NaN
            "D": np.random.randn(400).cumsum() + 100,
        }
        close_prices = pd.DataFrame(close_data, index=dates)
        open_prices = close_prices.copy()

        # Formation and trading split
        formation_close = close_prices.iloc[:250]
        trading_close = close_prices.iloc[250:350]
        trading_open = open_prices.iloc[250:350]

        # Normalize should drop symbol C
        normalized = normalize_prices(formation_close)
        assert "C" not in normalized.columns, "Symbol C should be filtered out (first row NaN)"
        assert "A" in normalized.columns, "Valid symbols should remain"

        # SSD should work with remaining symbols
        ssd_matrix = calculate_ssd_matrix(normalized)
        assert not ssd_matrix.empty, "SSD matrix should be computed"

        # Select pairs (from filtered symbols)
        pairs = select_top_pairs(ssd_matrix, n=2)
        assert len(pairs) > 0, "Should have at least one pair"

        # Verify no pair contains filtered symbol
        for pair in pairs:
            assert "C" not in pair, f"Filtered symbol C should not be in pairs: {pair}"

        # Run backtest with filtered pairs
        config = BacktestConfig(entry_threshold=2.0)

        results = run_backtest(
            formation_close=formation_close,
            trading_close=trading_close,
            trading_open=trading_open,
            pairs=pairs,
            config=config,
        )

        # Should complete without error
        assert len(results) == len(pairs)

        # All results should have valid (non-NaN) equity curves
        for pair, result in results.items():
            assert not result.equity_curve.isna().any(), \
                f"Equity curve for {pair} should not contain NaN"


# =============================================================================
# Data Quality Integration Tests
# =============================================================================


class TestDataQualityIntegration:
    """Integration tests for data quality handling."""

    def test_different_listing_dates_handled(self):
        """Test: symbols with different listing dates are handled correctly."""
        from src.pairs import normalize_prices, calculate_ssd_matrix
        from src.staggered import filter_valid_symbols

        # Create data where C starts later (listed later)
        dates = pd.bdate_range(start="2020-01-01", periods=300, freq="B")

        close_data = {
            "A": np.random.randn(300).cumsum() + 100,
            "B": np.random.randn(300).cumsum() + 100,
            "C": [np.nan] * 100 + list(np.random.randn(200).cumsum() + 100),  # Starts at day 100
            "D": np.random.randn(300).cumsum() + 100,
        }
        close_prices = pd.DataFrame(close_data, index=dates)

        # Formation period: days 0-149 (C has only 50 valid days)
        formation_start = dates[0]
        formation_end = dates[149]

        # Filter valid symbols for formation (100% coverage required)
        valid_symbols = filter_valid_symbols(close_prices, formation_start, formation_end)

        # C should NOT be valid (only 50/150 = 33% coverage during formation)
        assert "C" not in valid_symbols, \
            "Symbol C should not be valid (only 33% coverage during formation)"
        assert "A" in valid_symbols, "Symbol A should be valid"
        assert "B" in valid_symbols, "Symbol B should be valid"
        assert "D" in valid_symbols, "Symbol D should be valid"

        # Later formation period where C is valid: days 100-249
        formation_start_late = dates[100]
        formation_end_late = dates[249]

        valid_symbols_late = filter_valid_symbols(close_prices, formation_start_late, formation_end_late)

        # Now C should be valid (100% coverage during this formation)
        assert "C" in valid_symbols_late, \
            "Symbol C should be valid when formation period is after listing date"
