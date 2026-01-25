"""Tests for data integrity and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import timedelta

from src.data import validate_price_data, find_data_gaps, get_trading_days_coverage


class TestDataIntegrity:
    """Test suite for data integrity validation."""

    def test_detect_missing_dates(self):
        """Should detect gaps in trading data."""
        # Create data with a gap (missing Wed, Thu, Fri)
        dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-08', '2024-01-09'])
        prices = pd.DataFrame({
            'A': [100.0, 101.0, 105.0, 106.0],
            'B': [50.0, 51.0, 53.0, 54.0],
        }, index=dates)

        gaps = find_data_gaps(prices, max_gap_days=3)

        assert len(gaps) > 0, "Should detect the gap in data"
        assert gaps[0]['gap_days'] > 3, "Gap should be more than 3 days"

    def test_no_false_positive_weekends(self):
        """Should not flag weekends as gaps."""
        # Friday to Monday is normal (3 days gap)
        dates = pd.to_datetime(['2024-01-05', '2024-01-08'])  # Fri to Mon
        prices = pd.DataFrame({
            'A': [100.0, 101.0],
        }, index=dates)

        gaps = find_data_gaps(prices, max_gap_days=3)

        assert len(gaps) == 0, "Weekend gaps should not be flagged"

    def test_detect_nan_values(self):
        """Should detect NaN values in price data."""
        dates = pd.date_range('2024-01-01', periods=5)
        prices = pd.DataFrame({
            'A': [100.0, 101.0, np.nan, 103.0, 104.0],
            'B': [50.0, 51.0, 52.0, 53.0, 54.0],
        }, index=dates)

        issues = validate_price_data(prices)

        assert any('nan' in str(i).lower() for i in issues), \
            "Should detect NaN values"

    def test_detect_zero_prices(self):
        """Should detect zero or negative prices."""
        dates = pd.date_range('2024-01-01', periods=5)
        prices = pd.DataFrame({
            'A': [100.0, 101.0, 0.0, 103.0, 104.0],
            'B': [50.0, -1.0, 52.0, 53.0, 54.0],
        }, index=dates)

        issues = validate_price_data(prices)

        assert any('zero' in str(i).lower() or 'negative' in str(i).lower() for i in issues), \
            "Should detect zero or negative prices"

    def test_detect_extreme_price_jumps(self):
        """Should detect suspiciously large price changes."""
        dates = pd.date_range('2024-01-01', periods=5)
        prices = pd.DataFrame({
            'A': [100.0, 101.0, 500.0, 103.0, 104.0],  # 5x jump then drop
        }, index=dates)

        issues = validate_price_data(prices, max_daily_change=0.5)

        assert any('jump' in str(i).lower() or 'change' in str(i).lower() for i in issues), \
            "Should detect extreme price jumps"

    def test_valid_data_passes(self):
        """Valid data should pass all checks."""
        dates = pd.date_range('2024-01-01', periods=10, freq='B')  # Business days
        prices = pd.DataFrame({
            'A': [100.0 + i * 0.5 for i in range(10)],
            'B': [50.0 + i * 0.3 for i in range(10)],
        }, index=dates)

        issues = validate_price_data(prices)

        assert len(issues) == 0, f"Valid data should have no issues, got: {issues}"

    def test_trading_days_coverage(self):
        """Should calculate coverage percentage correctly."""
        # 10 days requested, 8 days of data = 80% coverage
        dates = pd.date_range('2024-01-01', periods=8, freq='B')
        prices = pd.DataFrame({
            'A': [100.0] * 8,
        }, index=dates)

        coverage = get_trading_days_coverage(
            prices,
            start_date='2024-01-01',
            end_date='2024-01-12'
        )

        assert coverage['total_trading_days'] > 0
        assert coverage['actual_days'] == 8
        assert coverage['coverage_pct'] < 1.0, "Coverage should be less than 100%"

    def test_symbols_aligned(self):
        """All symbols should have data for the same dates after dropna."""
        dates = pd.date_range('2024-01-01', periods=5)
        prices = pd.DataFrame({
            'A': [100.0, 101.0, 102.0, 103.0, 104.0],
            'B': [50.0, np.nan, 52.0, 53.0, 54.0],  # B missing one day
        }, index=dates)

        # After dropna, all symbols should be aligned
        aligned = prices.dropna()

        assert len(aligned) == 4, "Should have 4 rows after dropping NaN"
        assert aligned.notna().all().all(), "All values should be valid after dropna"


class TestDataGapDetection:
    """Detailed tests for gap detection logic."""

    def test_single_day_gap(self):
        """Should detect single trading day gaps."""
        # Missing Tuesday
        dates = pd.to_datetime(['2024-01-01', '2024-01-03'])  # Mon, Wed
        prices = pd.DataFrame({'A': [100.0, 101.0]}, index=dates)

        gaps = find_data_gaps(prices, max_gap_days=1)

        assert len(gaps) == 1, "Should detect 1-day gap"

    def test_holiday_gap_acceptable(self):
        """Typical holiday gaps (1-2 extra days) should be acceptable."""
        # 4-day gap (typical for Thanksgiving week)
        dates = pd.to_datetime(['2024-01-01', '2024-01-05'])
        prices = pd.DataFrame({'A': [100.0, 101.0]}, index=dates)

        gaps = find_data_gaps(prices, max_gap_days=5)

        assert len(gaps) == 0, "Holiday gaps should be acceptable with higher threshold"

    def test_multi_week_gap_detected(self):
        """Should always detect multi-week gaps."""
        dates = pd.to_datetime(['2024-01-01', '2024-01-22'])  # 3 week gap
        prices = pd.DataFrame({'A': [100.0, 101.0]}, index=dates)

        gaps = find_data_gaps(prices, max_gap_days=5)

        assert len(gaps) == 1, "Should detect multi-week gap"
        assert gaps[0]['gap_days'] >= 14, "Gap should be at least 2 weeks"
