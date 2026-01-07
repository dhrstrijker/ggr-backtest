"""Tests for src/data.py - Data fetching and price extraction functions."""

import pandas as pd
import numpy as np
import pytest

from src.data import get_close_prices, get_open_prices


def create_multi_index_prices():
    """Create a MultiIndex DataFrame similar to fetch_prices output."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")

    # Create price data for two symbols
    data = {
        ("AAPL", "open"): [150.0 + i * 0.5 for i in range(10)],
        ("AAPL", "high"): [152.0 + i * 0.5 for i in range(10)],
        ("AAPL", "low"): [148.0 + i * 0.5 for i in range(10)],
        ("AAPL", "close"): [151.0 + i * 0.5 for i in range(10)],
        ("AAPL", "volume"): [1000000 + i * 10000 for i in range(10)],
        ("MSFT", "open"): [300.0 + i * 1.0 for i in range(10)],
        ("MSFT", "high"): [305.0 + i * 1.0 for i in range(10)],
        ("MSFT", "low"): [295.0 + i * 1.0 for i in range(10)],
        ("MSFT", "close"): [302.0 + i * 1.0 for i in range(10)],
        ("MSFT", "volume"): [2000000 + i * 20000 for i in range(10)],
    }

    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


class TestGetClosePrices:
    """Tests for get_close_prices function."""

    def test_extracts_close_prices(self):
        """Should extract close prices from MultiIndex DataFrame."""
        prices_df = create_multi_index_prices()

        close_prices = get_close_prices(prices_df)

        assert isinstance(close_prices, pd.DataFrame)
        assert "AAPL" in close_prices.columns
        assert "MSFT" in close_prices.columns

    def test_close_prices_correct_values(self):
        """Extracted close prices should match original values."""
        prices_df = create_multi_index_prices()

        close_prices = get_close_prices(prices_df)

        # Check first value matches
        assert close_prices["AAPL"].iloc[0] == 151.0
        assert close_prices["MSFT"].iloc[0] == 302.0

        # Check last value matches
        assert close_prices["AAPL"].iloc[-1] == 151.0 + 9 * 0.5
        assert close_prices["MSFT"].iloc[-1] == 302.0 + 9 * 1.0

    def test_preserves_index(self):
        """Should preserve the date index."""
        prices_df = create_multi_index_prices()

        close_prices = get_close_prices(prices_df)

        pd.testing.assert_index_equal(close_prices.index, prices_df.index)

    def test_returns_flat_columns(self):
        """Returned DataFrame should have flat column names (not MultiIndex)."""
        prices_df = create_multi_index_prices()

        close_prices = get_close_prices(prices_df)

        assert not isinstance(close_prices.columns, pd.MultiIndex)


class TestGetOpenPrices:
    """Tests for get_open_prices function."""

    def test_extracts_open_prices(self):
        """Should extract open prices from MultiIndex DataFrame."""
        prices_df = create_multi_index_prices()

        open_prices = get_open_prices(prices_df)

        assert isinstance(open_prices, pd.DataFrame)
        assert "AAPL" in open_prices.columns
        assert "MSFT" in open_prices.columns

    def test_open_prices_correct_values(self):
        """Extracted open prices should match original values."""
        prices_df = create_multi_index_prices()

        open_prices = get_open_prices(prices_df)

        # Check first value matches (AAPL open starts at 150.0)
        assert open_prices["AAPL"].iloc[0] == 150.0
        assert open_prices["MSFT"].iloc[0] == 300.0

    def test_open_different_from_close(self):
        """Open prices should be different from close prices."""
        prices_df = create_multi_index_prices()

        open_prices = get_open_prices(prices_df)
        close_prices = get_close_prices(prices_df)

        # Open and close should not be identical
        assert not open_prices["AAPL"].equals(close_prices["AAPL"])

    def test_preserves_index(self):
        """Should preserve the date index."""
        prices_df = create_multi_index_prices()

        open_prices = get_open_prices(prices_df)

        pd.testing.assert_index_equal(open_prices.index, prices_df.index)


class TestPriceExtractionConsistency:
    """Tests for consistency between price extraction functions."""

    def test_same_shape(self):
        """Open and close prices should have same shape."""
        prices_df = create_multi_index_prices()

        open_prices = get_open_prices(prices_df)
        close_prices = get_close_prices(prices_df)

        assert open_prices.shape == close_prices.shape

    def test_same_symbols(self):
        """Open and close should have same symbols."""
        prices_df = create_multi_index_prices()

        open_prices = get_open_prices(prices_df)
        close_prices = get_close_prices(prices_df)

        assert list(open_prices.columns) == list(close_prices.columns)

    def test_no_nan_in_extracted_prices(self):
        """Extracted prices should not have NaN if source has none."""
        prices_df = create_multi_index_prices()

        open_prices = get_open_prices(prices_df)
        close_prices = get_close_prices(prices_df)

        assert not open_prices.isna().any().any()
        assert not close_prices.isna().any().any()
