"""Tests for SSD (Sum of Squared Differences) calculation."""

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(pd.io.common.Path(__file__).parent.parent))

from src.pairs import (
    normalize_prices,
    calculate_ssd,
    calculate_ssd_matrix,
    _calculate_ssd_matrix_loop,
    _calculate_ssd_matrix_vectorized,
    _calculate_ssd_matrix_chunked,
    select_top_pairs,
    get_pair_stats,
)


class TestNormalization:
    """Test suite for price normalization."""

    def test_normalize_starts_at_one(self):
        """Normalized prices should start at 1.0."""
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 120.0],
            'B': [50.0, 55.0, 60.0],
        })
        normalized = normalize_prices(prices)

        assert normalized['A'].iloc[0] == 1.0, "A should start at 1.0"
        assert normalized['B'].iloc[0] == 1.0, "B should start at 1.0"

    def test_normalize_preserves_ratios(self):
        """Normalization should preserve relative changes."""
        prices = pd.DataFrame({
            'A': [100.0, 120.0, 150.0],  # +20%, +50%
        })
        normalized = normalize_prices(prices)

        assert abs(normalized['A'].iloc[1] - 1.2) < 0.001, "Second value should be 1.2"
        assert abs(normalized['A'].iloc[2] - 1.5) < 0.001, "Third value should be 1.5"


class TestSSDCalculation:
    """Test suite for SSD calculation."""

    def test_identical_series_zero_ssd(self):
        """Identical normalized series should have SSD = 0."""
        prices = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        normalized = normalize_prices(prices)
        ssd_matrix = calculate_ssd_matrix(normalized)

        assert ssd_matrix.loc['A', 'B'] == 0.0, \
            f"Identical series should have SSD=0, got {ssd_matrix.loc['A', 'B']}"

    def test_different_series_positive_ssd(self):
        """Different series should have SSD > 0."""
        prices = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [5.0, 4.0, 3.0, 2.0, 1.0],  # Inverse
        })
        normalized = normalize_prices(prices)
        ssd_matrix = calculate_ssd_matrix(normalized)

        assert ssd_matrix.loc['A', 'B'] > 0, \
            f"Different series should have SSD>0, got {ssd_matrix.loc['A', 'B']}"

    def test_ssd_symmetry(self):
        """SSD(A,B) should equal SSD(B,A)."""
        prices = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [1.0, 3.0, 2.0],
        })
        normalized = normalize_prices(prices)
        ssd_matrix = calculate_ssd_matrix(normalized)

        assert ssd_matrix.loc['A', 'B'] == ssd_matrix.loc['B', 'A'], \
            "SSD should be symmetric"

    def test_ssd_diagonal_zero(self):
        """Diagonal of SSD matrix should be 0 (self-comparison)."""
        prices = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [1.0, 3.0, 2.0],
            'C': [2.0, 1.0, 3.0],
        })
        normalized = normalize_prices(prices)
        ssd_matrix = calculate_ssd_matrix(normalized)

        for sym in ['A', 'B', 'C']:
            assert ssd_matrix.loc[sym, sym] == 0.0, \
                f"Diagonal element {sym},{sym} should be 0"

    def test_ssd_manual_calculation(self):
        """Verify SSD matches manual calculation with sufficient data points.

        Using 10 data points for robust verification with HARDCODED expected value.
        """
        # A: linear growth 100 -> 200 (10% increments)
        # B: stays flat at 100
        prices = pd.DataFrame({
            'A': [100.0 + i * 10 for i in range(11)],  # 100, 110, ..., 200
            'B': [100.0] * 11,  # stays at 100
        })
        normalized = normalize_prices(prices)
        ssd_matrix = calculate_ssd_matrix(normalized)

        # Pre-calculated expected SSD (hardcoded for test stability):
        # Normalized A: [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        # Normalized B: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # Diff:         [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # Squared:      [0.00, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.00]
        # Sum = 0.00 + 0.01 + 0.04 + 0.09 + 0.16 + 0.25 + 0.36 + 0.49 + 0.64 + 0.81 + 1.00 = 3.85
        HARDCODED_EXPECTED_SSD = 3.85  # Pre-calculated, not computed in test

        actual_ssd = ssd_matrix.loc['A', 'B']
        assert abs(actual_ssd - HARDCODED_EXPECTED_SSD) < 0.001, \
            f"Expected SSD={HARDCODED_EXPECTED_SSD:.4f}, got {actual_ssd:.4f}"

        # Also verify it's substantial (catches bugs that always return 0)
        assert actual_ssd > 3.0, f"SSD should be substantial (~3.85), got {actual_ssd}"

    def test_ssd_calculation_accuracy_detailed(self):
        """Verify SSD calculation with detailed manual verification.

        Per user test spec: A=[1, 2, 3], B=[1.1, 1.9, 3.1]
        Normalized A: [1.0, 2.0, 3.0]
        Normalized B: [1.0, 1.727..., 2.818...]
        SSD = (1-1)^2 + (2-1.727)^2 + (3-2.818)^2
        """
        prices = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [1.1, 1.9, 3.1],
        })
        normalized = normalize_prices(prices)

        # Verify normalization first
        assert abs(normalized['A'].iloc[0] - 1.0) < 0.001
        assert abs(normalized['A'].iloc[1] - 2.0) < 0.001
        assert abs(normalized['A'].iloc[2] - 3.0) < 0.001

        assert abs(normalized['B'].iloc[0] - 1.0) < 0.001
        assert abs(normalized['B'].iloc[1] - (1.9/1.1)) < 0.001  # 1.727...
        assert abs(normalized['B'].iloc[2] - (3.1/1.1)) < 0.001  # 2.818...

        # Calculate expected SSD manually
        diff_1 = 1.0 - 1.0  # 0
        diff_2 = 2.0 - (1.9/1.1)  # 0.2727...
        diff_3 = 3.0 - (3.1/1.1)  # 0.1818...
        expected_ssd = diff_1**2 + diff_2**2 + diff_3**2

        ssd_matrix = calculate_ssd_matrix(normalized)
        actual_ssd = ssd_matrix.loc['A', 'B']

        assert abs(actual_ssd - expected_ssd) < 0.0001, \
            f"Expected SSD={expected_ssd:.6f}, got {actual_ssd:.6f}"


class TestFormationPeriod:
    """Test suite for formation period handling."""

    def test_ssd_uses_formation_period_only(self):
        """SSD should be calculated using only formation period data.

        Setup: 2 years of data
        Formation: First 12 months
        Trading: Next 6 months (should be IGNORED for SSD)
        """
        # Create 2 years of daily data (~500 days)
        formation_days = 252  # ~1 year
        trading_days = 126    # ~6 months
        total_days = formation_days + trading_days

        dates = pd.date_range('2022-01-01', periods=total_days, freq='B')

        # Formation period: A and B move together (low SSD)
        # Trading period: A and B diverge wildly (high SSD if included)
        prices_a_formation = [100.0 + i * 0.1 for i in range(formation_days)]
        prices_b_formation = [100.0 + i * 0.1 for i in range(formation_days)]

        prices_a_trading = [prices_a_formation[-1] + i * 2 for i in range(trading_days)]  # Spike up
        prices_b_trading = [prices_b_formation[-1] - i * 2 for i in range(trading_days)]  # Spike down

        full_prices = pd.DataFrame({
            'A': prices_a_formation + prices_a_trading,
            'B': prices_b_formation + prices_b_trading,
        }, index=dates)

        # SSD calculated on FORMATION ONLY
        formation_prices = full_prices.iloc[:formation_days]
        normalized_formation = normalize_prices(formation_prices)
        ssd_formation = calculate_ssd_matrix(normalized_formation)

        # SSD calculated on FULL DATA (wrong approach)
        normalized_full = normalize_prices(full_prices)
        ssd_full = calculate_ssd_matrix(normalized_full)

        # Formation-only SSD should be very low (stocks moved together)
        # Full-data SSD should be much higher (divergence in trading period)
        assert ssd_formation.loc['A', 'B'] < ssd_full.loc['A', 'B'], \
            "Formation-only SSD should be lower than full-data SSD"

        # Formation SSD should be very close to 0 (stocks track perfectly)
        assert ssd_formation.loc['A', 'B'] < 0.01, \
            f"Formation SSD should be near 0, got {ssd_formation.loc['A', 'B']}"


class TestPairSelection:
    """Test suite for pair selection."""

    def test_select_top_pairs_ordering(self):
        """Top pairs should be ordered by SSD ascending."""
        # Create prices where A-B are similar, C-D are different
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 120.0, 130.0],
            'B': [100.0, 111.0, 121.0, 131.0],  # Very similar to A
            'C': [100.0, 150.0, 100.0, 150.0],  # Volatile
            'D': [100.0, 80.0, 120.0, 90.0],    # Inverse-ish
        })
        normalized = normalize_prices(prices)
        ssd_matrix = calculate_ssd_matrix(normalized)
        top_pairs = select_top_pairs(ssd_matrix, n=3)

        # A-B should be first (lowest SSD)
        first_pair = tuple(sorted(top_pairs[0]))
        assert first_pair == ('A', 'B'), \
            f"A-B should be top pair (lowest SSD), got {top_pairs[0]}"

        # Verify SSD ordering: each subsequent pair should have higher SSD
        for i in range(len(top_pairs) - 1):
            pair1 = top_pairs[i]
            pair2 = top_pairs[i + 1]
            ssd1 = ssd_matrix.loc[pair1[0], pair1[1]]
            ssd2 = ssd_matrix.loc[pair2[0], pair2[1]]
            assert ssd1 <= ssd2, \
                f"Pairs should be ordered by SSD: {pair1} (SSD={ssd1:.4f}) should be <= {pair2} (SSD={ssd2:.4f})"

        # Verify A-B SSD is actually small (similar pairs)
        ab_ssd = ssd_matrix.loc['A', 'B']
        assert ab_ssd < 0.01, f"A-B SSD should be very small, got {ab_ssd:.4f}"

    def test_select_top_pairs_count(self):
        """Should return requested number of pairs."""
        prices = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [1.0, 2.1, 3.0],
            'C': [1.0, 1.9, 3.1],
            'D': [1.0, 2.5, 2.5],
        })
        normalized = normalize_prices(prices)
        ssd_matrix = calculate_ssd_matrix(normalized)

        for n in [1, 2, 3]:
            top_pairs = select_top_pairs(ssd_matrix, n=n)
            assert len(top_pairs) == n, f"Expected {n} pairs, got {len(top_pairs)}"

    def test_select_top_pairs_unique(self):
        """All selected pairs should be unique."""
        prices = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [1.0, 2.1, 3.0],
            'C': [1.0, 1.9, 3.1],
            'D': [1.0, 2.5, 2.5],
        })
        normalized = normalize_prices(prices)
        ssd_matrix = calculate_ssd_matrix(normalized)
        top_pairs = select_top_pairs(ssd_matrix, n=6)  # All pairs

        # Check no duplicates
        pair_set = set(tuple(sorted(p)) for p in top_pairs)
        assert len(pair_set) == len(top_pairs), "Pairs should be unique"


class TestGetPairStats:
    """Test suite for get_pair_stats function."""

    def test_returns_dict_with_required_keys(self):
        """Should return dict with all expected keys."""
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 120.0, 130.0],
            'B': [100.0, 105.0, 115.0, 125.0],
        })
        normalized = normalize_prices(prices)
        stats = get_pair_stats(normalized, ('A', 'B'))

        assert 'pair' in stats
        assert 'ssd' in stats
        assert 'correlation' in stats
        assert 'spread_mean' in stats
        assert 'spread_std' in stats

    def test_pair_tuple_preserved(self):
        """Should preserve the pair tuple in result."""
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 120.0],
            'B': [100.0, 105.0, 115.0],
        })
        normalized = normalize_prices(prices)
        stats = get_pair_stats(normalized, ('A', 'B'))

        assert stats['pair'] == ('A', 'B')

    def test_ssd_matches_matrix_calculation(self):
        """SSD from get_pair_stats should match SSD matrix."""
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 120.0, 130.0],
            'B': [100.0, 108.0, 118.0, 128.0],
        })
        normalized = normalize_prices(prices)
        stats = get_pair_stats(normalized, ('A', 'B'))
        ssd_matrix = calculate_ssd_matrix(normalized)

        assert abs(stats['ssd'] - ssd_matrix.loc['A', 'B']) < 0.0001

    def test_correlation_between_minus_one_and_one(self):
        """Correlation should be between -1 and 1, and reflect actual relationship.

        Verifies both bounds AND that negatively correlated pairs show negative correlation.
        """
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 120.0, 130.0],  # Increasing
            'B': [100.0, 90.0, 80.0, 70.0],     # Decreasing (negatively correlated)
        })
        normalized = normalize_prices(prices)
        stats = get_pair_stats(normalized, ('A', 'B'))

        # Verify bounds
        assert -1.0 <= stats['correlation'] <= 1.0, \
            f"Correlation must be in [-1, 1], got {stats['correlation']}"

        # Verify actual value: perfectly inverse linear should be -1.0
        assert stats['correlation'] < -0.9, \
            f"Inverse linear prices should have correlation near -1, got {stats['correlation']}"

    def test_highly_correlated_pair(self):
        """Highly similar prices should have high correlation."""
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 120.0, 130.0],
            'B': [100.0, 110.5, 120.5, 130.5],  # Very similar
        })
        normalized = normalize_prices(prices)
        stats = get_pair_stats(normalized, ('A', 'B'))

        assert stats['correlation'] > 0.99

    def test_spread_mean_near_zero_for_similar_pairs(self):
        """Spread mean should be near zero for similar normalized prices."""
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 120.0, 130.0],
            'B': [100.0, 110.0, 120.0, 130.0],  # Identical
        })
        normalized = normalize_prices(prices)
        stats = get_pair_stats(normalized, ('A', 'B'))

        assert abs(stats['spread_mean']) < 0.001

    def test_spread_std_positive(self):
        """Spread std should be positive for non-identical series and reflect spread variance.

        Verifies both that std > 0 AND that it's in a reasonable range for the test data.
        """
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 120.0, 130.0],  # +10% each step
            'B': [100.0, 108.0, 122.0, 128.0],  # Slightly different (+8%, +22%, +28%)
        })
        normalized = normalize_prices(prices)
        stats = get_pair_stats(normalized, ('A', 'B'))

        # Basic positivity check
        assert stats['spread_std'] > 0, "Spread std should be positive for different series"

        # Spread (A - B normalized) varies from:
        # Day 0: 1.0 - 1.0 = 0.0
        # Day 1: 1.1 - 1.08 = 0.02
        # Day 2: 1.2 - 1.22 = -0.02
        # Day 3: 1.3 - 1.28 = 0.02
        # std of [0, 0.02, -0.02, 0.02] ≈ 0.018
        assert 0.01 < stats['spread_std'] < 0.05, \
            f"Spread std should be ~0.018 for this data, got {stats['spread_std']}"


class TestVectorizedSSDEquivalence:
    """Test that vectorized SSD implementations match the original loop version."""

    def test_vectorized_matches_loop_small(self):
        """Vectorized SSD should match loop-based for small dataset."""
        np.random.seed(42)
        prices = pd.DataFrame(
            np.random.randn(20, 5).cumsum(axis=0) + 100,
            columns=['A', 'B', 'C', 'D', 'E']
        )
        normalized = normalize_prices(prices)

        ssd_loop = _calculate_ssd_matrix_loop(normalized)
        ssd_vec = _calculate_ssd_matrix_vectorized(normalized)

        np.testing.assert_allclose(
            ssd_loop.values, ssd_vec.values, rtol=1e-10,
            err_msg="Vectorized SSD should match loop-based implementation"
        )

    def test_vectorized_matches_loop_medium(self):
        """Vectorized SSD should match loop-based for medium dataset."""
        np.random.seed(123)
        prices = pd.DataFrame(
            np.random.randn(100, 20).cumsum(axis=0) + 100,
            columns=[f'S{i}' for i in range(20)]
        )
        normalized = normalize_prices(prices)

        ssd_loop = _calculate_ssd_matrix_loop(normalized)
        ssd_vec = _calculate_ssd_matrix_vectorized(normalized)

        np.testing.assert_allclose(
            ssd_loop.values, ssd_vec.values, rtol=1e-10,
            err_msg="Vectorized SSD should match loop-based implementation"
        )

    def test_chunked_matches_vectorized(self):
        """Chunked SSD should match full vectorized version."""
        np.random.seed(456)
        prices = pd.DataFrame(
            np.random.randn(50, 15).cumsum(axis=0) + 100,
            columns=[f'S{i}' for i in range(15)]
        )
        normalized = normalize_prices(prices)

        ssd_vec = _calculate_ssd_matrix_vectorized(normalized)
        ssd_chunked = _calculate_ssd_matrix_chunked(normalized, chunk_size=5)

        np.testing.assert_allclose(
            ssd_vec.values, ssd_chunked.values, rtol=1e-10,
            err_msg="Chunked SSD should match full vectorized version"
        )

    def test_vectorized_with_nan_values(self):
        """Vectorized SSD should handle NaN values correctly."""
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            'B': [100.0, 105.0, np.nan, 125.0, 135.0, 145.0],  # One NaN
            'C': [100.0, 108.0, 118.0, np.nan, np.nan, 148.0],  # Two NaNs
        })
        normalized = normalize_prices(prices)

        ssd_loop = _calculate_ssd_matrix_loop(normalized)
        ssd_vec = _calculate_ssd_matrix_vectorized(normalized)

        np.testing.assert_allclose(
            ssd_loop.values, ssd_vec.values, rtol=1e-10,
            err_msg="Vectorized SSD should match loop-based with NaN values"
        )

    def test_vectorized_insufficient_overlap_returns_inf(self):
        """Both implementations should return inf for pairs with <50% overlap."""
        # Create data where B only has 4 valid values out of 10 (40% overlap)
        # First value must be valid for normalization to work
        prices = pd.DataFrame({
            'A': [100.0 + i for i in range(10)],
            'B': [100.0, np.nan, np.nan, np.nan, np.nan, np.nan, 105.0, 110.0, 115.0, np.nan],  # 40% overlap
        })
        normalized = normalize_prices(prices)

        ssd_loop = _calculate_ssd_matrix_loop(normalized)
        ssd_vec = _calculate_ssd_matrix_vectorized(normalized)

        # Both should return infinity for insufficient overlap (<50%)
        assert np.isinf(ssd_loop.loc['A', 'B']), "Loop: Should return inf for <50% overlap"
        assert np.isinf(ssd_vec.loc['A', 'B']), "Vectorized: Should return inf for <50% overlap"

    def test_calculate_ssd_matrix_uses_vectorized(self):
        """Main calculate_ssd_matrix should use vectorized by default."""
        np.random.seed(789)
        prices = pd.DataFrame(
            np.random.randn(30, 8).cumsum(axis=0) + 100,
            columns=[f'S{i}' for i in range(8)]
        )
        normalized = normalize_prices(prices)

        ssd_default = calculate_ssd_matrix(normalized)
        ssd_vec = _calculate_ssd_matrix_vectorized(normalized)

        np.testing.assert_allclose(
            ssd_default.values, ssd_vec.values, rtol=1e-10,
            err_msg="Default calculate_ssd_matrix should use vectorized version"
        )


# -----------------------------------------------------------------------------
# Bug Fix Tests - normalize_prices Zero Handling (Bug #3)
# -----------------------------------------------------------------------------


class TestNormalizePricesZeroHandling:
    """Tests for zero/negative price handling in normalization (Bug #3 fix)."""

    def test_normalize_zero_first_price_filtered(self):
        """Column with first price = 0 should be dropped."""
        prices = pd.DataFrame({
            'A': [100.0, 101.0, 102.0],
            'B': [0.0, 50.0, 51.0],  # First price is 0
            'C': [200.0, 201.0, 202.0]
        })
        result = normalize_prices(prices)

        assert 'B' not in result.columns, "Column with first price 0 should be dropped"
        assert 'A' in result.columns, "Valid column A should remain"
        assert 'C' in result.columns, "Valid column C should remain"

    def test_normalize_negative_first_price_filtered(self):
        """Column with first price < 0 should be dropped (data error)."""
        prices = pd.DataFrame({
            'A': [100.0, 101.0, 102.0],
            'B': [-10.0, 50.0, 51.0],  # Negative first price
        })
        result = normalize_prices(prices)

        assert 'B' not in result.columns, "Column with negative first price should be dropped"
        assert 'A' in result.columns, "Valid column A should remain"

    def test_normalize_all_zero_first_prices(self):
        """All columns with zero first price returns empty DataFrame."""
        prices = pd.DataFrame({
            'A': [0.0, 101.0, 102.0],
            'B': [0.0, 50.0, 51.0],
        })
        result = normalize_prices(prices)

        assert result.empty, "Result should be empty when all columns have zero first price"

    def test_normalize_no_infinity_values(self):
        """Result should never contain infinity values."""
        prices = pd.DataFrame({
            'A': [100.0, 101.0, 102.0],
            'B': [0.0001, 50.0, 51.0],  # Very small but valid
        })
        result = normalize_prices(prices)

        assert not np.isinf(result.values).any(), "Result should never contain infinity values"

    def test_normalize_empty_dataframe(self):
        """Empty DataFrame should return empty DataFrame."""
        prices = pd.DataFrame()
        result = normalize_prices(prices)

        assert result.empty


# -----------------------------------------------------------------------------
# Bug Fix Tests - Auto-Chunked SSD for Large Universes (Bug #9)
# -----------------------------------------------------------------------------


class TestAutoChunkedSSD:
    """Tests for auto-selecting chunked SSD for large universes (Bug #9 fix)."""

    def test_auto_selects_vectorized_for_small_universe(self):
        """Small universes (< threshold) should use vectorized implementation."""
        np.random.seed(42)
        # 50 symbols (below default threshold of 200)
        prices = pd.DataFrame(
            np.random.randn(30, 50).cumsum(axis=0) + 100,
            columns=[f'S{i}' for i in range(50)]
        )
        normalized = normalize_prices(prices)

        # Both should return same results
        ssd_auto = calculate_ssd_matrix(normalized)  # Auto-selects
        ssd_vec = _calculate_ssd_matrix_vectorized(normalized)

        np.testing.assert_allclose(
            ssd_auto.values, ssd_vec.values, rtol=1e-10,
            err_msg="Auto-selected should match vectorized for small universe"
        )

    def test_chunked_matches_vectorized_results(self):
        """Chunked and vectorized should produce identical results."""
        np.random.seed(123)
        prices = pd.DataFrame(
            np.random.randn(30, 30).cumsum(axis=0) + 100,
            columns=[f'S{i}' for i in range(30)]
        )
        normalized = normalize_prices(prices)

        ssd_vec = _calculate_ssd_matrix_vectorized(normalized)
        ssd_chunked = _calculate_ssd_matrix_chunked(normalized, chunk_size=10)

        np.testing.assert_allclose(
            ssd_vec.values, ssd_chunked.values, rtol=1e-10,
            err_msg="Chunked and vectorized should produce identical results"
        )

    def test_explicit_chunked_flag(self):
        """Explicit use_chunked=True should use chunked implementation."""
        np.random.seed(456)
        prices = pd.DataFrame(
            np.random.randn(30, 20).cumsum(axis=0) + 100,
            columns=[f'S{i}' for i in range(20)]
        )
        normalized = normalize_prices(prices)

        ssd_explicit_chunked = calculate_ssd_matrix(normalized, use_chunked=True)
        ssd_chunked = _calculate_ssd_matrix_chunked(normalized)

        np.testing.assert_allclose(
            ssd_explicit_chunked.values, ssd_chunked.values, rtol=1e-10,
            err_msg="Explicit use_chunked=True should use chunked implementation"
        )

    def test_explicit_vectorized_flag(self):
        """Explicit use_chunked=False should use vectorized implementation."""
        np.random.seed(789)
        prices = pd.DataFrame(
            np.random.randn(30, 20).cumsum(axis=0) + 100,
            columns=[f'S{i}' for i in range(20)]
        )
        normalized = normalize_prices(prices)

        ssd_explicit_vec = calculate_ssd_matrix(normalized, use_chunked=False)
        ssd_vec = _calculate_ssd_matrix_vectorized(normalized)

        np.testing.assert_allclose(
            ssd_explicit_vec.values, ssd_vec.values, rtol=1e-10,
            err_msg="Explicit use_chunked=False should use vectorized implementation"
        )


# -----------------------------------------------------------------------------
# Bug Fix Tests - Configurable SSD Overlap Threshold (Bug #16)
# -----------------------------------------------------------------------------


class TestSSDOverlapThreshold:
    """Tests for configurable SSD overlap threshold (Bug #16 fix)."""

    def test_custom_overlap_threshold(self):
        """Custom min_overlap should be respected."""
        # Create data where overlap is ~60%
        prices_a = pd.Series([100.0 + i for i in range(10)])
        prices_b = pd.Series([100.0, np.nan, np.nan, np.nan, np.nan, 105.0, 106.0, 107.0, 108.0, 109.0])
        # 6 valid points out of 10 = 60% overlap

        # With 50% threshold (default), should work
        ssd_50 = calculate_ssd(prices_a, prices_b, min_overlap=0.5)
        assert np.isfinite(ssd_50), "60% overlap should pass 50% threshold"

        # With 70% threshold, should return inf
        ssd_70 = calculate_ssd(prices_a, prices_b, min_overlap=0.7)
        assert np.isinf(ssd_70), "60% overlap should fail 70% threshold"

    def test_default_overlap_threshold(self):
        """Default 50% threshold should work as expected."""
        prices_a = pd.Series([100.0 + i for i in range(10)])
        prices_b = pd.Series([100.0, np.nan, np.nan, np.nan, np.nan, 105.0, 106.0, 107.0, 108.0, 109.0])
        # 6 valid points = 60% overlap

        ssd = calculate_ssd(prices_a, prices_b)  # Default min_overlap=0.5
        assert np.isfinite(ssd), "60% overlap should pass default 50% threshold"

    def test_exactly_threshold_overlap(self):
        """Overlap exactly at threshold should pass."""
        prices_a = pd.Series([100.0 + i for i in range(10)])
        prices_b = pd.Series([100.0, np.nan, np.nan, np.nan, np.nan, 105.0, 106.0, 107.0, 108.0, 109.0])
        # 6 valid points = 60% overlap

        # With exactly 60% threshold
        ssd = calculate_ssd(prices_a, prices_b, min_overlap=0.6)
        assert np.isfinite(ssd), "60% overlap should pass 60% threshold"
