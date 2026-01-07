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
    select_top_pairs,
    get_pair_stats,
    rank_all_pairs,
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
        """Verify SSD matches manual calculation."""
        # Simple case: A = [1, 2], B = [1, 1]
        # Normalized: A = [1, 2], B = [1, 1] (already start at 1)
        # Diff = [0, 1], Squared = [0, 1], Sum = 1
        prices = pd.DataFrame({
            'A': [1.0, 2.0],
            'B': [1.0, 1.0],
        })
        normalized = normalize_prices(prices)
        ssd_matrix = calculate_ssd_matrix(normalized)

        expected_ssd = 1.0  # (1-1)^2 + (2-1)^2 = 0 + 1 = 1
        assert abs(ssd_matrix.loc['A', 'B'] - expected_ssd) < 0.001, \
            f"Expected SSD={expected_ssd}, got {ssd_matrix.loc['A', 'B']}"

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
        top_pairs = select_top_pairs(ssd_matrix, n=2)

        # A-B should be first (lowest SSD)
        assert top_pairs[0] == ('A', 'B') or top_pairs[0] == ('B', 'A'), \
            f"A-B should be top pair, got {top_pairs[0]}"

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
        """Correlation should be between -1 and 1."""
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 120.0, 130.0],
            'B': [100.0, 90.0, 80.0, 70.0],  # Negatively correlated
        })
        normalized = normalize_prices(prices)
        stats = get_pair_stats(normalized, ('A', 'B'))

        assert -1.0 <= stats['correlation'] <= 1.0

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
        """Spread std should be positive for non-identical series."""
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 120.0, 130.0],
            'B': [100.0, 108.0, 122.0, 128.0],  # Slightly different
        })
        normalized = normalize_prices(prices)
        stats = get_pair_stats(normalized, ('A', 'B'))

        assert stats['spread_std'] > 0


class TestRankAllPairs:
    """Test suite for rank_all_pairs function."""

    def test_returns_dataframe(self):
        """Should return a DataFrame."""
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 120.0],
            'B': [100.0, 105.0, 115.0],
            'C': [100.0, 120.0, 100.0],
        })
        normalized = normalize_prices(prices)
        result = rank_all_pairs(normalized)

        assert isinstance(result, pd.DataFrame)

    def test_correct_number_of_pairs(self):
        """Should have n*(n-1)/2 pairs for n symbols."""
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 120.0],
            'B': [100.0, 105.0, 115.0],
            'C': [100.0, 120.0, 100.0],
            'D': [100.0, 90.0, 80.0],
        })
        normalized = normalize_prices(prices)
        result = rank_all_pairs(normalized)

        # 4 symbols = 4*3/2 = 6 pairs
        assert len(result) == 6

    def test_has_required_columns(self):
        """Should have all expected columns."""
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 120.0],
            'B': [100.0, 105.0, 115.0],
        })
        normalized = normalize_prices(prices)
        result = rank_all_pairs(normalized)

        expected_columns = ['symbol_a', 'symbol_b', 'ssd', 'correlation',
                           'spread_mean', 'spread_std', 'rank']
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_sorted_by_ssd_ascending(self):
        """Results should be sorted by SSD ascending."""
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 120.0, 130.0],
            'B': [100.0, 110.0, 120.0, 130.0],  # Identical to A (SSD=0)
            'C': [100.0, 200.0, 100.0, 200.0],  # Very different
        })
        normalized = normalize_prices(prices)
        result = rank_all_pairs(normalized)

        # Check SSD is ascending
        ssd_values = result['ssd'].tolist()
        assert ssd_values == sorted(ssd_values), "Should be sorted by SSD ascending"

    def test_rank_column_sequential(self):
        """Rank column should be sequential starting from 1."""
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 120.0],
            'B': [100.0, 105.0, 115.0],
            'C': [100.0, 120.0, 100.0],
        })
        normalized = normalize_prices(prices)
        result = rank_all_pairs(normalized)

        expected_ranks = list(range(1, len(result) + 1))
        assert result['rank'].tolist() == expected_ranks

    def test_best_pair_has_rank_one(self):
        """Pair with lowest SSD should have rank 1."""
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 120.0, 130.0],
            'B': [100.0, 110.0, 120.0, 130.0],  # Identical to A
            'C': [100.0, 200.0, 100.0, 200.0],  # Very different
        })
        normalized = normalize_prices(prices)
        result = rank_all_pairs(normalized)

        # A-B should be rank 1
        best_pair = result[result['rank'] == 1].iloc[0]
        assert (best_pair['symbol_a'] == 'A' and best_pair['symbol_b'] == 'B') or \
               (best_pair['symbol_a'] == 'B' and best_pair['symbol_b'] == 'A')
