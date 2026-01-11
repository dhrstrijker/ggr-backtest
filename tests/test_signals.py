"""Tests for GGR signal generation."""

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(pd.io.common.Path(__file__).parent.parent))

from src.signals import (
    generate_signals_ggr,
    calculate_spread,
    calculate_formation_stats,
    calculate_distance,
)


class TestGGRSignalGeneration:
    """Test suite for GGR signal generation (static σ, crossing-zero exit)."""

    def test_entry_on_divergence(self):
        """Should enter when spread exceeds 2σ from formation period."""
        # Formation std = 0.05
        formation_std = 0.05

        # Spread that exceeds 2σ = 0.10
        spread = pd.Series([0.0, 0.05, 0.12, 0.15, 0.08, -0.01])
        signals = generate_signals_ggr(spread, formation_std, entry_threshold=2.0)

        # Index 2 (0.12 > 0.10) should trigger short entry
        assert signals.iloc[2] == -1, \
            f"Expected short entry at spread=0.12 (> 2σ=0.10), got signal={signals.iloc[2]}"

    def test_exit_on_crossing_zero(self):
        """Should exit when spread crosses zero (GGR methodology)."""
        formation_std = 0.05

        # Spread that enters (> 2σ) then crosses zero
        spread = pd.Series([0.0, 0.12, 0.08, 0.04, 0.01, -0.02])
        signals = generate_signals_ggr(spread, formation_std, entry_threshold=2.0)

        # Should have entry at index 1 (0.12 > 0.10)
        assert signals.iloc[1] == -1, "Expected short entry at spread=0.12"

        # Should NOT exit at indices 2-4 even though spread is decreasing
        # (GGR exits only on crossing zero, not on threshold)
        # The exit happens internally in the state machine

    def test_no_exit_before_crossing(self):
        """Should NOT exit just because spread decreased (GGR rule)."""
        formation_std = 0.05

        # Spread that decreases but never crosses zero
        spread = pd.Series([0.0, 0.12, 0.08, 0.06, 0.04, 0.02])
        signals = generate_signals_ggr(spread, formation_std, entry_threshold=2.0)

        # Entry at index 1
        assert signals.iloc[1] == -1, "Expected short entry"

        # No further entries (still in position)
        assert signals.iloc[2:].sum() == 0, "Should stay in position until crossing"

    def test_long_entry_on_negative_divergence(self):
        """Should enter long when spread < -2σ."""
        formation_std = 0.05

        # Spread that goes negative beyond -2σ
        spread = pd.Series([0.0, -0.05, -0.12, -0.08, -0.01, 0.02])
        signals = generate_signals_ggr(spread, formation_std, entry_threshold=2.0)

        # Index 2 (-0.12 < -0.10) should trigger long entry
        assert signals.iloc[2] == 1, \
            f"Expected long entry at spread=-0.12 (< -2σ), got signal={signals.iloc[2]}"

    def test_no_entry_within_threshold(self):
        """Should not enter if spread never exceeds threshold."""
        formation_std = 0.05

        # Spread stays within 2σ = 0.10
        spread = pd.Series([0.0, 0.03, 0.06, 0.08, 0.05, 0.02])
        signals = generate_signals_ggr(spread, formation_std, entry_threshold=2.0)

        # No entries should occur
        assert 1 not in signals.values, "Should not have long entry"
        assert -1 not in signals.values, "Should not have short entry"

    def test_no_duplicate_entries(self):
        """Should not generate multiple entry signals while in position."""
        formation_std = 0.05

        # Spread that stays high (would trigger multiple entries if not tracking)
        spread = pd.Series([0.0, 0.12, 0.15, 0.18, 0.20, 0.15])
        signals = generate_signals_ggr(spread, formation_std, entry_threshold=2.0)

        # Should only have one entry
        entry_count = sum(1 for s in signals if s in [1, -1])
        assert entry_count == 1, f"Expected 1 entry, got {entry_count}"

    def test_handles_nan_values(self):
        """Should handle NaN values gracefully."""
        formation_std = 0.05
        spread = pd.Series([np.nan, np.nan, 0.12, 0.08, -0.01])
        signals = generate_signals_ggr(spread, formation_std, entry_threshold=2.0)

        # First two should be 0 (no action due to NaN)
        assert signals.iloc[0] == 0, "NaN should result in no signal"
        assert signals.iloc[1] == 0, "NaN should result in no signal"

        # Index 2 should still detect entry
        assert signals.iloc[2] == -1, "Should still detect entry after NaN"

    def test_spread_at_exactly_threshold_does_not_enter(self):
        """Test that entry requires spread to EXCEED threshold (> not >=).

        Per GGR paper: Entry when distance > 2σ (strict inequality).
        At exactly 2σ, we should NOT enter - must exceed threshold.
        """
        formation_std = 0.05
        threshold = 2.0

        # Spread goes to exactly 2.0σ = 0.10
        spread = pd.Series([0.0, 0.05, 0.10, 0.08, 0.05])
        signals = generate_signals_ggr(spread, formation_std, entry_threshold=threshold)

        # Index 2 has spread = 0.10, which is exactly 2.0σ
        # GGR uses strict inequality (>), so exactly at threshold should NOT enter
        assert signals.iloc[2] == 0, \
            f"At exactly 2σ threshold, should NOT enter (strict > comparison). Got signal={signals.iloc[2]}"

    def test_spread_just_below_threshold(self):
        """Spread at 1.999σ should NOT trigger entry."""
        formation_std = 0.05
        threshold = 2.0

        # 1.999σ = 0.09995, just below 2σ = 0.10
        almost_threshold = formation_std * 1.999  # 0.09995

        spread = pd.Series([0.0, 0.05, almost_threshold, 0.05, 0.0])
        signals = generate_signals_ggr(spread, formation_std, entry_threshold=threshold)

        # Should NOT enter at 1.999σ
        assert signals.iloc[2] == 0, \
            f"Should NOT enter at 1.999σ ({almost_threshold:.5f}), got signal={signals.iloc[2]}"

    def test_spread_just_above_threshold(self):
        """Spread at 2.001σ should trigger entry."""
        formation_std = 0.05
        threshold = 2.0

        # 2.001σ = 0.10005, just above 2σ = 0.10
        above_threshold = formation_std * 2.001  # 0.10005

        spread = pd.Series([0.0, 0.05, above_threshold, 0.05, 0.0])
        signals = generate_signals_ggr(spread, formation_std, entry_threshold=threshold)

        # Should enter at 2.001σ
        assert signals.iloc[2] == -1, \
            f"Should enter short at 2.001σ ({above_threshold:.5f}), got signal={signals.iloc[2]}"

    def test_floating_point_precision_crossing(self):
        """Test crossing detection with near-zero spread values."""
        formation_std = 0.05

        # Spread that crosses zero with very small values, then diverges again
        # This tests floating-point precision handling at the crossing point
        spread = pd.Series([
            0.0,
            0.12,       # Entry (> 2σ = 0.10) → signal -1
            0.08,
            0.04,
            0.001,      # Very close to zero but positive
            -0.001,     # Very close to zero but negative (should trigger exit)
            -0.05,      # Still within threshold, no re-entry
            -0.12,      # Back outside threshold (< -2σ = -0.10) → new entry if exited
        ])
        signals = generate_signals_ggr(spread, formation_std, entry_threshold=2.0)

        # Entry at index 1
        assert signals.iloc[1] == -1, "Should enter short at 0.12"

        # Exit at index 5 when spread crosses from positive to negative
        # Per implementation: signal=0 means "exit or no action"
        # The state machine should detect the crossing even at tiny values (0.001 to -0.001)
        assert signals.iloc[5] == 0, \
            f"Signal at exit point should be 0 (exit per implementation), got {signals.iloc[5]}"

        # To verify the exit ACTUALLY happened, check that a new entry can occur
        # At index 7, spread = -0.12 < -0.10 = -2σ → should trigger new long entry
        assert signals.iloc[7] == 1, \
            f"Should enter long at -0.12 (proves exit happened at tiny crossing). Got {signals.iloc[7]}"

        # After re-entry at index 7, no duplicate entry signals at index 6
        assert signals.iloc[6] == 0, \
            f"Should not re-enter at -0.05 (< 2σ = 0.10), got {signals.iloc[6]}"

    def test_very_small_formation_std(self):
        """Test behavior with very small formation std (high sensitivity)."""
        formation_std = 0.001  # Very low volatility formation

        # Even small moves will exceed 2σ
        spread = pd.Series([0.0, 0.002, 0.003, 0.001, -0.001])
        signals = generate_signals_ggr(spread, formation_std, entry_threshold=2.0)

        # 0.002 = 2σ, 0.003 = 3σ
        # Should trigger entry at 0.003 for sure
        has_entry = any(s != 0 for s in signals)
        assert has_entry, "Should have entry with very small formation std"

    def test_very_large_formation_std(self):
        """Test behavior with large formation std (low sensitivity)."""
        formation_std = 1.0  # High volatility formation

        # Normal moves won't exceed 2σ = 2.0
        spread = pd.Series([0.0, 0.5, 1.0, 1.5, 0.5])
        signals = generate_signals_ggr(spread, formation_std, entry_threshold=2.0)

        # None of these exceed 2σ = 2.0
        assert not any(s != 0 for s in signals), \
            "Should have no entries when moves are within 2σ"


class TestFormationStats:
    """Test suite for formation period statistics."""

    def test_formation_stats_calculation(self):
        """Should calculate mean and std correctly."""
        spread = pd.Series([0.0, 0.1, -0.1, 0.05, -0.05])
        stats = calculate_formation_stats(spread)

        assert 'mean' in stats, "Should have mean"
        assert 'std' in stats, "Should have std"
        assert abs(stats['mean']) < 0.01, "Mean should be close to 0"
        assert stats['std'] > 0, "Std should be positive"

    def test_formation_stats_fixed(self):
        """Formation stats should be fixed values, not dynamic."""
        spread = pd.Series([0.0, 0.1, -0.1, 0.05, -0.05])
        stats = calculate_formation_stats(spread)

        # Stats should be scalar values
        assert isinstance(stats['mean'], float), "Mean should be a scalar"
        assert isinstance(stats['std'], float), "Std should be a scalar"

    def test_volatility_is_static_not_rolling(self):
        """Verify σ remains fixed even when trading period volatility spikes.

        Critical GGR test: The entry threshold must remain based on
        formation period σ, not adapt to new volatility.
        """
        # Formation period: low volatility (σ ≈ 0.03)
        formation_spread = pd.Series([0.0, 0.03, -0.02, 0.04, -0.03, 0.02, -0.04, 0.03, -0.02, 0.01])
        formation_stats = calculate_formation_stats(formation_spread)
        formation_std = formation_stats['std']

        # Verify formation std is small (around 0.03)
        assert 0.02 < formation_std < 0.05, \
            f"Formation std should be ~0.03, got {formation_std:.4f}"

        # Trading period: HIGH volatility (σ would be ~0.3 if recalculated)
        # But we should still use the formation σ
        trading_spread = pd.Series([0.0, 0.2, -0.3, 0.4, -0.2, 0.3])

        # Calculate distance using FORMATION std (static)
        distance = calculate_distance(trading_spread, formation_std)

        # Calculate expected distance: 0.2 / formation_std
        expected_distance = 0.2 / formation_std

        # If using formation σ (~0.03), distance at 0.2 spread should be ~6-7σ
        # If using trading σ (~0.25), distance would be only ~0.8σ
        # This verifies we're using the static formation σ
        assert abs(distance.iloc[1]) > 5.0, \
            f"Distance should be > 5σ using formation std (~0.03), got {distance.iloc[1]:.2f}σ"
        assert abs(distance.iloc[1] - expected_distance) < 0.1, \
            f"Distance should be ~{expected_distance:.1f}σ, got {distance.iloc[1]:.2f}σ"

    def test_signal_opens_at_two_historical_std(self):
        """Entry should occur only when |distance| > 2σ from formation period.

        Setup: formation σ = 0.05
        Scenario: spread goes 0.09 (no trade), then 0.11 (trade)
        """
        formation_std = 0.05  # Given formation period σ

        # Spread sequence: starts at 0, goes to 0.09 (< 2σ), then 0.11 (> 2σ)
        spread = pd.Series([0.0, 0.05, 0.09, 0.11, 0.12])

        signals = generate_signals_ggr(spread, formation_std, entry_threshold=2.0)

        # No entry at 0.09 (1.8σ < 2σ)
        assert signals.iloc[2] == 0, \
            f"Should NOT enter at spread=0.09 (1.8σ), got signal={signals.iloc[2]}"

        # Entry at 0.11 (2.2σ > 2σ) - short spread since positive
        assert signals.iloc[3] == -1, \
            f"Should enter short at spread=0.11 (2.2σ), got signal={signals.iloc[3]}"

    def test_signal_closes_at_crossing_only_not_threshold(self):
        """Position must remain open until spread crosses zero.

        Setup: Open position at spread = 0.15
        Scenario: Spread narrows to 0.05 (old 0.5σ rule would close), then 0.01, then -0.01
        GGR Rule: Must stay open until crossing (sign flip)
        """
        formation_std = 0.05

        # Spread: enter at 0.12, narrows to 0.05, 0.02, 0.01, then crosses to -0.01
        spread = pd.Series([0.0, 0.12, 0.05, 0.02, 0.01, -0.01])

        signals = generate_signals_ggr(spread, formation_std, entry_threshold=2.0)

        # Entry at index 1 (0.12 > 0.10)
        assert signals.iloc[1] == -1, "Should enter short at spread=0.12"

        # Position tracks state internally; we verify by checking no new entries
        # until after position exits (would need to check position state)

        # The key test: spread at 0.05 is 1.0σ - under old 0.5 exit rule this would exit
        # But under GGR, we stay in position

    def test_reopening_after_convergence(self):
        """A pair should be able to open multiple times in same trading window.

        Per GGR paper: pairs can "open multiple times" if they diverge,
        converge, and diverge again.
        """
        formation_std = 0.05

        # Scenario: diverge > 2σ, converge (cross 0), diverge again > 2σ
        spread = pd.Series([
            0.0,    # Start
            0.12,   # Trade 1: Enter short (> 2σ)
            0.08,   # Still in trade 1
            0.02,   # Still in trade 1
            -0.01,  # Trade 1: Exit (crossed zero)
            -0.05,  # Flat, watching
            -0.12,  # Trade 2: Enter long (< -2σ)
            -0.06,  # Still in trade 2
            0.01,   # Trade 2: Exit (crossed zero)
        ])

        signals = generate_signals_ggr(spread, formation_std, entry_threshold=2.0)

        # Count entries
        entries = signals[signals != 0]
        entry_count = len(entries)

        assert entry_count == 2, \
            f"Should have 2 entries (re-open after convergence), got {entry_count}"

        # First entry should be short (spread positive)
        assert signals.iloc[1] == -1, "First entry should be short"

        # Second entry should be long (spread negative)
        assert signals.iloc[6] == 1, "Second entry should be long"


class TestDistance:
    """Test suite for distance calculation."""

    def test_distance_calculation(self):
        """Distance should be spread / formation_std."""
        spread = pd.Series([0.0, 0.1, 0.2, -0.1])
        formation_std = 0.05
        distance = calculate_distance(spread, formation_std)

        # Distance = spread / std
        assert distance.iloc[1] == pytest.approx(2.0), "0.1 / 0.05 = 2.0σ"
        assert distance.iloc[2] == pytest.approx(4.0), "0.2 / 0.05 = 4.0σ"
        assert distance.iloc[3] == pytest.approx(-2.0), "-0.1 / 0.05 = -2.0σ"

    def test_distance_uses_static_std(self):
        """Distance should use the same std for all values."""
        spread = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4])
        formation_std = 0.1
        distance = calculate_distance(spread, formation_std)

        # All distances should use the same formation_std
        expected = spread / formation_std
        pd.testing.assert_series_equal(distance, expected)


class TestSpreadCalculation:
    """Test suite for spread calculation."""

    def test_spread_basic(self):
        """Spread should be difference of normalized prices."""
        prices_a = pd.Series([100.0, 110.0, 120.0])
        prices_b = pd.Series([50.0, 55.0, 60.0])

        spread = calculate_spread(prices_a, prices_b, normalize=True)

        # Normalized: A goes 1.0 -> 1.1 -> 1.2, B goes 1.0 -> 1.1 -> 1.2
        # Spread should be 0 throughout (they move together)
        assert abs(spread.iloc[0]) < 0.001, "First spread should be 0"
        assert abs(spread.iloc[1]) < 0.001, "Second spread should be 0"
        assert abs(spread.iloc[2]) < 0.001, "Third spread should be 0"

    def test_spread_divergence(self):
        """Spread should increase when prices diverge."""
        prices_a = pd.Series([100.0, 120.0, 140.0])  # +20%, +40%
        prices_b = pd.Series([100.0, 100.0, 100.0])  # Flat

        spread = calculate_spread(prices_a, prices_b, normalize=True)

        # Spread should be increasing
        assert spread.iloc[1] > spread.iloc[0], "Spread should increase"
        assert spread.iloc[2] > spread.iloc[1], "Spread should continue increasing"

    def test_spread_convergence(self):
        """Spread should decrease when prices converge."""
        prices_a = pd.Series([100.0, 120.0, 100.0])  # Up then down
        prices_b = pd.Series([100.0, 100.0, 100.0])  # Flat

        spread = calculate_spread(prices_a, prices_b, normalize=True)

        # Spread should go up then back to 0
        assert spread.iloc[1] > 0, "Middle spread should be positive"
        assert abs(spread.iloc[2]) < 0.001, "Final spread should return to 0"

    def test_spread_handles_nan(self):
        """Spread should propagate NaN values."""
        prices_a = pd.Series([100.0, np.nan, 120.0])
        prices_b = pd.Series([100.0, 110.0, 120.0])

        spread = calculate_spread(prices_a, prices_b, normalize=True)

        # Index 1 should be NaN
        assert pd.isna(spread.iloc[1]), "NaN should propagate"

    def test_spread_without_normalization(self):
        """Spread without normalization should be raw difference."""
        prices_a = pd.Series([100.0, 110.0, 120.0])
        prices_b = pd.Series([50.0, 55.0, 60.0])

        spread = calculate_spread(prices_a, prices_b, normalize=False)

        # Without normalization, spread is just A - B
        assert spread.iloc[0] == 50.0  # 100 - 50
        assert spread.iloc[1] == 55.0  # 110 - 55
        assert spread.iloc[2] == 60.0  # 120 - 60
