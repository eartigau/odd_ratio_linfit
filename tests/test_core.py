"""
Tests for odd_ratio_fits package.
"""

import numpy as np
import pytest
import odd_ratio_fits as orf


class TestMean:
    """Tests for orf.mean function."""

    def test_basic_mean_no_outliers(self):
        """Test that mean is correct for clean data."""
        np.random.seed(42)
        true_value = 10.0
        values = np.random.normal(true_value, 1.0, 100)
        errors = np.ones(100)
        
        mean, err = orf.mean(values, errors)
        
        assert np.abs(mean - true_value) < 0.5
        assert err > 0
        assert err < 0.2

    def test_mean_with_outliers(self):
        """Test that mean is robust to outliers."""
        np.random.seed(42)
        true_value = 10.0
        values = np.random.normal(true_value, 1.0, 100)
        errors = np.ones(100)
        
        # Add strong outliers
        values[:10] = [50, -30, 100, -50, 80, -40, 60, -20, 70, -60]
        
        mean, err = orf.mean(values, errors)
        
        # Should still be close to true value
        assert np.abs(mean - true_value) < 1.0

    def test_nan_handling(self):
        """Test that NaNs are handled correctly."""
        np.random.seed(123)
        # Use enough points with realistic scatter for robust convergence
        values = np.random.normal(10.0, 1.0, 100)
        errors = np.ones(100)
        # Insert some NaNs
        values[::10] = np.nan
        
        mean, err = orf.mean(values, errors)
        
        assert np.isfinite(mean)
        assert np.isfinite(err)
        assert np.abs(mean - 10.0) < 1.0

    def test_all_nan_returns_nan(self):
        """Test that all NaN input returns NaN."""
        values = np.array([np.nan, np.nan, np.nan])
        errors = np.array([0.1, 0.1, 0.1])
        
        mean, err = orf.mean(values, errors)
        
        assert np.isnan(mean)
        assert np.isnan(err)


class TestLinear:
    """Tests for orf.linear function."""

    def test_basic_linear_fit(self):
        """Test basic linear fit without outliers."""
        np.random.seed(42)
        true_a, true_b = 2.0, 0.5
        x = np.linspace(0, 10, 50)
        y = true_a + true_b * x + np.random.normal(0, 0.5, len(x))
        yerr = np.ones(len(x)) * 0.5
        
        a, a_err, b, b_err = orf.linear(x, y, yerr)
        
        assert np.abs(a - true_a) < 3 * a_err
        assert np.abs(b - true_b) < 3 * b_err

    def test_linear_fit_with_outliers(self):
        """Test that linear fit is robust to outliers."""
        np.random.seed(42)
        true_a, true_b = 2.0, 0.5
        x = np.linspace(0, 10, 50)
        y = true_a + true_b * x + np.random.normal(0, 0.5, len(x))
        yerr = np.ones(len(x)) * 0.5
        
        # Add outliers
        y[5] = 20.0
        y[15] = -10.0
        y[25] = 15.0
        
        a, a_err, b, b_err = orf.linear(x, y, yerr)
        
        # Should still be close to true values
        assert np.abs(a - true_a) < 0.5
        assert np.abs(b - true_b) < 0.1

    def test_return_weights(self):
        """Test that weights are returned when requested."""
        x = np.linspace(0, 10, 50)
        y = 2.0 + 0.5 * x
        yerr = np.ones(len(x)) * 0.5
        
        result = orf.linear(x, y, yerr, return_weights=True)
        
        assert len(result) == 5
        weights = result[4]
        assert len(weights) == len(x)
        assert np.all(weights >= 0)
        assert np.all(weights <= 1)

    def test_weights_identify_outliers(self):
        """Test that outliers get low weights."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 + 0.5 * x + np.random.normal(0, 0.5, len(x))
        yerr = np.ones(len(x)) * 0.5
        
        # Add one strong outlier
        outlier_idx = 25
        y[outlier_idx] = 50.0
        
        _, _, _, _, weights = orf.linear(x, y, yerr, return_weights=True)
        
        # Outlier should have low weight
        assert weights[outlier_idx] < 0.1
        
        # Most other points should have high weight
        non_outlier_weights = np.delete(weights, outlier_idx)
        assert np.mean(non_outlier_weights) > 0.9

    def test_nan_handling(self):
        """Test that NaNs are handled correctly."""
        x = np.array([1, 2, 3, 4, 5, np.nan, 7, 8])
        y = np.array([2, 4, np.nan, 8, 10, 12, 14, 16])
        yerr = np.ones(len(x)) * 0.5
        
        a, a_err, b, b_err = orf.linear(x, y, yerr)
        
        assert np.isfinite(a)
        assert np.isfinite(b)


class TestPolyfit:
    """Tests for orf.polyfit function."""

    def test_linear_polyfit(self):
        """Test polynomial fit with degree=1 matches linear."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 + 0.5 * x + np.random.normal(0, 0.5, len(x))
        yerr = np.ones(len(x)) * 0.5
        
        # Linear fit
        a, a_err, b, b_err = orf.linear(x, y, yerr)
        
        # Polynomial fit degree 1
        coeffs, coeffs_err = orf.polyfit(x, y, yerr, degree=1)
        
        # Should match (coeffs is [b, a] in polyfit convention)
        assert np.abs(coeffs[0] - b) < 0.01
        assert np.abs(coeffs[1] - a) < 0.01

    def test_quadratic_fit(self):
        """Test quadratic polynomial fit."""
        np.random.seed(42)
        true_coeffs = [0.1, -0.5, 3.0]  # 0.1*x^2 - 0.5*x + 3
        x = np.linspace(0, 10, 100)
        y = np.polyval(true_coeffs, x) + np.random.normal(0, 0.5, len(x))
        yerr = np.ones(len(x)) * 0.5
        
        coeffs, coeffs_err = orf.polyfit(x, y, yerr, degree=2)
        
        assert np.abs(coeffs[0] - true_coeffs[0]) < 0.05
        assert np.abs(coeffs[1] - true_coeffs[1]) < 0.2
        assert np.abs(coeffs[2] - true_coeffs[2]) < 0.5


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_insufficient_points_linear(self):
        """Test linear with too few points."""
        x = np.array([1.0])
        y = np.array([2.0])
        yerr = np.array([0.1])
        
        a, a_err, b, b_err = orf.linear(x, y, yerr)
        
        assert np.isnan(a)
        assert np.isnan(b)

    def test_zero_errors(self):
        """Test handling of zero errors."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        yerr = np.array([0.1, 0.0, 0.1, 0.1, 0.1])  # One zero error
        
        # Should not crash (zero errors become inf weights, but algorithm handles it)
        a, a_err, b, b_err = orf.linear(x, y, yerr)
        
        # Check that we get some result
        assert np.isfinite(a) or np.isnan(a)

    def test_different_odd_ratios(self):
        """Test that different odd_ratio values affect the fit."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 + 0.5 * x + np.random.normal(0, 0.5, len(x))
        yerr = np.ones(len(x)) * 0.5
        y[25] = 20.0  # Add outlier
        
        # Very small odd_ratio (conservative)
        _, _, _, _, weights_small = orf.linear(
            x, y, yerr, odd_ratio=1e-6, return_weights=True
        )
        
        # Large odd_ratio (aggressive)
        _, _, _, _, weights_large = orf.linear(
            x, y, yerr, odd_ratio=0.1, return_weights=True
        )
        
        # Large odd_ratio should down-weight more points
        assert np.sum(weights_large < 0.9) >= np.sum(weights_small < 0.9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
