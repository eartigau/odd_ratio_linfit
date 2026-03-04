"""
Core implementation of odd ratio fits: weighted mean, linear, and polynomial.

This module provides robust statistical methods using a Gaussian mixture model
approach to down-weight outliers while preserving the statistical properties
of the fit.
"""

import numpy as np
from typing import Tuple, Optional


def odd_ratio_mean(value: np.ndarray, error: np.ndarray,
                   odd_ratio: float = 2e-4, nmax: int = 10,
                   conv_cut: float = 1e-2) -> Tuple[float, float]:
    """
    Compute a weighted mean using a Gaussian mixture model for outlier rejection.

    This function implements a robust weighted mean that models each data point
    as coming from either a Gaussian distribution centered on the true value
    (valid measurements) or from a uniform background distribution (outliers).
    The probability of being an outlier is controlled by the `odd_ratio` parameter.

    Parameters
    ----------
    value : np.ndarray
        1D array of measured values.
    error : np.ndarray
        1D array of uncertainties (standard deviations) for each value.
    odd_ratio : float, optional
        Prior probability that any given point is an outlier.
        Recommended value: 0.0002 (i.e., 1 in 5000 points is bad).
        Default is 2e-4.
    nmax : int, optional
        Maximum number of iterations. Default is 10.
    conv_cut : float, optional
        Convergence criterion: iteration stops when the relative change
        in the mean is less than conv_cut times the bulk error.
        Default is 1e-2.

    Returns
    -------
    mean : float
        The robust weighted mean.
    error : float
        The uncertainty on the weighted mean.

    Notes
    -----
    The algorithm works as follows:

    1. Initialize the guess with the median of the values
    2. For each iteration:
       a. Compute the Gaussian likelihood for each point being valid
       b. Compute the posterior probability of each point being an outlier
       c. Weight points by their probability of being valid
       d. Update the mean estimate
    3. Iterate until convergence or maximum iterations reached

    The odd_ratio parameter controls how aggressively outliers are down-weighted.
    A smaller odd_ratio (e.g., 1e-5) will reject fewer points, while a larger
    value (e.g., 1e-2) will be more aggressive in rejecting outliers.

    References
    ----------
    Artigau et al. (2022), AJ, 164, 84, Appendix A
    "Line-by-line Velocity Measurements: an Outlier-resistant Method for
    Precision Velocimetry"

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> # Generate data with outliers
    >>> true_value = 10.0
    >>> n_points = 100
    >>> values = np.random.normal(true_value, 1.0, n_points)
    >>> errors = np.ones(n_points)
    >>> # Add outliers
    >>> values[0:5] = [50, -30, 100, -50, 80]
    >>> mean, err = odd_ratio_mean(values, errors)
    >>> print(f"Estimated mean: {mean:.2f} ± {err:.2f}")
    """
    # Deal with NaNs in value or error
    keep = np.isfinite(value) & np.isfinite(error)
    # Deal with no finite values
    if np.sum(keep) == 0:
        return np.nan, np.nan
    # Remove NaNs from arrays
    value, error = value[keep], error[keep]
    # Work out some values to speed up loop
    error2 = error ** 2
    # Placeholders for the "while" below
    guess_prev = np.inf
    # The 'guess' must be started as close as we possibly can to the actual
    # value. Starting beyond ~3 sigma (or whatever the odd_ratio implies)
    # would lead to the rejection of pretty much all points and would
    # completely mess the convergence of the loop
    guess = np.nanmedian(value)
    bulk_error = 1.0
    ite = 0
    # Loop around until we do all required iterations
    while (np.abs(guess - guess_prev) / bulk_error > conv_cut) and (ite < nmax):
        # Store the previous guess
        guess_prev = float(guess)
        # Model points as Gaussian weighted by likelihood of being a valid point
        # Nearly but not exactly one for low-sigma values
        gfit = (1 - odd_ratio) * np.exp(-0.5 * ((value - guess) ** 2 / error2))
        # Find the probability that a point is bad
        odd_bad = odd_ratio / (gfit + odd_ratio)
        # Find the probability that a point is good
        odd_good = 1 - odd_bad
        # Calculate the weights based on the probability of being good
        weights = odd_good / error2
        # Update the guess based on the weights
        if np.sum(np.isfinite(weights)) == 0:
            guess = np.nan
        else:
            guess = np.nansum(value * weights) / np.nansum(weights)
            # Work out the bulk error
            bulk_error = np.sqrt(1.0 / np.nansum(odd_good / error2))
        # Keep track of the number of iterations
        ite += 1
    # Return the guess and bulk error
    return guess, bulk_error


def odd_ratio_linfit(x: np.ndarray, y: np.ndarray, yerr: np.ndarray,
                     odd_ratio: float = 2e-4, nmax: int = 10,
                     conv_cut: float = 1e-2,
                     return_weights: bool = False
                     ) -> Tuple[float, float, float, float, Optional[np.ndarray]]:
    """
    Perform robust linear regression using a Gaussian mixture model.

    This function fits a linear model y = a + b*x to data with uncertainties,
    using the odd ratio mixture model to down-weight outliers. Each data point
    is modeled as either coming from a Gaussian distribution around the true
    line (valid measurement) or from a uniform background (outlier).

    Parameters
    ----------
    x : np.ndarray
        1D array of independent variable values.
    y : np.ndarray
        1D array of dependent variable values (measurements).
    yerr : np.ndarray
        1D array of uncertainties (standard deviations) on y values.
    odd_ratio : float, optional
        Prior probability that any given point is an outlier.
        Recommended value: 0.0002 (i.e., 1 in 5000 points is bad).
        Default is 2e-4.
    nmax : int, optional
        Maximum number of iterations. Default is 10.
    conv_cut : float, optional
        Convergence criterion: iteration stops when the relative change
        in parameters is less than conv_cut times their errors.
        Default is 1e-2.
    return_weights : bool, optional
        If True, also return the final weights (probability of being valid)
        for each data point. Default is False.

    Returns
    -------
    a : float
        Intercept of the linear fit.
    a_err : float
        Uncertainty on the intercept.
    b : float
        Slope of the linear fit.
    b_err : float
        Uncertainty on the slope.
    weights : np.ndarray, optional
        Only returned if return_weights=True. Array of weights (0 to 1)
        indicating the probability that each point is valid.

    Notes
    -----
    The algorithm extends the odd_ratio_mean approach to linear regression:

    1. Initialize with a standard weighted least squares fit
    2. For each iteration:
       a. Compute residuals from the current fit
       b. Compute Gaussian likelihood for each point being valid
       c. Compute posterior probability of each point being an outlier
       d. Re-fit using weights based on probability of being valid
    3. Iterate until convergence or maximum iterations reached

    The fit is performed using weighted least squares at each iteration,
    where the weights are updated based on the mixture model probabilities.

    For the linear model y = a + b*x, the weighted least squares solution is:

    .. math::

        b = \\frac{\\sum w_i (x_i - \\bar{x})(y_i - \\bar{y})}{\\sum w_i (x_i - \\bar{x})^2}

        a = \\bar{y} - b \\bar{x}

    where :math:`w_i = p_i / \\sigma_i^2` and :math:`p_i` is the probability
    that point i is valid.

    References
    ----------
    Artigau et al. (2022), AJ, 164, 84, Appendix A
    "Line-by-line Velocity Measurements: an Outlier-resistant Method for
    Precision Velocimetry"

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> # Generate linear data with outliers
    >>> x = np.linspace(0, 10, 50)
    >>> true_a, true_b = 2.0, 0.5
    >>> y = true_a + true_b * x + np.random.normal(0, 0.5, len(x))
    >>> yerr = np.ones(len(x)) * 0.5
    >>> # Add outliers
    >>> y[5] = 20.0
    >>> y[15] = -10.0
    >>> y[25] = 15.0
    >>> a, a_err, b, b_err = odd_ratio_linfit(x, y, yerr)[:4]
    >>> print(f"Intercept: {a:.3f} ± {a_err:.3f} (true: {true_a})")
    >>> print(f"Slope: {b:.3f} ± {b_err:.3f} (true: {true_b})")
    """
    # Deal with NaNs
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
    if np.sum(keep) < 2:
        if return_weights:
            return np.nan, np.nan, np.nan, np.nan, np.full(len(x), np.nan)
        return np.nan, np.nan, np.nan, np.nan

    # Work with clean arrays
    x_clean = x[keep]
    y_clean = y[keep]
    yerr_clean = yerr[keep]
    yerr2 = yerr_clean ** 2

    # Initialize with standard weighted least squares
    weights = 1.0 / yerr2

    # Perform initial fit
    def weighted_linfit(x_arr, y_arr, w_arr):
        """Weighted least squares linear fit."""
        sw = np.nansum(w_arr)
        sx = np.nansum(w_arr * x_arr)
        sy = np.nansum(w_arr * y_arr)
        sxx = np.nansum(w_arr * x_arr * x_arr)
        sxy = np.nansum(w_arr * x_arr * y_arr)

        delta = sw * sxx - sx * sx

        if delta == 0:
            return np.nan, np.nan, np.nan, np.nan

        a_fit = (sxx * sy - sx * sxy) / delta
        b_fit = (sw * sxy - sx * sy) / delta

        # Uncertainties
        a_err_fit = np.sqrt(sxx / delta)
        b_err_fit = np.sqrt(sw / delta)

        return a_fit, a_err_fit, b_fit, b_err_fit

    # Initial guess
    a, a_err, b, b_err = weighted_linfit(x_clean, y_clean, weights)

    if not np.isfinite(a):
        if return_weights:
            full_weights = np.full(len(x), np.nan)
            full_weights[keep] = 0.0
            return np.nan, np.nan, np.nan, np.nan, full_weights
        return np.nan, np.nan, np.nan, np.nan

    # Iterative refinement
    a_prev, b_prev = np.inf, np.inf
    ite = 0
    odd_good = np.ones(len(x_clean))

    while ite < nmax:
        # Check convergence
        if a_err > 0 and b_err > 0:
            conv_a = np.abs(a - a_prev) / a_err
            conv_b = np.abs(b - b_prev) / b_err
            if conv_a < conv_cut and conv_b < conv_cut:
                break

        a_prev, b_prev = a, b

        # Compute residuals
        model = a + b * x_clean
        residuals = y_clean - model

        # Gaussian likelihood of being valid
        gfit = (1 - odd_ratio) * np.exp(-0.5 * (residuals ** 2 / yerr2))

        # Probability of being bad
        odd_bad = odd_ratio / (gfit + odd_ratio)

        # Probability of being good
        odd_good = 1 - odd_bad

        # Updated weights
        weights = odd_good / yerr2

        # Re-fit with updated weights
        a, a_err, b, b_err = weighted_linfit(x_clean, y_clean, weights)

        if not np.isfinite(a):
            break

        ite += 1

    # Prepare output
    if return_weights:
        full_weights = np.full(len(x), np.nan)
        full_weights[keep] = odd_good
        return a, a_err, b, b_err, full_weights

    return a, a_err, b, b_err


def odd_ratio_polyfit(x: np.ndarray, y: np.ndarray, yerr: np.ndarray,
                      degree: int = 1, odd_ratio: float = 2e-4,
                      nmax: int = 10, conv_cut: float = 1e-2,
                      return_weights: bool = False):
    """
    Perform robust polynomial regression using a Gaussian mixture model.

    This is a generalization of odd_ratio_linfit to arbitrary polynomial degrees.

    Parameters
    ----------
    x : np.ndarray
        1D array of independent variable values.
    y : np.ndarray
        1D array of dependent variable values.
    yerr : np.ndarray
        1D array of uncertainties on y values.
    degree : int, optional
        Degree of the polynomial fit. Default is 1 (linear).
    odd_ratio : float, optional
        Prior probability that any given point is an outlier. Default is 2e-4.
    nmax : int, optional
        Maximum number of iterations. Default is 10.
    conv_cut : float, optional
        Convergence criterion. Default is 1e-2.
    return_weights : bool, optional
        If True, also return the final weights. Default is False.

    Returns
    -------
    coeffs : np.ndarray
        Polynomial coefficients, highest power first (same convention as np.polyfit).
    coeffs_err : np.ndarray
        Uncertainties on the coefficients.
    weights : np.ndarray, optional
        Only returned if return_weights=True.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 100)
    >>> y = 1 + 2*x + 0.5*x**2 + np.random.normal(0, 1, len(x))
    >>> yerr = np.ones(len(x))
    >>> coeffs, coeffs_err = odd_ratio_polyfit(x, y, yerr, degree=2)[:2]
    """
    # Deal with NaNs
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
    if np.sum(keep) < degree + 1:
        nan_result = np.full(degree + 1, np.nan)
        if return_weights:
            return nan_result, nan_result, np.full(len(x), np.nan)
        return nan_result, nan_result

    x_clean = x[keep]
    y_clean = y[keep]
    yerr_clean = yerr[keep]
    yerr2 = yerr_clean ** 2

    # Initial weights
    weights = 1.0 / yerr2

    # Initial fit
    coeffs = np.polyfit(x_clean, y_clean, degree, w=np.sqrt(weights))

    # Iterative refinement
    coeffs_prev = np.full_like(coeffs, np.inf)
    ite = 0
    odd_good = np.ones(len(x_clean))

    while ite < nmax:
        # Check convergence (simplified)
        if np.all(np.abs(coeffs - coeffs_prev) < conv_cut * np.abs(coeffs) + 1e-10):
            break

        coeffs_prev = coeffs.copy()

        # Compute residuals
        model = np.polyval(coeffs, x_clean)
        residuals = y_clean - model

        # Gaussian likelihood
        gfit = (1 - odd_ratio) * np.exp(-0.5 * (residuals ** 2 / yerr2))

        # Probability of being good
        odd_bad = odd_ratio / (gfit + odd_ratio)
        odd_good = 1 - odd_bad

        # Updated weights
        weights = odd_good / yerr2

        # Re-fit
        try:
            coeffs = np.polyfit(x_clean, y_clean, degree, w=np.sqrt(weights))
        except np.linalg.LinAlgError:
            break

        ite += 1

    # Estimate uncertainties via bootstrap or approximation
    # For simplicity, use the covariance matrix from weighted fit
    # This is an approximation
    try:
        _, cov = np.polyfit(x_clean, y_clean, degree, w=np.sqrt(weights), cov=True)
        coeffs_err = np.sqrt(np.diag(cov))
    except (np.linalg.LinAlgError, ValueError):
        coeffs_err = np.full(degree + 1, np.nan)

    if return_weights:
        full_weights = np.full(len(x), np.nan)
        full_weights[keep] = odd_good
        return coeffs, coeffs_err, full_weights

    return coeffs, coeffs_err
