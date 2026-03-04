# Odd Ratio Linear Fit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Robust linear regression using a Gaussian mixture model for outlier rejection.**

This package implements a robust fitting algorithm that extends the odd ratio weighted mean method from Appendix A of [Artigau et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022AJ....164...84A) to linear and polynomial regression. The algorithm iteratively down-weights data points that are likely to be outliers while preserving proper statistical uncertainties.

## 🚀 Key Features

- **Robust to outliers**: Automatically identifies and down-weights outlier measurements
- **Proper error propagation**: Returns meaningful uncertainties on fit parameters
- **No manual sigma-clipping**: Uses a probabilistic mixture model instead of arbitrary thresholds
- **Fast convergence**: Typically converges in 3-5 iterations
- **Flexible**: Works with linear fits, weighted means, and polynomial regression

## 📦 Installation

Clone and install from source:
```bash
git clone https://github.com/eartigau/odd_ratio_linfit.git
cd odd_ratio_linfit
pip install -e .
```

## 🔧 Quick Start

```python
import numpy as np
from odd_ratio_linfit import odd_ratio_linfit, odd_ratio_mean

# Generate data with outliers
x = np.linspace(0, 10, 50)
y = 2.0 + 0.5 * x + np.random.normal(0, 0.5, len(x))
yerr = np.ones(len(x)) * 0.5

# Add some outliers
y[5], y[15], y[25] = 15.0, -5.0, 12.0

# Robust linear fit
a, a_err, b, b_err = odd_ratio_linfit(x, y, yerr)[:4]
print(f"Intercept: {a:.3f} ± {a_err:.3f}")
print(f"Slope: {b:.3f} ± {b_err:.3f}")
```

## 📖 How It Works

### The Mixture Model Approach

Traditional outlier rejection methods (e.g., sigma-clipping) use hard thresholds to reject points. This can be problematic because:

1. The threshold is arbitrary
2. Points near the threshold may be incorrectly classified
3. Error estimates become biased

The **odd ratio** approach models each data point as coming from a mixture of two distributions:

1. **Valid measurements**: Gaussian distribution centered on the true value
2. **Outliers**: Uniform background distribution

The probability that point $i$ is an outlier given residual $r_i$ and uncertainty $\sigma_i$ is:

$$P(\text{outlier}_i | r_i) = \frac{f_0}{(1-f_0) \cdot \exp\left(-\frac{r_i^2}{2\sigma_i^2}\right) + f_0}$$

where $f_0$ is the prior probability that any point is an outlier (default: $2 \times 10^{-4}$).

### Iterative Algorithm

1. **Initialize** with standard weighted least squares
2. **Compute** residuals from current fit
3. **Calculate** probability of each point being valid
4. **Re-fit** using probability-weighted least squares
5. **Repeat** until convergence

## 📊 Examples and Demonstrations

### Linear Fit with Outliers

The robust fit (blue) correctly recovers the true line (dashed black) despite strong outliers, while standard weighted least squares (orange) is significantly biased.

![Linear Fit Comparison](plots/linear_fit_comparison.png)

### Robust Weighted Mean

The odd ratio method provides an accurate estimate of the mean even with 10% outliers, outperforming both standard weighted mean and median.

![Weighted Mean Comparison](plots/weighted_mean_comparison.png)

### Robustness vs Outlier Fraction

The algorithm maintains accuracy even with up to 20-25% outliers, while standard methods degrade rapidly.

![Robustness vs Outliers](plots/robustness_vs_outliers.png)

### Effect of the `odd_ratio` Parameter

Smaller values of `odd_ratio` are more conservative (fewer rejections), while larger values are more aggressive. The default value of $2 \times 10^{-4}$ works well for most cases.

![Odd Ratio Sensitivity](plots/odd_ratio_sensitivity.png)

### Polynomial Fitting

The method extends naturally to polynomial regression:

![Polynomial Fit](plots/polynomial_fit_comparison.png)

### Convergence Behavior

The algorithm typically converges within 3-5 iterations:

![Convergence](plots/convergence.png)

## 📚 API Reference

### `odd_ratio_linfit`

```python
odd_ratio_linfit(x, y, yerr, odd_ratio=2e-4, nmax=10, conv_cut=1e-2, return_weights=False)
```

Perform robust linear regression $y = a + bx$.

**Parameters:**
- `x`: Independent variable (1D array)
- `y`: Dependent variable (1D array)  
- `yerr`: Uncertainties on y (1D array)
- `odd_ratio`: Prior probability of outlier (default: 2e-4)
- `nmax`: Maximum iterations (default: 10)
- `conv_cut`: Convergence criterion (default: 1e-2)
- `return_weights`: If True, also return point weights

**Returns:**
- `a, a_err`: Intercept and uncertainty
- `b, b_err`: Slope and uncertainty
- `weights` (optional): Probability each point is valid

### `odd_ratio_mean`

```python
odd_ratio_mean(value, error, odd_ratio=2e-4, nmax=10, conv_cut=1e-2)
```

Compute robust weighted mean.

**Parameters:**
- `value`: Values to average (1D array)
- `error`: Uncertainties on values (1D array)
- `odd_ratio`: Prior probability of outlier
- `nmax`: Maximum iterations
- `conv_cut`: Convergence criterion

**Returns:**
- `mean`: Robust weighted mean
- `error`: Uncertainty on the mean

### `odd_ratio_polyfit`

```python
odd_ratio_polyfit(x, y, yerr, degree=1, odd_ratio=2e-4, nmax=10, conv_cut=1e-2, return_weights=False)
```

Perform robust polynomial regression.

**Parameters:**
- `x, y, yerr`: Data arrays
- `degree`: Polynomial degree (default: 1)
- `odd_ratio, nmax, conv_cut`: Same as above
- `return_weights`: If True, also return weights

**Returns:**
- `coeffs`: Polynomial coefficients (highest power first)
- `coeffs_err`: Uncertainties on coefficients
- `weights` (optional): Point weights

## 🎯 When to Use This Method

✅ **Good use cases:**
- Data with occasional strong outliers
- Radial velocity measurements
- Photometric time series
- Any measurement with heteroscedastic errors
- When you need proper uncertainty estimates

⚠️ **Limitations:**
- Assumes outliers are rare (< 20-30% of data)
- Assumes Gaussian errors for valid points
- May not work well if outliers dominate

## 📜 Citation

If you use this method in your research, please cite:

```bibtex
@ARTICLE{2022AJ....164...84A,
       author = {{Artigau}, {\'E}tienne and {Cadieux}, Charles and {Cook}, Neil J. and
         {Doyon}, Ren{\'e} and {Vandal}, Thomas and {Donati}, Jean-Fran{\c{c}}ois and
         {Moutou}, Claire and {Delfosse}, Xavier and {Fouqu{\'e}}, Pascal and
         {Collier Cameron}, Andrew and others},
        title = "{Line-by-line Velocity Measurements: an Outlier-resistant Method for Precision Velocimetry}",
      journal = {\aj},
         year = 2022,
        month = sep,
       volume = {164},
       number = {3},
          eid = {84},
        pages = {84},
          doi = {10.3847/1538-3881/ac7f2c},
archivePrefix = {arXiv},
       eprint = {2207.13524},
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

- **Author**: Étienne Artigau
- **Email**: etienne.artigau@umontreal.ca
- **Institution**: Université de Montréal / Institut de Recherche sur les Exoplanètes (iREx)
