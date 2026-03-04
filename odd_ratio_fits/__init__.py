"""
Odd Ratio Fits - Robust fitting using a mixture model.

This package implements a robust linear fitting algorithm based on a Gaussian
mixture model approach, extending the odd_ratio_mean method from Artigau et al.
(2021). The algorithm iteratively weights data points based on their probability
of being valid measurements versus outliers.

Reference:
    Artigau et al. (2022), AJ, 164, 84, Appendix A
    "Line-by-line Velocity Measurements: an Outlier-resistant Method for 
    Precision Velocimetry"
"""

from .core import odd_ratio_linfit, odd_ratio_mean, odd_ratio_polyfit

__version__ = "1.0.0"
__author__ = "Étienne Artigau, Neil J. Cook, Charles Cadieux, Atanas K. Stefanov, René Doyon"
__email__ = "etienne.artigau@umontreal.ca"

__all__ = ["odd_ratio_linfit", "odd_ratio_mean", "odd_ratio_polyfit", "__version__"]
