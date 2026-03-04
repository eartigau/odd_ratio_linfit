#!/usr/bin/env python
"""
Demo script for odd_ratio_linfit.

This script demonstrates the robust linear fitting capabilities of the
odd_ratio_linfit package, comparing it with standard weighted least squares
fitting in the presence of outliers.

Run this script to generate the demonstration plots:
    python demo.py

The plots will be saved in the 'plots/' directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import our package
from odd_ratio_linfit import odd_ratio_linfit, odd_ratio_mean, odd_ratio_polyfit

# Create plots directory
PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Style settings for publication-quality plots
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


def standard_weighted_linfit(x, y, yerr):
    """Standard weighted least squares linear fit."""
    w = 1.0 / yerr**2
    sw = np.sum(w)
    sx = np.sum(w * x)
    sy = np.sum(w * y)
    sxx = np.sum(w * x * x)
    sxy = np.sum(w * x * y)
    
    delta = sw * sxx - sx * sx
    a = (sxx * sy - sx * sxy) / delta
    b = (sw * sxy - sx * sy) / delta
    
    a_err = np.sqrt(sxx / delta)
    b_err = np.sqrt(sw / delta)
    
    return a, a_err, b, b_err


def demo_linear_fit_comparison():
    """
    Demonstrate comparison between standard and robust linear fitting.
    """
    print("=" * 60)
    print("Demo 1: Linear Fit Comparison with Outliers")
    print("=" * 60)
    
    # Generate synthetic data
    n_points = 50
    true_a, true_b = 2.0, 0.5  # True intercept and slope
    noise_level = 0.5
    
    x = np.linspace(0, 10, n_points)
    y_true = true_a + true_b * x
    y = y_true + np.random.normal(0, noise_level, n_points)
    yerr = np.ones(n_points) * noise_level
    
    # Add outliers (10% of data)
    n_outliers = 5
    outlier_idx = np.random.choice(n_points, n_outliers, replace=False)
    y[outlier_idx] = y[outlier_idx] + np.random.choice([-1, 1], n_outliers) * np.random.uniform(5, 15, n_outliers)
    
    # Standard fit
    a_std, a_std_err, b_std, b_std_err = standard_weighted_linfit(x, y, yerr)
    
    # Robust fit
    a_rob, a_rob_err, b_rob, b_rob_err, weights = odd_ratio_linfit(
        x, y, yerr, return_weights=True
    )
    
    print(f"\nTrue values: a = {true_a:.3f}, b = {true_b:.3f}")
    print(f"Standard fit: a = {a_std:.3f} ± {a_std_err:.3f}, b = {b_std:.3f} ± {b_std_err:.3f}")
    print(f"Robust fit:   a = {a_rob:.3f} ± {a_rob_err:.3f}, b = {b_rob:.3f} ± {b_rob_err:.3f}")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: Data and fits
    ax = axes[0]
    
    # Plot data points, colored by weight
    outlier_mask = weights < 0.5
    ax.errorbar(x[~outlier_mask], y[~outlier_mask], yerr=yerr[~outlier_mask], 
                fmt='o', color='#2E86AB', markersize=8, alpha=0.8, 
                label='Valid points', capsize=3, elinewidth=1.5)
    ax.errorbar(x[outlier_mask], y[outlier_mask], yerr=yerr[outlier_mask], 
                fmt='s', color='#E94F37', markersize=10, alpha=0.8, 
                label=f'Outliers (weight < 0.5)', capsize=3, elinewidth=1.5)
    
    # Plot fits
    x_fit = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_fit, true_a + true_b * x_fit, 'k--', lw=2, label='True line')
    ax.plot(x_fit, a_std + b_std * x_fit, '#F28123', lw=2.5, label='Standard WLS')
    ax.plot(x_fit, a_rob + b_rob * x_fit, '#2E86AB', lw=2.5, label='Robust (odd ratio)')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Linear Fit Comparison')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Right panel: Weights
    ax = axes[1]
    colors = plt.cm.RdYlGn(weights)
    scatter = ax.scatter(x, y, c=weights, cmap='RdYlGn', s=100, 
                         edgecolors='black', linewidth=0.5, vmin=0, vmax=1)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Weight (probability of being valid)')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Point Weights from Mixture Model')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'linear_fit_comparison.png')
    plt.savefig(PLOT_DIR / 'linear_fit_comparison.pdf')
    print(f"\nSaved: {PLOT_DIR / 'linear_fit_comparison.png'}")
    
    return fig


def demo_weighted_mean():
    """
    Demonstrate the odd_ratio_mean function for robust averaging.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Robust Weighted Mean")
    print("=" * 60)
    
    # Generate data with outliers
    n_points = 100
    true_value = 10.0
    noise_level = 1.0
    
    values = np.random.normal(true_value, noise_level, n_points)
    errors = np.ones(n_points) * noise_level
    
    # Add outliers
    n_outliers = 10
    outlier_idx = np.random.choice(n_points, n_outliers, replace=False)
    values[outlier_idx] = values[outlier_idx] + np.random.choice([-1, 1], n_outliers) * np.random.uniform(10, 30, n_outliers)
    
    # Standard weighted mean
    w = 1.0 / errors**2
    std_mean = np.sum(values * w) / np.sum(w)
    std_err = np.sqrt(1.0 / np.sum(w))
    
    # Robust mean
    rob_mean, rob_err = odd_ratio_mean(values, errors)
    
    # Simple median for comparison
    median_val = np.median(values)
    
    print(f"\nTrue value: {true_value:.3f}")
    print(f"Standard weighted mean: {std_mean:.3f} ± {std_err:.3f}")
    print(f"Robust (odd ratio) mean: {rob_mean:.3f} ± {rob_err:.3f}")
    print(f"Median: {median_val:.3f}")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: Histogram
    ax = axes[0]
    bins = np.linspace(values.min(), values.max(), 30)
    ax.hist(values, bins=bins, alpha=0.7, color='#2E86AB', edgecolor='black')
    ax.axvline(true_value, color='black', linestyle='--', lw=2, label=f'True = {true_value:.2f}')
    ax.axvline(std_mean, color='#F28123', linestyle='-', lw=2, label=f'Standard = {std_mean:.2f}')
    ax.axvline(rob_mean, color='#28A745', linestyle='-', lw=2, label=f'Robust = {rob_mean:.2f}')
    ax.axvline(median_val, color='#6C757D', linestyle=':', lw=2, label=f'Median = {median_val:.2f}')
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.set_title('Distribution with Outliers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right panel: Error comparison
    ax = axes[1]
    methods = ['Standard\nWeighted Mean', 'Robust\n(Odd Ratio)', 'Median']
    estimates = [std_mean, rob_mean, median_val]
    method_errors = [std_err, rob_err, noise_level / np.sqrt(n_points)]  # MAD-based for median
    
    # Calculate bias from true value
    biases = [np.abs(est - true_value) for est in estimates]
    
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, biases, color=['#F28123', '#28A745', '#6C757D'], 
                  edgecolor='black', alpha=0.8)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_ylabel('|Bias| from True Value')
    ax.set_title('Accuracy Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, bias in zip(bars, biases):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{bias:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'weighted_mean_comparison.png')
    plt.savefig(PLOT_DIR / 'weighted_mean_comparison.pdf')
    print(f"\nSaved: {PLOT_DIR / 'weighted_mean_comparison.png'}")
    
    return fig


def demo_varying_outlier_fraction():
    """
    Demonstrate robustness across varying outlier fractions.
    """
    print("\n" + "=" * 60)
    print("Demo 3: Robustness vs Outlier Fraction")
    print("=" * 60)
    
    n_points = 100
    true_a, true_b = 2.0, 0.5
    noise_level = 0.5
    outlier_fractions = np.linspace(0, 0.3, 16)  # 0% to 30% outliers
    n_trials = 50
    
    std_bias_a = []
    std_bias_b = []
    rob_bias_a = []
    rob_bias_b = []
    
    for frac in outlier_fractions:
        n_outliers = int(frac * n_points)
        trial_std_a, trial_std_b = [], []
        trial_rob_a, trial_rob_b = [], []
        
        for _ in range(n_trials):
            x = np.linspace(0, 10, n_points)
            y = true_a + true_b * x + np.random.normal(0, noise_level, n_points)
            yerr = np.ones(n_points) * noise_level
            
            if n_outliers > 0:
                outlier_idx = np.random.choice(n_points, n_outliers, replace=False)
                y[outlier_idx] += np.random.choice([-1, 1], n_outliers) * np.random.uniform(5, 15, n_outliers)
            
            a_std, _, b_std, _ = standard_weighted_linfit(x, y, yerr)
            a_rob, _, b_rob, _ = odd_ratio_linfit(x, y, yerr)[:4]
            
            trial_std_a.append(a_std - true_a)
            trial_std_b.append(b_std - true_b)
            trial_rob_a.append(a_rob - true_a)
            trial_rob_b.append(b_rob - true_b)
        
        std_bias_a.append(np.std(trial_std_a))
        std_bias_b.append(np.std(trial_std_b))
        rob_bias_a.append(np.std(trial_rob_a))
        rob_bias_b.append(np.std(trial_rob_b))
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.plot(outlier_fractions * 100, std_bias_a, 'o-', color='#F28123', 
            lw=2, markersize=8, label='Standard WLS')
    ax.plot(outlier_fractions * 100, rob_bias_a, 's-', color='#2E86AB', 
            lw=2, markersize=8, label='Robust (odd ratio)')
    ax.set_xlabel('Outlier Fraction (%)')
    ax.set_ylabel('RMS Error in Intercept')
    ax.set_title('Intercept Recovery')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(outlier_fractions * 100, std_bias_b, 'o-', color='#F28123', 
            lw=2, markersize=8, label='Standard WLS')
    ax.plot(outlier_fractions * 100, rob_bias_b, 's-', color='#2E86AB', 
            lw=2, markersize=8, label='Robust (odd ratio)')
    ax.set_xlabel('Outlier Fraction (%)')
    ax.set_ylabel('RMS Error in Slope')
    ax.set_title('Slope Recovery')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'robustness_vs_outliers.png')
    plt.savefig(PLOT_DIR / 'robustness_vs_outliers.pdf')
    print(f"\nSaved: {PLOT_DIR / 'robustness_vs_outliers.png'}")
    
    return fig


def demo_odd_ratio_sensitivity():
    """
    Demonstrate the effect of the odd_ratio parameter.
    """
    print("\n" + "=" * 60)
    print("Demo 4: Sensitivity to odd_ratio Parameter")
    print("=" * 60)
    
    n_points = 50
    true_a, true_b = 2.0, 0.5
    noise_level = 0.5
    
    x = np.linspace(0, 10, n_points)
    y = true_a + true_b * x + np.random.normal(0, noise_level, n_points)
    yerr = np.ones(n_points) * noise_level
    
    # Add outliers
    outlier_idx = [5, 15, 25, 35, 45]
    y[outlier_idx] = y[outlier_idx] + np.array([10, -8, 12, -10, 8])
    
    odd_ratios = [1e-6, 1e-5, 1e-4, 2e-4, 1e-3, 1e-2, 1e-1]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, odd_ratio in enumerate(odd_ratios):
        ax = axes[i]
        
        a, a_err, b, b_err, weights = odd_ratio_linfit(
            x, y, yerr, odd_ratio=odd_ratio, return_weights=True
        )
        
        # Plot data colored by weight
        scatter = ax.scatter(x, y, c=weights, cmap='RdYlGn', s=80, 
                            edgecolors='black', linewidth=0.5, vmin=0, vmax=1)
        
        # Plot fits
        x_fit = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_fit, true_a + true_b * x_fit, 'k--', lw=1.5, alpha=0.7)
        ax.plot(x_fit, a + b * x_fit, '#2E86AB', lw=2)
        
        ax.set_title(f'odd_ratio = {odd_ratio:.0e}\na={a:.2f}, b={b:.2f}', fontsize=10)
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Use last subplot for colorbar
    axes[-1].axis('off')
    cbar = fig.colorbar(scatter, ax=axes[-1], orientation='vertical', 
                        fraction=0.8, aspect=20)
    cbar.set_label('Weight', fontsize=12)
    
    plt.suptitle('Effect of odd_ratio Parameter on Fit\n(dashed = true line, solid = fit)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'odd_ratio_sensitivity.png')
    plt.savefig(PLOT_DIR / 'odd_ratio_sensitivity.pdf')
    print(f"\nSaved: {PLOT_DIR / 'odd_ratio_sensitivity.png'}")
    
    return fig


def demo_polynomial_fit():
    """
    Demonstrate polynomial fitting with outliers.
    """
    print("\n" + "=" * 60)
    print("Demo 5: Polynomial Fit with Outliers")
    print("=" * 60)
    
    n_points = 80
    true_coeffs = [0.05, -0.5, 3.0]  # quadratic: 0.05*x^2 - 0.5*x + 3
    noise_level = 0.5
    
    x = np.linspace(0, 10, n_points)
    y_true = np.polyval(true_coeffs, x)
    y = y_true + np.random.normal(0, noise_level, n_points)
    yerr = np.ones(n_points) * noise_level
    
    # Add outliers
    n_outliers = 8
    outlier_idx = np.random.choice(n_points, n_outliers, replace=False)
    y[outlier_idx] += np.random.choice([-1, 1], n_outliers) * np.random.uniform(3, 8, n_outliers)
    
    # Standard polynomial fit
    std_coeffs = np.polyfit(x, y, 2, w=1/yerr)
    
    # Robust polynomial fit
    rob_coeffs, rob_coeffs_err, weights = odd_ratio_polyfit(
        x, y, yerr, degree=2, return_weights=True
    )
    
    print(f"\nTrue coefficients: {true_coeffs}")
    print(f"Standard fit: [{std_coeffs[0]:.4f}, {std_coeffs[1]:.4f}, {std_coeffs[2]:.4f}]")
    print(f"Robust fit: [{rob_coeffs[0]:.4f}, {rob_coeffs[1]:.4f}, {rob_coeffs[2]:.4f}]")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: Data and fits
    ax = axes[0]
    
    outlier_mask = weights < 0.5
    ax.errorbar(x[~outlier_mask], y[~outlier_mask], yerr=yerr[~outlier_mask], 
                fmt='o', color='#2E86AB', markersize=7, alpha=0.7, 
                label='Valid points', capsize=2)
    ax.errorbar(x[outlier_mask], y[outlier_mask], yerr=yerr[outlier_mask], 
                fmt='s', color='#E94F37', markersize=9, alpha=0.8, 
                label='Outliers', capsize=2)
    
    x_fit = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_fit, np.polyval(true_coeffs, x_fit), 'k--', lw=2, label='True curve')
    ax.plot(x_fit, np.polyval(std_coeffs, x_fit), '#F28123', lw=2.5, label='Standard fit')
    ax.plot(x_fit, np.polyval(rob_coeffs, x_fit), '#2E86AB', lw=2.5, label='Robust fit')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Quadratic Fit Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right panel: Coefficient errors
    ax = axes[1]
    
    coeff_names = ['$a_2$ (x²)', '$a_1$ (x)', '$a_0$ (const)']
    x_pos = np.arange(len(coeff_names))
    width = 0.35
    
    std_errors = np.abs(std_coeffs - true_coeffs)
    rob_errors = np.abs(rob_coeffs - true_coeffs)
    
    bars1 = ax.bar(x_pos - width/2, std_errors, width, label='Standard', 
                   color='#F28123', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, rob_errors, width, label='Robust', 
                   color='#2E86AB', edgecolor='black', alpha=0.8)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(coeff_names)
    ax.set_ylabel('|Error| from True Value')
    ax.set_title('Coefficient Recovery Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'polynomial_fit_comparison.png')
    plt.savefig(PLOT_DIR / 'polynomial_fit_comparison.pdf')
    print(f"\nSaved: {PLOT_DIR / 'polynomial_fit_comparison.png'}")
    
    return fig


def demo_convergence():
    """
    Demonstrate the iterative convergence of the algorithm.
    """
    print("\n" + "=" * 60)
    print("Demo 6: Algorithm Convergence")
    print("=" * 60)
    
    n_points = 50
    true_a, true_b = 2.0, 0.5
    noise_level = 0.5
    
    x = np.linspace(0, 10, n_points)
    y = true_a + true_b * x + np.random.normal(0, noise_level, n_points)
    yerr = np.ones(n_points) * noise_level
    
    # Add outliers
    outlier_idx = [5, 15, 25, 35, 45]
    y[outlier_idx] += np.array([10, -8, 12, -10, 8])
    
    # Track convergence by running single iterations
    a_history = []
    b_history = []
    
    # Manual iteration tracking
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
    x_clean, y_clean, yerr_clean = x[keep], y[keep], yerr[keep]
    yerr2 = yerr_clean ** 2
    weights = 1.0 / yerr2
    odd_ratio = 2e-4
    
    def weighted_linfit_internal(x_arr, y_arr, w_arr):
        sw = np.nansum(w_arr)
        sx = np.nansum(w_arr * x_arr)
        sy = np.nansum(w_arr * y_arr)
        sxx = np.nansum(w_arr * x_arr * x_arr)
        sxy = np.nansum(w_arr * x_arr * y_arr)
        delta = sw * sxx - sx * sx
        a_fit = (sxx * sy - sx * sxy) / delta
        b_fit = (sw * sxy - sx * sy) / delta
        return a_fit, b_fit
    
    a, b = weighted_linfit_internal(x_clean, y_clean, weights)
    a_history.append(a)
    b_history.append(b)
    
    for _ in range(15):
        model = a + b * x_clean
        residuals = y_clean - model
        gfit = (1 - odd_ratio) * np.exp(-0.5 * (residuals ** 2 / yerr2))
        odd_bad = odd_ratio / (gfit + odd_ratio)
        odd_good = 1 - odd_bad
        weights = odd_good / yerr2
        a, b = weighted_linfit_internal(x_clean, y_clean, weights)
        a_history.append(a)
        b_history.append(b)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    iterations = np.arange(len(a_history))
    
    ax = axes[0]
    ax.plot(iterations, a_history, 'o-', color='#2E86AB', lw=2, markersize=8)
    ax.axhline(true_a, color='black', linestyle='--', lw=2, label=f'True = {true_a}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Intercept (a)')
    ax.set_title('Intercept Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(iterations, b_history, 'o-', color='#2E86AB', lw=2, markersize=8)
    ax.axhline(true_b, color='black', linestyle='--', lw=2, label=f'True = {true_b}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Slope (b)')
    ax.set_title('Slope Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'convergence.png')
    plt.savefig(PLOT_DIR / 'convergence.pdf')
    print(f"\nSaved: {PLOT_DIR / 'convergence.png'}")
    
    return fig


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("ODD RATIO LINEAR FIT - DEMONSTRATION")
    print("=" * 60)
    print(f"\nPlots will be saved to: {PLOT_DIR.absolute()}\n")
    
    demo_linear_fit_comparison()
    demo_weighted_mean()
    demo_varying_outlier_fraction()
    demo_odd_ratio_sensitivity()
    demo_polynomial_fit()
    demo_convergence()
    
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)
    print(f"\nPlots saved in: {PLOT_DIR.absolute()}")
    print("\nGenerated files:")
    for f in sorted(PLOT_DIR.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
