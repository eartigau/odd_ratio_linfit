#!/usr/bin/env python
"""
Demo script for odd_ratio_fits.

This script demonstrates the robust fitting capabilities of the
odd_ratio_fits package, comparing it with standard weighted least squares
fitting in the presence of outliers.

Run this script to generate the demonstration plots:
    python demo.py

The plots will be saved in the 'plots/' directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import our package
import odd_ratio_fits as orf

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
    
    # Generate synthetic data with heteroscedastic errors
    # (See 'Proper Handling of Heteroscedastic Uncertainties' section)
    n_points = 1000
    true_a, true_b = 2.0, 0.5  # True intercept and slope
    
    x = np.linspace(0, 10, n_points)
    y_true = true_a + true_b * x
    
    # Heteroscedastic errors: varying from 0.3 to 1.5
    yerr = 0.3 + 1.2 * np.random.random(n_points)
    y = y_true + np.random.normal(0, 1, n_points) * yerr
    
    # Add outliers: uniform between -15σ and +15σ (some overlap with main distribution)
    n_outliers = 20
    outlier_idx = np.random.choice(n_points, n_outliers, replace=False)
    y[outlier_idx] = y[outlier_idx] + yerr[outlier_idx] * np.random.uniform(-15, 15, n_outliers)
    
    # Standard fit
    a_std, a_std_err, b_std, b_std_err = standard_weighted_linfit(x, y, yerr)
    
    # Robust fit
    a_rob, a_rob_err, b_rob, b_rob_err, weights = orf.linear(
        x, y, yerr, return_weights=True
    )
    
    print(f"\nTrue values: a = {true_a:.3f}, b = {true_b:.3f}")
    print(f"Standard fit: a = {a_std:.3f} ± {a_std_err:.3f}, b = {b_std:.3f} ± {b_std_err:.3f}")
    print(f"Robust fit:   a = {a_rob:.3f} ± {a_rob_err:.3f}, b = {b_rob:.3f} ± {b_rob_err:.3f}")
    
    # Create figure - vertical layout
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Top panel: Data and fits
    ax = axes[0]
    
    # Plot data points, colored by weight
    outlier_mask = weights < 0.5
    ax.errorbar(x[~outlier_mask], y[~outlier_mask], yerr=yerr[~outlier_mask], 
                fmt='o', color='#2E86AB', markersize=8, alpha=0.5, 
                label='Valid points', capsize=3, elinewidth=1.5)
    ax.errorbar(x[outlier_mask], y[outlier_mask], yerr=yerr[outlier_mask], 
                fmt='s', color='#E94F37', markersize=10, alpha=0.5, 
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
    
    # Bottom panel: Weights
    ax = axes[1]
    scatter = ax.scatter(x, y, c=weights, cmap='RdYlGn', s=100, alpha=0.5,
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
    Demonstrate the orf.mean function for robust averaging with Monte Carlo validation.
    Shows comparison between ORF and naive weighted mean with histograms.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Robust Weighted Mean - Monte Carlo Validation")
    print("=" * 60)
    
    # Monte Carlo parameters
    n_points = 10000
    true_value = 10.0
    noise_level = 1.0
    n_realizations = 1000
    outlier_fraction = 0.10  # 10% outliers
    
    # Expected uncertainty from first principles: sigma / sqrt(N)
    expected_err = noise_level / np.sqrt(n_points)
    
    # Storage for results
    rob_means = []
    rob_errs = []
    naive_means = []
    naive_errs = []
    
    print(f"\nRunning {n_realizations} Monte Carlo realizations...")
    print(f"Each realization: {n_points} points, {int(outlier_fraction*100)}% outliers")
    print(f"Expected uncertainty (σ/√N): {expected_err:.4f}")
    
    for i in range(n_realizations):
        # Generate data
        values = np.random.normal(true_value, noise_level, n_points)
        errors = np.ones(n_points) * noise_level
        
        # Add outliers: uniform between -15σ and +15σ (some overlap with main distribution)
        n_outliers = int(outlier_fraction * n_points)
        outlier_idx = np.random.choice(n_points, n_outliers, replace=False)
        values[outlier_idx] += errors[outlier_idx] * np.random.uniform(-15, 15, n_outliers)
        
        # Robust mean
        rob_mean, rob_err = orf.mean(values, errors)
        rob_means.append(rob_mean)
        rob_errs.append(rob_err)
        
        # Naive weighted mean
        w = 1.0 / errors**2
        naive_mean = np.sum(values * w) / np.sum(w)
        naive_err = np.sqrt(1.0 / np.sum(w))
        naive_means.append(naive_mean)
        naive_errs.append(naive_err)
    
    rob_means = np.array(rob_means)
    rob_errs = np.array(rob_errs)
    naive_means = np.array(naive_means)
    naive_errs = np.array(naive_errs)
    
    # Compute statistics
    rob_scatter = np.std(rob_means)
    rob_bias = np.mean(rob_means) - true_value
    naive_scatter = np.std(naive_means)
    naive_bias = np.mean(naive_means) - true_value
    
    print(f"\nResults from {n_realizations} realizations:")
    print(f"\n{'='*50}")
    print("ROBUST (ORF) METHOD:")
    print(f"  Mean recovered: {np.mean(rob_means):.4f}")
    print(f"  Bias from truth: {rob_bias:.4f}")
    print(f"  Actual scatter: {rob_scatter:.4f}")
    print(f"  Mean reported error: {np.mean(rob_errs):.4f}")
    print(f"  Ratio (scatter/error): {rob_scatter/np.mean(rob_errs):.2f}")
    print(f"\n{'='*50}")
    print("NAIVE WEIGHTED MEAN:")
    print(f"  Mean recovered: {np.mean(naive_means):.4f}")
    print(f"  Bias from truth: {naive_bias:.4f}")
    print(f"  Actual scatter: {naive_scatter:.4f}")
    print(f"  Mean reported error: {np.mean(naive_errs):.4f}")
    print(f"  Ratio (scatter/error): {naive_scatter/np.mean(naive_errs):.2f}")
    
    # Create figure with histograms - vertical layout
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Top panel: Overlaid histograms of recovered means
    ax = axes[0]
    
    # Determine common bin range
    all_means = np.concatenate([rob_means, naive_means])
    bin_min = min(all_means.min(), true_value - 4*naive_scatter)
    bin_max = max(all_means.max(), true_value + 4*naive_scatter)
    bins = np.linspace(bin_min, bin_max, 100)  # Finer bins
    
    # Plot histograms
    ax.hist(naive_means, bins=bins, density=True, alpha=0.5, color='#E94F37',
            edgecolor='#E94F37', linewidth=1.5, label=f'Naive (σ={naive_scatter:.4f})')
    ax.hist(rob_means, bins=bins, density=True, alpha=0.5, color='#2E86AB',
            edgecolor='#2E86AB', linewidth=1.5, label=f'ORF (σ={rob_scatter:.4f})')
    
    # Overlay theoretical Gaussians
    x_gauss = np.linspace(bin_min, bin_max, 200)
    # Theoretical Gaussian for ORF (centered on truth, width = expected_err)
    gauss_theory = np.exp(-(x_gauss - true_value)**2 / (2 * expected_err**2)) / (expected_err * np.sqrt(2*np.pi))
    ax.plot(x_gauss, gauss_theory, 'k-', lw=2, label=f'Theory N(μ, σ/√N)\nσ/√N = {expected_err:.4f}')
    
    ax.axvline(true_value, color='black', linestyle='--', lw=2, alpha=0.7)
    ax.set_xlabel('Recovered Mean')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Distribution of Recovered Means ({n_realizations} realizations)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Bottom panel: Comparison bar chart
    ax = axes[1]
    
    methods = ['Naive\nWeighted Mean', 'Robust\n(ORF)', 'Theory\n(σ/√N)']
    x_pos = np.arange(len(methods))
    width = 0.35
    
    scatters = [naive_scatter, rob_scatter, expected_err]
    biases = [np.abs(naive_bias), np.abs(rob_bias), 0]
    
    bars1 = ax.bar(x_pos - width/2, scatters, width, label='Scatter (std)', 
                   color=['#E94F37', '#2E86AB', '#333333'], alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x_pos + width/2, biases, width, label='|Bias| (systematic offset)',
                   color=['#E94F37', '#2E86AB', '#333333'], alpha=0.4, edgecolor='black', hatch='///')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Value')
    ax.set_title('Scatter and Bias Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, scatters):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, biases):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'Weighted Mean: {n_points} points, {int(outlier_fraction*100)}% outliers', 
                 fontsize=14, y=1.02)
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
    
    n_points = 1000
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
            a_rob, _, b_rob, _ = orf.linear(x, y, yerr)
            
            trial_std_a.append(a_std - true_a)
            trial_std_b.append(b_std - true_b)
            trial_rob_a.append(a_rob - true_a)
            trial_rob_b.append(b_rob - true_b)
        
        std_bias_a.append(np.std(trial_std_a))
        std_bias_b.append(np.std(trial_std_b))
        rob_bias_a.append(np.std(trial_rob_a))
        rob_bias_b.append(np.std(trial_rob_b))
    
    # Create figure - vertical layout
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
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
    
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    axes = axes.flatten()
    
    for i, odd_ratio in enumerate(odd_ratios):
        ax = axes[i]
        
        a, a_err, b, b_err, weights = orf.linear(
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
    Demonstrate polynomial fitting with outliers using Monte Carlo validation.
    """
    print("\n" + "=" * 60)
    print("Demo 5: Polynomial Fit with Outliers - Monte Carlo Validation")
    print("=" * 60)
    
    n_points = 80
    true_coeffs = np.array([0.05, -0.5, 3.0])  # quadratic: 0.05*x^2 - 0.5*x + 3
    noise_level = 0.5
    n_realizations = 500
    outlier_fraction = 0.10
    
    x = np.linspace(0, 10, n_points)
    yerr = np.ones(n_points) * noise_level
    
    # Compute theoretical errors from first principles
    # For WLS: Cov(coeffs) = (X^T W X)^(-1) where W = diag(1/sigma^2)
    # Design matrix for quadratic: columns are [x^2, x, 1] (highest power first like polyfit)
    X = np.column_stack([x**2, x, np.ones_like(x)])
    W = np.diag(1.0 / yerr**2)
    XtWX = X.T @ W @ X
    cov_theory = np.linalg.inv(XtWX)
    theory_stds = np.sqrt(np.diag(cov_theory))
    
    print(f"\nTheoretical errors (from covariance matrix, no outliers):")
    print(f"  σ(a₂) = {theory_stds[0]:.4f}")
    print(f"  σ(a₁) = {theory_stds[1]:.4f}")
    print(f"  σ(a₀) = {theory_stds[2]:.4f}")
    
    # Storage for MC results
    rob_coeffs_all = []
    std_coeffs_all = []
    
    print(f"\nRunning {n_realizations} Monte Carlo realizations...")
    
    for i in range(n_realizations):
        y_true = np.polyval(true_coeffs, x)
        y = y_true + np.random.normal(0, noise_level, n_points)
        
        # Add outliers
        n_outliers = int(outlier_fraction * n_points)
        outlier_idx = np.random.choice(n_points, n_outliers, replace=False)
        y[outlier_idx] += np.random.choice([-1, 1], n_outliers) * np.random.uniform(3, 8, n_outliers)
        
        # Standard polynomial fit
        std_coeffs = np.polyfit(x, y, 2, w=1/yerr)
        std_coeffs_all.append(std_coeffs)
        
        # Robust polynomial fit
        rob_coeffs, _ = orf.polyfit(x, y, yerr, degree=2)
        rob_coeffs_all.append(rob_coeffs)
    
    rob_coeffs_all = np.array(rob_coeffs_all)
    std_coeffs_all = np.array(std_coeffs_all)
    
    # Compute statistics
    rob_means = np.mean(rob_coeffs_all, axis=0)
    rob_stds = np.std(rob_coeffs_all, axis=0)
    std_means = np.mean(std_coeffs_all, axis=0)
    std_stds = np.std(std_coeffs_all, axis=0)
    
    coeff_names = ['a₂ (x²)', 'a₁ (x)', 'a₀ (const)']
    
    print(f"\nResults from {n_realizations} realizations:")
    print(f"\n{'Parameter':<12} {'True':>8} {'Theory σ':>10} {'ORF σ':>10} {'Naive σ':>10} {'ORF/Theory':>12}")
    print("-" * 70)
    for i, name in enumerate(coeff_names):
        ratio = rob_stds[i] / theory_stds[i]
        print(f"{name:<12} {true_coeffs[i]:>8.4f} {theory_stds[i]:>10.4f} {rob_stds[i]:>10.4f} {std_stds[i]:>10.4f} {ratio:>12.2f}")
    
    # Generate one example realization for the fit plot
    np.random.seed(42)
    y_example = np.polyval(true_coeffs, x) + np.random.normal(0, noise_level, n_points)
    n_outliers = int(outlier_fraction * n_points)
    outlier_idx = np.random.choice(n_points, n_outliers, replace=False)
    y_example[outlier_idx] += np.random.choice([-1, 1], n_outliers) * np.random.uniform(3, 8, n_outliers)
    
    rob_coeffs_ex, rob_coeffs_err_ex, weights = orf.polyfit(x, y_example, yerr, degree=2, return_weights=True)
    std_coeffs_ex = np.polyfit(x, y_example, 2, w=1/yerr)
    
    # Figure 1: Fit comparison with blue/red points
    fig1, ax = plt.subplots(figsize=(10, 6))
    
    outlier_mask = weights < 0.5
    ax.errorbar(x[~outlier_mask], y_example[~outlier_mask], yerr=yerr[~outlier_mask], 
                fmt='o', color='#2E86AB', markersize=7, alpha=0.5, 
                label='Valid points', capsize=2)
    ax.errorbar(x[outlier_mask], y_example[outlier_mask], yerr=yerr[outlier_mask], 
                fmt='s', color='#E94F37', markersize=9, alpha=0.5, 
                label='Outliers', capsize=2)
    
    x_fit = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_fit, np.polyval(true_coeffs, x_fit), 'k--', lw=2, label='True curve')
    ax.plot(x_fit, np.polyval(std_coeffs_ex, x_fit), '#F28123', lw=2.5, label='Standard fit')
    ax.plot(x_fit, np.polyval(rob_coeffs_ex, x_fit), '#2E86AB', lw=2.5, label='Robust fit')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Quadratic Fit Comparison ({n_points} points, {int(outlier_fraction*100)}% outliers)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'polynomial_fit_comparison.png')
    plt.savefig(PLOT_DIR / 'polynomial_fit_comparison.pdf')
    print(f"\nSaved: {PLOT_DIR / 'polynomial_fit_comparison.png'}")
    
    # Figure 2: Histograms for each coefficient with theoretical Gaussian
    fig2, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    for i, (ax, name) in enumerate(zip(axes, coeff_names)):
        # Determine bin range
        all_vals = np.concatenate([rob_coeffs_all[:, i], std_coeffs_all[:, i]])
        bin_min = min(all_vals.min(), true_coeffs[i] - 4*std_stds[i])
        bin_max = max(all_vals.max(), true_coeffs[i] + 4*std_stds[i])
        bins = np.linspace(bin_min, bin_max, 50)
        
        # Plot histograms
        ax.hist(std_coeffs_all[:, i], bins=bins, density=True, alpha=0.5, color='#E94F37',
                edgecolor='#E94F37', linewidth=1.5, label=f'Naive (σ={std_stds[i]:.4f})')
        ax.hist(rob_coeffs_all[:, i], bins=bins, density=True, alpha=0.5, color='#2E86AB',
                edgecolor='#2E86AB', linewidth=1.5, label=f'ORF (σ={rob_stds[i]:.4f})')
        
        # Theoretical Gaussian
        x_gauss = np.linspace(bin_min, bin_max, 200)
        gauss_theory = np.exp(-(x_gauss - true_coeffs[i])**2 / (2 * theory_stds[i]**2)) / (theory_stds[i] * np.sqrt(2*np.pi))
        ax.plot(x_gauss, gauss_theory, 'k-', lw=2, label=f'Theory (σ={theory_stds[i]:.4f})')
        
        # True value line
        ax.axvline(true_coeffs[i], color='black', linestyle='--', lw=2, alpha=0.7)
        
        ax.set_xlabel(f'{name}')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Distribution of {name}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Monte Carlo Validation: {n_realizations} realizations, {int(outlier_fraction*100)}% outliers',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'polynomial_coefficients_mc.png')
    plt.savefig(PLOT_DIR / 'polynomial_coefficients_mc.pdf')
    print(f"Saved: {PLOT_DIR / 'polynomial_coefficients_mc.png'}")
    
    return fig1, fig2


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
    
    # Create figure - show convergence to final value
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    iterations = np.arange(len(a_history))
    a_final = a_history[-1]
    b_final = b_history[-1]
    
    ax = axes[0]
    a_diff = np.abs(np.array(a_history) - a_final)
    # Avoid log(0) by setting final point to a small value
    a_diff[-1] = 1e-16
    ax.semilogy(iterations[:-1], a_diff[:-1], 'o-', color='#2E86AB', lw=2, markersize=8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('|a - a_final|')
    ax.set_title(f'Intercept Convergence (final a = {a_final:.4f})')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    b_diff = np.abs(np.array(b_history) - b_final)
    b_diff[-1] = 1e-16
    ax.semilogy(iterations[:-1], b_diff[:-1], 'o-', color='#2E86AB', lw=2, markersize=8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('|b - b_final|')
    ax.set_title(f'Slope Convergence (final b = {b_final:.4f})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'convergence.png')
    plt.savefig(PLOT_DIR / 'convergence.pdf')
    print(f"\nSaved: {PLOT_DIR / 'convergence.png'}")
    
    return fig


def demo_heteroscedastic():
    """
    Demonstrate proper handling of heteroscedastic (varying) uncertainties.
    
    This demo shows that outliers are identified based on their deviation
    in units of their own uncertainty (sigma), not absolute deviation.
    It demonstrates the transition zone where points get progressively
    down-weighted as their sigma deviation increases.
    """
    print("\n" + "=" * 60)
    print("Demo 7: Heteroscedastic Uncertainties")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate data with heteroscedastic (varying) uncertainties
    n_points = 100
    true_a, true_b = 2.0, 0.5
    
    x = np.linspace(0, 10, n_points)
    # Uncertainties vary from 0.3 to 2.0 across the x range
    yerr = 0.3 + 1.7 * (x / x.max())
    
    # Generate data following the true line with heteroscedastic noise
    y_true = true_a + true_b * x
    y = y_true + np.random.normal(0, 1, n_points) * yerr
    
    # Add outliers at specific sigma levels: 3, 4, 5, 8, and 10 sigma
    # These span the transition zone where weights go from ~1 to ~0
    sigma_levels = [3, 4, 5, 8, 10]
    outlier_indices = [10, 30, 50, 70, 90]  # Spread across x range
    
    for idx, sigma in zip(outlier_indices, sigma_levels):
        y[idx] = y_true[idx] + sigma * yerr[idx]
    
    # Fit with robust method
    a_rob, a_rob_err, b_rob, b_rob_err, weights = orf.linear(
        x, y, yerr, return_weights=True
    )
    
    # Calculate sigma deviations for annotation
    residuals = y - (a_rob + b_rob * x)
    sigma_dev = residuals / yerr
    
    print(f"\nTrue values: a = {true_a:.3f}, b = {true_b:.3f}")
    print(f"Robust fit:  a = {a_rob:.3f} ± {a_rob_err:.3f}, b = {b_rob:.3f} ± {b_rob_err:.3f}")
    print(f"\nOutlier weights by sigma level:")
    for idx, sigma in zip(outlier_indices, sigma_levels):
        print(f"  {sigma}σ outlier: weight = {weights[idx]:.4f}")
    
    # Create figure - vertical layout
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Top panel: Data with outliers
    ax = axes[0]
    
    # Color by weight
    for i in range(n_points):
        color = plt.cm.RdYlGn(weights[i])
        ax.errorbar(x[i], y[i], yerr=yerr[i], fmt='o', color=color, 
                   markersize=6, capsize=2, elinewidth=1, markeredgecolor='black',
                   markeredgewidth=0.3, alpha=0.8)
    
    # Annotate the outliers
    for idx, sigma in zip(outlier_indices, sigma_levels):
        ax.annotate(f'{sigma}σ\nw={weights[idx]:.2f}',
                    xy=(x[idx], y[idx]),
                    xytext=(x[idx]+0.3, y[idx]+0.5),
                    fontsize=9, ha='left',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    
    # Plot fit
    x_fit = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_fit, true_a + true_b * x_fit, 'k--', lw=2, label='True line')
    ax.plot(x_fit, a_rob + b_rob * x_fit, '#2E86AB', lw=2.5, label='Robust fit')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Heteroscedastic errors (σ = 0.3 to 2.0) with outliers at 3σ, 4σ, 5σ, 8σ, 10σ')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Weight')
    
    # Bottom panel: sigma deviation vs weight
    ax = axes[1]
    ax.scatter(np.abs(sigma_dev), weights, c=weights, cmap='RdYlGn', 
               s=80, edgecolors='black', linewidth=0.5, vmin=0, vmax=1)
    
    # Highlight the specific outliers
    for idx, sigma in zip(outlier_indices, sigma_levels):
        ax.scatter(np.abs(sigma_dev[idx]), weights[idx],
                   s=150, facecolors='none', edgecolors='red', linewidth=2)
        ax.annotate(f'{sigma}σ', 
                    xy=(np.abs(sigma_dev[idx]), weights[idx]),
                    xytext=(np.abs(sigma_dev[idx])+0.3, weights[idx]+0.05),
                    fontsize=10)
    
    ax.set_xlabel('|Residual / σ| (sigma deviation)')
    ax.set_ylabel('Weight')
    ax.set_title('Weight Transition: 3σ→4σ→5σ→8σ→10σ')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 12)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'heteroscedastic.png')
    plt.savefig(PLOT_DIR / 'heteroscedastic.pdf')
    print(f"\nSaved: {PLOT_DIR / 'heteroscedastic.png'}")
    
    return fig


def demo_uncertainty_validation():
    """
    Monte Carlo validation of returned uncertainties.
    
    This demo performs many realizations of the fitting procedure and
    verifies that the returned uncertainties correctly describe the
    scatter in the recovered parameters.
    """
    print("\n" + "=" * 60)
    print("Demo 8: Monte Carlo Uncertainty Validation")
    print("=" * 60)
    
    np.random.seed(42)
    
    n_points = 100
    true_a, true_b = 2.0, 0.5
    n_realizations = 1000
    outlier_fraction = 0.05  # 5 outliers out of 100
    
    x = np.linspace(0, 10, n_points)
    yerr = np.ones(n_points) * 0.5
    
    # Storage for robust fit results
    a_values = []
    b_values = []
    a_errors = []
    b_errors = []
    
    # Storage for naive (standard WLS) fit results
    a_values_naive = []
    b_values_naive = []
    a_errors_naive = []
    b_errors_naive = []
    
    print(f"\nRunning {n_realizations} Monte Carlo realizations...")
    
    for i in range(n_realizations):
        # Generate new realization
        y = true_a + true_b * x + np.random.normal(0, yerr)
        
        # Add outliers
        n_outliers = int(outlier_fraction * n_points)
        if n_outliers > 0:
            outlier_idx = np.random.choice(n_points, n_outliers, replace=False)
            y[outlier_idx] += np.random.choice([-1, 1], n_outliers) * np.random.uniform(5, 15, n_outliers)
        
        # Robust fit
        a, a_err, b, b_err = orf.linear(x, y, yerr)
        a_values.append(a)
        b_values.append(b)
        a_errors.append(a_err)
        b_errors.append(b_err)
        
        # Naive fit (standard weighted least squares)
        a_naive, a_err_naive, b_naive, b_err_naive = standard_weighted_linfit(x, y, yerr)
        a_values_naive.append(a_naive)
        b_values_naive.append(b_naive)
        a_errors_naive.append(a_err_naive)
        b_errors_naive.append(b_err_naive)
    
    a_values = np.array(a_values)
    b_values = np.array(b_values)
    a_errors = np.array(a_errors)
    b_errors = np.array(b_errors)
    
    a_values_naive = np.array(a_values_naive)
    b_values_naive = np.array(b_values_naive)
    a_errors_naive = np.array(a_errors_naive)
    b_errors_naive = np.array(b_errors_naive)
    
    # Compute statistics for robust fit
    a_scatter = np.std(a_values)
    b_scatter = np.std(b_values)
    a_mean_error = np.mean(a_errors)
    b_mean_error = np.mean(b_errors)
    
    # Compute statistics for naive fit
    a_scatter_naive = np.std(a_values_naive)
    b_scatter_naive = np.std(b_values_naive)
    a_mean_error_naive = np.mean(a_errors_naive)
    b_mean_error_naive = np.mean(b_errors_naive)
    
    # Compute normalized residuals (should be ~N(0,1) if errors are correct)
    a_normalized = (a_values - true_a) / a_errors
    b_normalized = (b_values - true_b) / b_errors
    a_normalized_naive = (a_values_naive - true_a) / a_errors_naive
    b_normalized_naive = (b_values_naive - true_b) / b_errors_naive
    
    print(f"\nResults from {n_realizations} realizations:")
    print(f"\n{'='*60}")
    print("ROBUST FIT (odd ratio method)")
    print(f"{'='*60}")
    print(f"\nIntercept (a):")
    print(f"  True value: {true_a:.3f}")
    print(f"  Mean recovered: {np.mean(a_values):.3f}")
    print(f"  Actual scatter (std): {a_scatter:.4f}")
    print(f"  Mean reported error: {a_mean_error:.4f}")
    print(f"  Ratio (scatter/error): {a_scatter/a_mean_error:.3f} (should be ~1.0)")
    
    print(f"\nSlope (b):")
    print(f"  True value: {true_b:.3f}")
    print(f"  Mean recovered: {np.mean(b_values):.3f}")
    print(f"  Actual scatter (std): {b_scatter:.4f}")
    print(f"  Mean reported error: {b_mean_error:.4f}")
    print(f"  Ratio (scatter/error): {b_scatter/b_mean_error:.3f} (should be ~1.0)")
    
    print(f"\n{'='*60}")
    print("NAIVE FIT (standard weighted least squares)")
    print(f"{'='*60}")
    print(f"\nIntercept (a):")
    print(f"  True value: {true_a:.3f}")
    print(f"  Mean recovered: {np.mean(a_values_naive):.3f} (BIASED)")
    print(f"  Actual scatter (std): {a_scatter_naive:.4f}")
    print(f"  Mean reported error: {a_mean_error_naive:.4f}")
    print(f"  Ratio (scatter/error): {a_scatter_naive/a_mean_error_naive:.3f} (>> 1 = underestimated!)")
    
    print(f"\nSlope (b):")
    print(f"  True value: {true_b:.3f}")
    print(f"  Mean recovered: {np.mean(b_values_naive):.3f}")
    print(f"  Actual scatter (std): {b_scatter_naive:.4f}")
    print(f"  Mean reported error: {b_mean_error_naive:.4f}")
    print(f"  Ratio (scatter/error): {b_scatter_naive/b_mean_error_naive:.3f} (>> 1 = underestimated!)")
    
    print(f"\nNormalized residual statistics (should be ~N(0,1) if errors correct):")
    print(f"  Robust intercept: mean={np.mean(a_normalized):.3f}, std={np.std(a_normalized):.3f}")
    print(f"  Robust slope:     mean={np.mean(b_normalized):.3f}, std={np.std(b_normalized):.3f}")
    print(f"  Naive intercept:  mean={np.mean(a_normalized_naive):.3f}, std={np.std(a_normalized_naive):.3f}")
    print(f"  Naive slope:      mean={np.mean(b_normalized_naive):.3f}, std={np.std(b_normalized_naive):.3f}")
    
    # Create figure (2x2: histograms comparing robust vs naive)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top row: intercept distributions
    ax = axes[0, 0]
    ax.hist(a_values, bins=40, density=True, alpha=0.6, color='#2E86AB', 
            edgecolor='black', label=f'Robust (std={a_scatter:.3f})')
    ax.hist(a_values_naive, bins=40, density=True, alpha=0.6, color='#E94F37',
            edgecolor='black', label=f'Naive (std={a_scatter_naive:.3f})')
    ax.axvline(true_a, color='black', linestyle='--', lw=2, label=f'True = {true_a}')
    ax.set_xlabel('Intercept (a)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Recovered Intercepts')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.hist(a_normalized, bins=40, density=True, alpha=0.6, color='#2E86AB', 
            edgecolor='black', label=f'Robust (std={np.std(a_normalized):.2f})')
    ax.hist(a_normalized_naive, bins=40, density=True, alpha=0.6, color='#E94F37',
            edgecolor='black', label=f'Naive (std={np.std(a_normalized_naive):.2f})')
    # Overlay standard normal
    x_norm = np.linspace(-6, 6, 100)
    ax.plot(x_norm, np.exp(-x_norm**2/2)/np.sqrt(2*np.pi), 'k-', lw=2, label='N(0,1)')
    ax.set_xlabel('(a - a_true) / σ_a')
    ax.set_ylabel('Density')
    ax.set_title('Normalized Intercept Residuals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-6, 6)
    
    # Bottom row: slope distributions
    ax = axes[1, 0]
    ax.hist(b_values, bins=40, density=True, alpha=0.6, color='#2E86AB',
            edgecolor='black', label=f'Robust (std={b_scatter:.4f})')
    ax.hist(b_values_naive, bins=40, density=True, alpha=0.6, color='#E94F37',
            edgecolor='black', label=f'Naive (std={b_scatter_naive:.4f})')
    ax.axvline(true_b, color='black', linestyle='--', lw=2, label=f'True = {true_b}')
    ax.set_xlabel('Slope (b)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Recovered Slopes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.hist(b_normalized, bins=40, density=True, alpha=0.6, color='#2E86AB',
            edgecolor='black', label=f'Robust (std={np.std(b_normalized):.2f})')
    ax.hist(b_normalized_naive, bins=40, density=True, alpha=0.6, color='#E94F37',
            edgecolor='black', label=f'Naive (std={np.std(b_normalized_naive):.2f})')
    ax.plot(x_norm, np.exp(-x_norm**2/2)/np.sqrt(2*np.pi), 'k-', lw=2, label='N(0,1)')
    ax.set_xlabel('(b - b_true) / σ_b')
    ax.set_ylabel('Density')
    ax.set_title('Normalized Slope Residuals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-6, 6)
    
    plt.suptitle(f'Monte Carlo Validation: {n_realizations} realizations, {int(outlier_fraction*100)}% outliers\n'
                 'Robust fit vs Naive WLS',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'uncertainty_validation.png')
    plt.savefig(PLOT_DIR / 'uncertainty_validation.pdf')
    print(f"\nSaved: {PLOT_DIR / 'uncertainty_validation.png'}")
    
    # Also create a summary figure comparing robust vs naive
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Intercept (a)\nRobust', 'Intercept (a)\nNaive', 
                  'Slope (b)\nRobust', 'Slope (b)\nNaive']
    x_pos = np.arange(len(categories))
    width = 0.35
    
    actual_scatter = [a_scatter, a_scatter_naive, b_scatter, b_scatter_naive]
    reported_error = [a_mean_error, a_mean_error_naive, b_mean_error, b_mean_error_naive]
    colors = ['#2E86AB', '#E94F37', '#2E86AB', '#E94F37']
    
    bars1 = ax.bar(x_pos - width/2, actual_scatter, width, label='Actual Scatter (MC)', 
                   color=colors, edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, reported_error, width, label='Mean Reported Error',
                   color=['#28A745']*4, edgecolor='black', alpha=0.8)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Uncertainty Validation: Reported Errors Match Actual Scatter')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add ratio labels
    for i, (actual, reported) in enumerate(zip(actual_scatter, reported_error)):
        ratio = actual / reported
        ax.text(i, max(actual, reported) + 0.005, f'Ratio: {ratio:.2f}', 
                ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'uncertainty_summary.png')
    plt.savefig(PLOT_DIR / 'uncertainty_summary.pdf')
    print(f"Saved: {PLOT_DIR / 'uncertainty_summary.png'}")
    
    return fig, fig2


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("ODD RATIO FITS - DEMONSTRATION")
    print("=" * 60)
    print(f"\nPlots will be saved to: {PLOT_DIR.absolute()}\n")
    
    demo_linear_fit_comparison()
    demo_weighted_mean()
    demo_varying_outlier_fraction()
    demo_odd_ratio_sensitivity()
    demo_polynomial_fit()
    demo_convergence()
    demo_heteroscedastic()
    demo_uncertainty_validation()
    
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)
    print(f"\nPlots saved in: {PLOT_DIR.absolute()}")
    print("\nGenerated files:")
    for f in sorted(PLOT_DIR.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
