"""
bo_to_theoretical_ef.py

Utilities for estimating the theoretical EF_max value based on:
- a Bayesian Optimization (BO) trace stored in a .pkl file,
- the target function f(x, y),
- its known contrast,
- and the BO step where EF_max is observed.

Author: [Your Name]
"""

import numpy as np
import pickle


def compute_theoretical_random_curve(f, steps, resolution=500):
    """
    Generate theoretical random max curve based on sampling target function f(x, y).
    
    Args:
        f: Callable, the target function f(X, Y)
        steps: int, number of steps (same as BO length)
        resolution: int, resolution of sampling grid (default=500)

    Returns:
        np.ndarray of shape (steps,), theoretical random max at each step
    """
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    values = f(X, Y).flatten()
    sorted_values = np.sort(values)
    cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

    random_curve = []
    for n in range(1, steps + 1):
        target_prob = 0.5 ** (1 / n)
        y_n = np.interp(target_prob, cdf, sorted_values)
        random_curve.append(y_n)
    return np.array(random_curve)


def compute_alpha_beta_at_step(bo_max_curve, random_max_curve, median, step):
    """
    Compute alpha and beta at a given step from BO and random curves.

    Args:
        bo_max_curve: np.ndarray, median BO max curve
        random_max_curve: np.ndarray, theoretical random max curve
        median: float, baseline median value
        step: int, 1-based step index

    Returns:
        alpha, beta: floats
    """
    bo_val = bo_max_curve[step - 1]
    random_val = random_max_curve[step - 1]

    alpha = (bo_val - random_val) / (bo_val - median)
    beta = (random_val - median) / (bo_val - median)
    return alpha, beta


def compute_theoretical_ef_max(alpha, beta, contrast):
    """
    Compute theoretical EF_max using derived alpha and beta.

    Args:
        alpha: float
        beta: float
        contrast: float, max / median of the function

    Returns:
        EF_max: float
    """
    numerator = contrast
    denominator = beta * contrast + (1 - beta)
    return numerator / denominator


def compute_and_report_theoretical_ef_max(
    bo_pkl_path, f, contrast, ef_max_step, value_key="obs_max_values", random_resolution=500
):
    """
    Full pipeline to compute and report theoretical EF_max given a BO .pkl trace.

    Args:
        bo_pkl_path: str, path to cached BO result (.pkl)
        f: callable, target function f(X, Y)
        contrast: float, max / median of the function
        ef_max_step: int, step at which BO achieved EF peak
        value_key: str, key in pickle file for BO max values (default: "obs_max_values")
        random_resolution: int, resolution of sampling grid for random curve (default=500)

    Returns:
        theoretical EF_max: float
    """
    # Load BO result
    with open(bo_pkl_path, "rb") as f_pkl:
        data = pickle.load(f_pkl)
        bo_max_curve = np.median(np.array(data[value_key]), axis=0)

    # Compute random max curve
    random_max_curve = compute_theoretical_random_curve(f, len(bo_max_curve), resolution=random_resolution)

    # Median from step 1
    median_val = random_max_curve[0]

    # Alpha and beta
    alpha, beta = compute_alpha_beta_at_step(bo_max_curve, random_max_curve, median_val, ef_max_step)

    # Compute theoretical EF max
    ef_max_theory = compute_theoretical_ef_max(alpha, beta, contrast)

    print(f"[Step {ef_max_step}] alpha = {alpha:.4f}, beta = {beta:.4f}, Theoretical EF_max = {ef_max_theory:.4f}")
    return ef_max_theory