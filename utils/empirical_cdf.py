import numpy as np

def compute_best_obs_curve_stable(y_samples, steps):
    """
    Compute the expected best observation at each step under random sampling.

    Parameters:
        y_samples (array-like): Flattened samples of the objective function.
        steps (array-like): List of step indices (e.g., range(1, 101)).

    Returns:
        np.ndarray: Expected best observation values at each step.
    """
    sorted_y = np.sort(y_samples)
    n = len(sorted_y)
    cdf = np.arange(1, n + 1) / n

    expected_values = []
    for n_step in steps:
        target_prob = 0.5 ** (1 / n_step)
        y_n = np.interp(target_prob, cdf, sorted_y)
        expected_values.append(y_n)
    return np.array(expected_values)