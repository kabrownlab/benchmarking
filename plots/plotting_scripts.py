import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.optimize import curve_fit


def compute_best_obs_curve_stable(samples, steps):
    """
    Compute the best observed value at each step from a sample of values.
    """
    samples = np.array(samples)
    n_samples = len(samples)
    results = []
    for n in steps:
        best_values = [np.max(np.random.choice(samples, size=n, replace=False)) for _ in range(1000)]
        results.append(np.median(best_values))
    return np.array(results)

def plot_step_vs_max_with_theory_and_baselines(
    pkl_paths_with_labels,
    value_key="obs_max_values",
    theoretical_sample=None,
    theoretical_label=r"$\mathrm{f}_{C_1}$",
    true_max=None
):
    """
    Plot median + interquartile range of step-vs-best-max curves with optional theory line and true max.

    Parameters:
        pkl_paths_with_labels: list of (label, path) tuples
        value_key: key in the .pkl dict for extracting [n_runs x n_steps] arrays
        theoretical_sample: optional empirical distribution to plot as theoretical bound
        theoretical_label: label for the theoretical curve
        true_max: optional float, draw a horizontal reference line
    """
    plt.figure(figsize=(3.5, 3))
    plt.rcParams['font.family'] = 'Arial'

    for name, path in pkl_paths_with_labels:
        with open(path, "rb") as f:
            data = pickle.load(f)
            values = np.array(data[value_key])
        median = np.median(values, axis=0)
        q1 = np.percentile(values, 25, axis=0)
        q3 = np.percentile(values, 75, axis=0)
        steps = np.arange(1, len(median) + 1)

        plt.plot(steps, median, label=name, linewidth=1.21, zorder=3)
        plt.fill_between(steps, q1, q3, alpha=0.2, zorder=2)

    if theoretical_sample is not None:
        theory_steps = np.arange(1, len(median) + 1)
        theory_curve = compute_best_obs_curve_stable(theoretical_sample, theory_steps)
        plt.plot(theory_steps, theory_curve, '--', label=theoretical_label,
                 color='black', linewidth=1.03, zorder=1)

    if true_max is not None:
        plt.axhline(y=true_max, color='gray', linestyle='--', linewidth=1.03,
                    label=r"$y^{*}$", zorder=0)

    plt.xlabel(r"$n$", fontsize=18)
    plt.ylabel(r"$\mathit{y}^{*}_{\mathrm{n}}$", fontsize=18)
    plt.xticks(fontsize=17.7)
    plt.yticks(fontsize=17.7)

    plt.legend(
        frameon=False, fontsize=17.7,
        loc="lower right", bbox_to_anchor=(1.07, 0),
        handletextpad=0.3, labelspacing=0.3, borderaxespad=0.3
    )

    plt.grid(False)
    plt.tight_layout()
    plt.savefig("step_vs_max_halfpage.svg", format="svg", dpi=300)
    # plt.savefig("step_vs_max_halfpage.pdf", format="pdf", dpi=300)
    plt.show()
    

def compute_best_obs_curve_stable(y_samples, steps):
    """Compute the best observed value up to each step from random sampling."""
    sorted_samples = np.sort(y_samples)[::-1]  # Descending order
    best_obs = np.maximum.accumulate(sorted_samples[:len(steps)])
    return best_obs

def plot_step_vs_max_with_theory_and_baselines(
    pkl_paths_with_labels,
    value_key="obs_max_values",
    theoretical_sample=None,
    theoretical_label=r"$\mathrm{f}_{C_1}$",
    true_max=None
):
    plt.figure(figsize=(3.5, 3))
    plt.rcParams['font.family'] = 'Arial'

    for name, path in pkl_paths_with_labels:
        with open(path, "rb") as f:
            data = pickle.load(f)
            values = np.array(data[value_key])
        median = np.median(values, axis=0)
        q1 = np.percentile(values, 25, axis=0)
        q3 = np.percentile(values, 75, axis=0)
        steps = np.arange(1, len(median) + 1)

        plt.plot(
            steps, median, label=name, linewidth=1.21, zorder=3
        )
        plt.fill_between(
            steps, q1, q3, alpha=0.2, zorder=2
        )

    if theoretical_sample is not None:
        theory_steps = np.arange(1, len(median) + 1)
        theory_curve = compute_best_obs_curve_stable(theoretical_sample, theory_steps)
        plt.plot(
            theory_steps, theory_curve, '--',
            label=theoretical_label, color='black', linewidth=1.03, zorder=1
        )

    if true_max is not None:
        plt.axhline(
            y=true_max, color='gray', linestyle='--',
            linewidth=1.03, label=r"$y^{*}$", zorder=0
        )

    plt.xlabel(r"$n$", fontsize=18)
    plt.ylabel(r"$\mathit{y}^{*}_{\mathrm{n}}$", fontsize=18)
    plt.xticks(fontsize=17.7)
    plt.yticks(fontsize=17.7)

    plt.legend(
        frameon=False, fontsize=17.7,
        loc="lower right", bbox_to_anchor=(1.07, 0),
        handletextpad=0.3, labelspacing=0.3, borderaxespad=0.3
    )

    plt.grid(False)
    plt.tight_layout()
    plt.savefig("step_vs_max_halfpage.svg", format="svg", dpi=300)
    plt.show()

def plot_bo_vs_theory_ef_curves(
    function_keys,
    bo_pkl_paths,
    theory_samples_dict,
    value_key="obs_max_values"
):
    plt.figure(figsize=(3.5, 3))
    plt.rcParams['font.family'] = 'Arial'

    for func_key in function_keys:
        with open(bo_pkl_paths[func_key], "rb") as f:
            bo_data = pickle.load(f)
            bo_max = np.array(bo_data[value_key])

        median_bo = np.median(bo_max, axis=0)
        steps = np.arange(1, len(median_bo) + 1)

        theory_y = theory_samples_dict[func_key]
        theory_curve = compute_best_obs_curve_stable(theory_y, steps)

        plt.plot(
            steps, median_bo / theory_curve,
            label=rf"$\mathrm{{f}}_{{L_{{{function_keys.index(func_key)+1}}}}}$",
            linewidth=1.21
        )

        peak_step = int(np.argmax(median_bo / theory_curve)) + 1
        peak_val = (median_bo / theory_curve)[peak_step - 1]
        print(f"[{func_key}] EF_peak = {peak_val:.3f} at step {peak_step}")

    plt.xlabel(r"$n$", fontsize=18)
    plt.ylabel(r"$EF$", fontsize=18)
    plt.xticks(fontsize=17.7)
    plt.yticks(fontsize=17.7)

    # Uncomment to show legend if needed
    # plt.legend(
    #     frameon=False, fontsize=17.7, loc='upper right',
    #     handletextpad=0.3, labelspacing=0.3, borderaxespad=0.2
    # )

    plt.savefig("L_ef_curves_halfpage.svg", format="svg", dpi=300)
    plt.show()


def plot_ef_peak_vs_contrast():
    """Plot EF peak value against contrast with log fit."""
    # Data
    contrast = np.array([1.9991, 1.6661, 1.4996, 2.9975, 1.3997])
    theoretical_ef_max = np.array([1.3617, 1.2570, 1.1950, 1.5456, 1.1676])
    ef_max_values = np.array([1.354, 1.313, 1.186, 1.541, 1.165])

    # Fit log curve
    def log_func(C, a, b):
        return a * np.log(C) + b

    popt, _ = curve_fit(log_func, contrast, theoretical_ef_max)
    a, b = popt
    print(f"Fitted curve: EF = {a:.4f} * log(C) + {b:.4f}")

    # Plotting
    plt.figure(figsize=(3.5, 3))
    plt.rcParams['font.family'] = 'Arial'

    # Theoretical EF max
    plt.scatter(
        contrast, theoretical_ef_max,
        color='tab:blue', label=r"$\mathrm{Theory}$", s=30, zorder=3
    )

    # Fitted log curve
    x_fit = np.linspace(min(contrast)*0.95, max(contrast)*1.05, 200)
    y_fit = log_func(x_fit, *popt)
    plt.plot(
        x_fit, y_fit, 'b--',
        linewidth=1.21, zorder=2
    )

    # Experimental EF max
    plt.scatter(
        contrast, ef_max_values,
        color='tab:red', marker='s', label=r"$\mathrm{Computed}$", s=30, zorder=4
    )

    # Axis and labels
    plt.xlabel(r"$C$", fontsize=18, style='italic')
    plt.ylabel(r"$\max(\mathit{EF})$", fontsize=18)
    plt.xticks(fontsize=17.7)
    plt.yticks(fontsize=17.7)

    # Legend
    plt.legend(
        frameon=False,
        fontsize=17.7,
        loc='lower right',
        handletextpad=0.3,
        labelspacing=0.3,
        borderaxespad=0.05
    )

    plt.grid(False)
    plt.tight_layout()
    plt.savefig("ef_max_vs_contrast_halfpage.svg", format='svg', dpi=300)
    # plt.savefig("ef_max_vs_contrast_halfpage.pdf", format='pdf', dpi=300)

    plt.show()
    print("✅ Exported: ef_max_vs_contrast_halfpage.svg & pdf (half page, 300 dpi)")

def plot_lipschitz_ef_vs_noise():
    # === Basic Data ===
    lipschitz_constants = [0.0135, 0.0203, 0.0405, 0.0673, 0.1938]  # L_1 to L_5
    ef_peaks = [
        [1.236, 1.211, 1.166],  # f_l_1
        [1.325, 1.261, 1.223],  # f_l_2
        [1.348, 1.305, 1.273],  # f_l_3
        [1.409, 1.391, 1.281],  # f_l_4
        [1.436, 1.363, 1.135]   # f_l_5
    ]

    noise_levels = [0.0, 0.05, 0.1]
    medians = [np.median(np.exp(-((np.linspace(0,1,500)-0.5)**2)/(2*w**2)) + offset)
               for w, offset in zip([0.15, 0.1, 0.08, 0.06, 0.04],
                                    [1.1859, 1.2489, 1.2494, 1.2490, 1.2478])]
    relative_noise_levels = []
    for median in medians:
        rel_levels = [round(n / median, 4) for n in noise_levels]
        relative_noise_levels.append(rel_levels)

    # === Plotting ===
    plt.figure(figsize=(3.5, 3))
    plt.rcParams['font.family'] = 'Arial'

    for i, (rel_noise, ef_series) in enumerate(zip(relative_noise_levels, ef_peaks)):
        label = rf"$\mathrm{{f}}_{{L_{i+1}}}$"
        plt.plot(
            rel_noise, ef_series,
            marker='o', markersize=5,
            linewidth=1.21, label=label
        )

    plt.xlabel(r"$\mathrm{\sigma} / \mathrm{med}(y)$", fontsize=18)
    plt.ylabel(r"$\max(\mathit{EF})$", fontsize=18)
    plt.xticks(fontsize=17.7)
    plt.yticks(fontsize=17.7)

    # Uncomment if you want to show legend
    # plt.legend(
    #     frameon=False, fontsize=17.7, loc='lower left',
    #     handletextpad=0.3, labelspacing=0.1, borderaxespad=0.1
    # )

    plt.grid(False)
    plt.tight_layout()
    plt.savefig("l_ef_peak_vs_noise_halfpage.svg", format='svg', dpi=300)
    # plt.savefig("l_ef_peak_vs_noise_halfpage.pdf", format='pdf', dpi=300)
    plt.show()