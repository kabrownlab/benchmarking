import numpy as np
import matplotlib.pyplot as plt

# Base 1D Gaussian function

def base_gaussian_1d(x, center=0.5, width=0.1):
    return np.exp(-((x - center) ** 2) / (2 * width ** 2)) + 1.0

# Function parameters
offsets = [0.0, 0.5, 1.0, -0.5, 1.5]
noise_levels = [0.0, 0.05, 0.1]
noise_labels = ["clean", "low_noise", "high_noise"]
noise_titles = ["Clean", "Low Noise", "High Noise"]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

x = np.linspace(0, 1, 500)
all_funcs = {}
noise_stats = {}

# Build noisy variants of f_c_i
for i, offset in enumerate(offsets):
    def clean_func(x, offset=offset):
        return base_gaussian_1d(x) + offset

    clean_y = clean_func(x)
    for j, noise_std in enumerate(noise_levels):
        name = f"f_c_{i+1}{noise_labels[j]}"
        if noise_std == 0.0:
            y = clean_y
        else:
            np.random.seed(42)
            y = clean_y + np.random.normal(0, noise_std, size=clean_y.shape)

        all_funcs[name] = y
        if noise_std > 0:
            rel_noise = np.std(y - clean_y) / np.median(y)
        else:
            rel_noise = 0.0
        noise_stats[name] = rel_noise

# Plot noisy functions grouped by noise level
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for j in range(3):
    ax = axes[j]
    for i in range(5):
        name = f"f_c_{i+1}{noise_labels[j]}"
        label = f"f_c_{i+1} (noise/med={noise_stats[name]:.3f})"
        ax.plot(x, all_funcs[name], color=colors[i], label=label)

    ax.set_title(f"{noise_titles[j]} Functions")
    ax.set_xlabel("x (slice at y=0.5)")
    ax.grid(True)
    if j == 0:
        ax.set_ylabel("f(x, y=0.5)")
    ax.legend()

plt.suptitle("f_c_n Family Grouped by Noise Level", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()