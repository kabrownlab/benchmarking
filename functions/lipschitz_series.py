import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# 1. Define 2D Gaussian function
def gaussian_2d(X, Y, width=0.1, offset=0.0):
    return np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2 * width**2)) + offset

# 2. Generate mesh grid
def create_grid(resolution=300):
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y

# 3. Compute contrast: max / median
def compute_contrast_2d(width, offset, resolution=300):
    _, _, X, Y = create_grid(resolution)
    Z = gaussian_2d(X, Y, width, offset)
    return np.max(Z) / np.median(Z)

# 4. Find offset to match target contrast
def find_offset_for_contrast_2d(width, target_contrast, resolution=300):
    def objective(offset):
        return compute_contrast_2d(width, offset, resolution) - target_contrast
    result = root_scalar(objective, bracket=[0.0, 10.0], method='brentq')
    return result.root if result.converged else None

# 5. Estimate Lipschitz constant (max gradient)
def compute_lipschitz_constant_2d(width, offset, resolution=300):
    _, _, X, Y = create_grid(resolution)
    Z = gaussian_2d(X, Y, width, offset)
    dx, dy = np.gradient(Z, axis=(0, 1))
    grad_magnitude = np.sqrt(dx**2 + dy**2)
    return np.max(grad_magnitude)

# 6. Build a list of target functions with fixed contrast and varying Lipschitz constants
target_contrast = 1.8
widths = [0.15, 0.1, 0.08, 0.06, 0.04]
f_l_n = []
metrics = []

for i, w in enumerate(widths):
    offset = find_offset_for_contrast_2d(w, target_contrast)
    if offset is None:
        print(f"[!] Failed to find offset for width={w}")
        continue
    lipschitz = compute_lipschitz_constant_2d(w, offset)
    contrast = compute_contrast_2d(w, offset)
    f_l_n.append(lambda x, y, w=w, o=offset: gaussian_2d(x, y, w, o))
    metrics.append((w, offset, contrast, lipschitz))
    print(f"f_l_{i+1}: width={w}, offset={offset:.4f}, contrast={contrast:.4f}, Lipschitz={lipschitz:.4f}")

# 7. Optional: plot a horizontal slice at y=0.5
x_vals = np.linspace(0, 1, 300)
y_fixed = 0.5 * np.ones_like(x_vals)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

plt.figure(figsize=(8, 5))
for i, f in enumerate(f_l_n):
    y_vals = f(x_vals, y_fixed)
    _, offset, contrast, lipschitz = metrics[i]
    label = f"f_l_{i+1} (L={lipschitz:.3f}, C={contrast:.2f})"
    plt.plot(x_vals, y_vals, color=colors[i], label=label)

plt.xlabel("x (slice at y=0.5)")
plt.ylabel("f(x, y=0.5)")
plt.title("Target Functions with Fixed Contrast and Varying Lipschitz")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()