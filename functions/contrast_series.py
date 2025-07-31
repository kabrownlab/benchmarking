import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_grid(bounds=[[0, 1], [0, 1]], resolution=200):
    x = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X, Y = np.meshgrid(x, y)
    return X, Y

def base_gaussian(X, Y, center=(0.5, 0.5), width=0.1):
    cx, cy = center
    Z = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * width ** 2))
    return Z + 1.0

def build_shifted_functions(offsets):
    functions = {}
    for i, offset in enumerate(offsets):
        def func(X, Y, offset=offset):
            return base_gaussian(X, Y) + offset
        functions[f"f_c_{i+1}"] = func
    return functions

def estimate_lipschitz(f, bounds=[[0, 1], [0, 1]], resolution=300):
    X, Y = create_grid(bounds, resolution)
    Z = f(X, Y)
    dx, dy = np.gradient(Z, axis=0), np.gradient(Z, axis=1)
    grad_magnitude = np.sqrt(dx**2 + dy**2)
    return np.max(grad_magnitude)

def compute_contrast(f, bounds=[[0, 1], [0, 1]], resolution=300):
    X, Y = create_grid(bounds, resolution)
    Z = f(X, Y).flatten()
    return np.max(Z) / np.median(Z)

def plot_stacked_functions(functions, bounds=[[0, 1], [0, 1]], resolution=200):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for i, (name, f) in enumerate(functions.items()):
        X, Y = create_grid(bounds, resolution)
        Z = f(X, Y) + i * 2.5

        L = estimate_lipschitz(f)
        contrast = compute_contrast(f)

        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.text(1.05, 1.05, Z.max(), f"{name}\nL={L:.3f}, C={contrast:.2f}", fontsize=10)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.set_title("Stacked Target Functions with Same Lipschitz, Different Contrast")
    plt.tight_layout()
    plt.show()