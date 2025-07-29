# Bayesian Optimization on 2D Gaussian Landscapes

This repository investigates how properties of the objective function—specifically **contrast**, **Lipschitz constant**, and **noise level**—affect the efficiency of **Bayesian Optimization (BO)**.

We design a family of synthetic 2D Gaussian functions with tunable characteristics and evaluate how quickly BO finds high-performing solutions under varying conditions.

---

## 🔍 Motivation

Bayesian Optimization is widely used for optimizing expensive black-box functions. However, the optimization landscape (e.g., sharp vs. flat peaks, or noisy vs. clean signals) can significantly impact BO’s performance.

We systematically explore:
- How **contrast** (peak/median ratio) affects BO.
- How **Lipschitz continuity** relates to optimization difficulty.
- How **noise** degrades performance under different function complexities.

---

## 🧪 Methodology

We construct synthetic 2D functions of the form:

```python
def gaussian_2d(X, Y, width, offset):
    return np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2 * width**2)) + offset
```

### 1. Contrast Series (`f_c_i`)

Functions with the same shape (fixed width) but different vertical shifts (`offset`) to achieve varying contrast (peak-to-median ratio):

| Function | Offset | Contrast (max / median) |
|----------|--------|--------------------------|
| `f_c_1`  | 0.0    | ~1.67                    |
| `f_c_2`  | 0.5    | ~1.50                    |
| `f_c_3`  | 1.0    | ~1.40                    |
| `f_c_4`  | -0.5   | ~3.00                    |
| `f_c_5`  | 1.5    | ~1.35                    |

### 2. Lipschitz Series (`f_l_i`)

Functions with increasing sharpness (i.e., decreasing Gaussian width) and adjusted offset to maintain fixed contrast (`C ≈ 1.8`). This results in increasing Lipschitz constant `L`.

| Function | Width  | Offset  | Contrast (max / median) | Lipschitz Constant `L` |
|----------|--------|---------|--------------------------|-------------------------|
| `f_l_1`  | 0.15   | 1.1859  | ~1.80                    | 0.0135                  |
| `f_l_2`  | 0.10   | 1.2489  | ~1.80                    | 0.0203                  |
| `f_l_3`  | 0.08   | 1.2494  | ~1.80                    | 0.0405                  |
| `f_l_4`  | 0.06   | 1.2490  | ~1.80                    | 0.0673                  |
| `f_l_5`  | 0.04   | 1.2478  | ~1.80                    | 0.1938                  |

### 3. Noise Perturbation

For each f_l_i, we add Gaussian noise of two levels:
	•	Low noise: σ = 0.05
	•	High noise: σ = 0.10

The relative noise level is normalized by the function’s median.

---

## 📈 Evaluation Protocol


To evaluate the performance of learning algorithms upon the objective functions, we use:
	•	Optimization method: Bayesian Optimization
	•	Repeats: 100 runs per function
	•	Steps per run: 100
	•	Metrics:
	•	Peak EF value: Best result found so far
	•	Step to Peak: Iteration index at which peak was reached
 
---

## 📂 File Structure
```
.
├── functions/
│   ├── contrast_series.py     # f_c_i definitions
│   ├── lipschitz_series.py    # f_l_i definitions
│   └── noise_wrappers.py      # noisy function wrappers
├── optimization/
│   ├── bo_runner.py           # Bayesian optimization loops
│   ├── random_search.py       # Random search baseline
│   └── multi_run.py           # batch execution
├── plots/
│   └── plotting_scripts.py    # Reproducible plot scripts
├── assets/                    # Exported SVG plots
└── README.md
```

## Citation
