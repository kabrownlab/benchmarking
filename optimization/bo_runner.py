import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# Global bounds for optimization (2D unit square)
BOUNDS = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

def run_bo_full_trace(f, n_steps):
    """
    Run full Bayesian Optimization trace.

    Args:
        f: Callable objective function f(x, y)
        n_steps: Number of optimization steps

    Returns:
        x_obs: Observed input points (numpy array of shape [n_steps, 2])
        y_obs_list: List of observed function values
        y_pred_list: List of predicted means from posterior
    """
    x_obs = np.random.uniform(0, 1, size=(1, 2))
    y_obs = f(x_obs[:, 0], x_obs[:, 1])

    y_obs_list = list(y_obs)
    y_pred_list = []

    for step in range(1, n_steps):
        train_x = torch.tensor(x_obs, dtype=torch.float32)
        train_y = torch.tensor(y_obs, dtype=torch.float32).unsqueeze(-1)

        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        EI = ExpectedImprovement(model, best_f=train_y.max())
        candidate, _ = optimize_acqf(
            EI,
            bounds=BOUNDS,
            q=1,
            num_restarts=5,
            raw_samples=50,
        )
        x_new = candidate.detach().numpy()

        with torch.no_grad():
            pred = model.posterior(torch.tensor(x_new, dtype=torch.float32)).mean.item()
            y_pred_list.append(pred)

        y_new = f(x_new[:, 0], x_new[:, 1])
        y_obs_list.append(y_new[0])

        x_obs = np.vstack([x_obs, x_new])
        y_obs = np.append(y_obs, y_new)

    return x_obs, np.array(y_obs_list), np.array(y_pred_list)