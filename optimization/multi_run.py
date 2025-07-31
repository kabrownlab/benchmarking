import os
import pickle
import numpy as np

def multi_run_cached_extended(method_fn, f, method_name, f_name, n_steps, n_runs=10, cache_dir="cache"):
    filename = f"{method_name}_{f_name}_n{n_steps}_r{n_runs}.pkl"
    path = os.path.join(cache_dir, filename)

    if os.path.exists(path):
        with open(path, "rb") as f_cache:
            print(f"Loaded from cache: {path}")
            return pickle.load(f_cache)

    all_x, all_y_obs, all_y_pred = [], [], []
    for _ in range(n_runs):
        x, y_obs, y_pred = method_fn(f, n_steps)
        all_x.append(x)
        all_y_obs.append(y_obs)
        all_y_pred.append(y_pred)

    results = {
        "x": all_x,
        "y_obs": all_y_obs,
        "y_pred": all_y_pred,
        "obs_max_values": [np.maximum.accumulate(y) for y in all_y_obs],
        "pred_max_values": [np.maximum.accumulate(y) for y in all_y_pred],
    }

    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as f_cache:
        pickle.dump(results, f_cache)
        print(f"Saved to cache: {path}")

    return results