import numpy as np

def run_random_search_full_trace(f, n_steps):
    x_obs = []
    y_obs_list = []
    y_pred_list = []

    for step in range(n_steps):
        x_new = np.random.uniform(0, 1, size=(1, 2))
        y_new = f(x_new[:, 0], x_new[:, 1])[0]

        x_obs.append(x_new)
        y_obs_list.append(y_new)
        y_pred_list.append(y_new)

    return np.vstack(x_obs), np.array(y_obs_list), np.array(y_pred_list)