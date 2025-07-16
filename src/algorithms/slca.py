import numpy as np
import time

def run_slca(X, y, error_func, args):

    # TODO implement lambda and tau search

    v_reset = 0
    v_threshold=1

    n_coefficients, max_time, max_steps,  = args["n_coefficients"], args["max_time"], args["max_steps"]
    lambda_, tau = args["lambda"], args["tau"]

    spikes = np.zeros((n_coefficients, max_steps))
    filtered_spikes = np.zeros((n_coefficients, max_steps))

    b = X.T @ y
    w = X.T @ X    

    times = []
    time_0 = time.time()
    t = 0

    v = np.zeros(n_coefficients)
    while time.time() - time_0 < max_time:
        if t > 0:
            filtered_spikes[:, t] = filtered_spikes[:, t - 1] * np.exp(-1 / tau) + spikes[:, t - 1]
        mu = b - w @ filtered_spikes[:, t]
        v += np.clip(mu - lambda_, 0, None)
        spikes[:, t] = (v >= v_threshold).astype(float)
        v[spikes[:, t] == 1] = v_reset
        times.append(time.time() - time_0)
        t += 1
    
    spikes = spikes[:, :t]
    spike_rates = np.zeros((n_coefficients, t))
    errors = []
    for j in range(t):
        for i in range(n_coefficients):
            spike_rates[i, j] = np.sum(spikes[i, :j + 1]) / (j + 1) if j > 0 else spikes[i, j]
        errors.append(error_func(spike_rates[:, j]))

    return spike_rates, times, errors