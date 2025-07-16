import numpy as np
import time

def run_slca(X, y, args):

    # TODO implement lambda and tau search

    params = {
        "lambda": args["lambda"],
        "tau": args["tau"]
    }

    v_reset = 0
    v_threshold=1

    n_coefficients, time_steps, lambda_, tau = args["n_coefficients"], args["time_steps"], args["lambda"], args["tau"]

    spikes = np.zeros((n_coefficients, time_steps))
    filtered_spikes = np.zeros((n_coefficients, time_steps))

    b = X.T @ y
    w = X.T @ X    

    times = []
    time_0 = time.time()

    v = np.zeros(n_coefficients)
    for t in range(time_steps):
        if t > 0:
            filtered_spikes[:, t] = filtered_spikes[:, t - 1] * np.exp(-1 / tau) + spikes[:, t - 1]
        mu = b - w @ filtered_spikes[:, t]
        v += np.clip(mu - lambda_, 0, None)
        spikes[:, t] = (v >= v_threshold).astype(float)
        v[spikes[:, t] == 1] = v_reset
        times.append(time.time() - time_0)

    spike_rates = np.zeros((n_coefficients, time_steps))
    for i in range(n_coefficients):
        for t in range(time_steps):
            spike_rates[i, t] = np.sum(spikes[i, :t + 1]) / (t + 1) if t > 0 else spikes[i, t]
            
    return spike_rates, params, times