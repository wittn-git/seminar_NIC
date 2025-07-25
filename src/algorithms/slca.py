import numpy as np
import time

def run_slca(X, y, error_func, args):

    max_lambda = np.max(np.abs(X.T @ y))
    n_coefficients, max_time, max_steps = args["n_coefficients"], args["max_time"], args["max_steps"]
    lambdas = np.logspace(np.log10(max_lambda * 1e-4), np.log10(max_lambda), 15)
    taus = np.linspace(1, 100, 10)

    best_coefficients, best_time, best_errors = None, None, float('inf')
    for lambda_ in lambdas:
        for tau in taus:
            coefficients, times = slca(X, y, n_coefficients, max_time, max_steps, lambda_, tau)
            error = error_func(coefficients[:, -1])
            if error < best_errors:
                best_errors = error
                best_coefficients = coefficients
                best_time = times
    
    return best_coefficients, best_time, [error_func(coeff) for coeff in best_coefficients.T]

def slca(X, y, n_coefficients, max_time, max_steps, lambda_, tau):
    spikes = np.zeros((n_coefficients, max_steps))
    filtered_spikes = np.zeros((n_coefficients, max_steps))

    b = X.T @ y
    w = X.T @ X    

    times = [0]
    time_0 = time.time()
    t = 1

    v = np.zeros(n_coefficients)
    while time.time() - time_0 < max_time:
        if t > 0:
            filtered_spikes[:, t] = filtered_spikes[:, t - 1] * np.exp(-1 / tau) + spikes[:, t - 1]
        mu = b - w @ filtered_spikes[:, t]
        v += np.clip(mu - lambda_, 0, None)
        spikes[:, t] = (v >= 1).astype(float)
        v[spikes[:, t] == 1] = 0
        times.append(time.time() - time_0)
        t += 1
    
    spikes = spikes[:, :t+1]
    spike_rates = np.zeros((n_coefficients, t))
    for j in range(t):
        for i in range(n_coefficients):
            spike_rates[i, j] = np.sum(spikes[i, :j + 1]) / (j + 1) if j > 0 else spikes[i, j]

    return spike_rates, times