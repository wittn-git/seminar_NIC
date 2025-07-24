import numpy as np
import time

def run_fista(X, y, error_function, args):
    
    max_lambda = np.max(np.abs(X.T @ y))
    n_coefficients, max_time, max_steps = args["n_coefficients"], args["max_time"], args["max_steps"]
    lambdas = np.logspace(np.log10(max_lambda * 1e-4), np.log10(max_lambda), 15)

    best_coefficients, best_time, best_error = None, None, float('inf')
    for lambda_ in lambdas:
        coefficients, times = fista(X, y, lambda_, n_coefficients, max_time, max_steps)
        error = error_function(coefficients[:, -1])
        if error < best_error:
            best_error = error
            best_coefficients = coefficients
            best_time = times

    return best_coefficients, best_time, [error_function(coeff) for coeff in best_coefficients.T]

def soft_thresh(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0)

def fista(A, b, l, n_coefficients, max_time, max_steps):
    x = np.zeros(A.shape[1])
    w = 1
    z = x.copy()
    L = np.linalg.norm(A, ord=2) ** 2

    coefficients = np.zeros((n_coefficients, max_steps))
    coefficients[:, 0] = x

    time0 = time.time()
    times = [0]
    t = 0

    while time.time() - time0 < max_time:

        x_old = x.copy()
        z = z + A.T @ (b - A @ z) / L
        x = soft_thresh(z, l / L)
        w0 = w
        w = (1. + np.sqrt(1. + 4. * w ** 2)) / 2.
        z = x + ((w0 - 1.) / w) * (x - x_old)

        coefficients[:, t + 1] = x
        times.append(time.time() - time0)
        t += 1

    coefficients = coefficients[:, :t+1]
    return coefficients, times