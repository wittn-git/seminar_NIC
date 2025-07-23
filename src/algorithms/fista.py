import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.nn import Parameter
import numpy as np
import time
from proxtorch.operators import L1

def run_fista(X, y, error_function, args):
    
    max_lambda = np.max(np.abs(X.T @ y))
    n_coefficients, max_time, max_steps = args["n_coefficients"], args["max_time"], args["max_steps"]
    lambdas = np.linspace(0.001 * max_lambda, max_lambda, 15)

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
    t = 1
    z = x.copy()
    L = np.linalg.norm(A, ord=2) ** 2

    coefficients = np.zeros((n_coefficients, max_steps))
    coefficients[:, 0] = x

    time0 = time.time()
    times = [0]

    for k in range(max_steps - 1):
        if time.time() - time0 > max_time:
            break

        x_old = x.copy()
        z = z + A.T @ (b - A @ z) / L
        x = soft_thresh(z, l / L)
        t0 = t
        t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - x_old)

        coefficients[:, k + 1] = x
        times.append(time.time() - time0)

    coefficients = coefficients[:, :k+1]
    return coefficients, times