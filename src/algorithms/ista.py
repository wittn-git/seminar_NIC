import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.nn import Parameter
import numpy as np
import time
from proxtorch.operators import L1

def run_ista(X, y, error_function, args):

    max_lambda = np.max(np.abs(X.T @ y))
    n_coefficients, max_time, max_steps = args["n_coefficients"], args["max_time"], args["max_steps"]
    lambdas = np.linspace(0.001 * max_lambda, max_lambda, 15)

    best_coefficients, best_time, best_error = None, None, float('inf')
    for lambda_ in lambdas:
        coefficients, times = ista(X, y, lambda_, n_coefficients, max_time, max_steps)
        error = error_function(coefficients[:, -1])
        if error < best_error:
            best_error = error
            best_coefficients = coefficients
            best_time = times

    return best_coefficients, best_time, [error_function(coeff) for coeff in best_coefficients.T]

def soft_thresh(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0)

def ista(A, b, l, n_coefficients, max_time, max_steps):
    x = np.zeros(A.shape[1])
    L = np.linalg.norm(A, ord=2) ** 2

    coefficients = np.zeros((n_coefficients, max_steps))
    coefficients[:, 0] = x

    time0 = time.time()
    times = [0]

    for k in range(max_steps - 1):
        if time.time() - time0 > max_time:
            break

        x = soft_thresh(x + A.T @ (b - A @ x) / L, l / L)

        coefficients[:, k + 1] = x
        times.append(time.time() - time0)

    coefficients = coefficients[:, :k+1]
    return coefficients, times