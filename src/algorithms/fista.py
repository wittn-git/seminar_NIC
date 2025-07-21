import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.nn import Parameter
import numpy as np
import time
from proxtorch.operators import L1

def run_fista(X, y, error_function, args):
    
    n_coefficients, max_time, max_steps = args["n_coefficients"], args["max_time"], args["max_steps"]
    lambdas, learning_rates = np.linspace(0.01, 1.5, 20), np.linspace(0.001, 0.3, 20)

    best_coefficients, best_time, best_error = None, None, float('inf')
    for lambda_ in lambdas:
        for learning_rate in learning_rates:
            coefficients, times = fista(X, y, lambda_, learning_rate, n_coefficients, max_time, max_steps)
            error = error_function(coefficients[:, -1])
            if error < best_error:
                best_error = error
                best_coefficients = coefficients
                best_time = times

    return best_coefficients, best_time, [error_function(coeff) for coeff in best_coefficients.T]

def fista(X, y, lambda_, learning_rate, n_coefficients, max_time, max_steps):

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    theta = torch.zeros(X.shape[1], dtype=torch.float32)
    y_k = theta.clone()
    t_k = 1

    l1_prox = L1(alpha=lambda_)

    coefficients = np.zeros((n_coefficients, max_steps))
    coefficients[:, 0] = theta.numpy()

    time_0 = time.time()
    times = [0]

    t = 0
    while time.time() - time_0 < max_time and t < max_steps - 1:

        y_k = y_k.detach().requires_grad_(True)
        y_pred = X @ y_k
        loss = ((y_pred - y) ** 2).mean()
        loss.backward()
        grad = y_k.grad

        theta_next = l1_prox.prox(y_k - learning_rate * grad, learning_rate)

        t_next = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2
        y_next = theta_next + ((t_k - 1) / t_next) * (theta_next - theta)

        theta = theta_next
        y_k = y_next
        t_k = t_next

        t += 1
        coefficients[:, t] = theta.detach().numpy()
        times.append(time.time() - time_0)

    coefficients = coefficients[:, :t+1]
    return coefficients, times