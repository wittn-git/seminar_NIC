import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.nn import Parameter
import numpy as np
import time
from proxtorch.operators import L1

def run_fista(X, y, error_function, args):

    n_coefficients, max_time, max_steps = args["n_coefficients"], args["max_time"], args["max_steps"]
    #alphas, learning_rates = np.linspace(0.01, 1, 10), np.linspace(0.001, 0.3, 10)
    alphas, learning_rates = [0.7], [0.01]

    best_coefficients, best_time, best_error = None, None, float('inf')
    for alpha in alphas:
        for learning_rate in learning_rates:
            coefficients, times = fista(X, y, alpha, learning_rate, n_coefficients, max_time, max_steps)
            error = error_function(coefficients[:, -1])
            if error < best_error:
                best_error = error
                best_coefficients = coefficients
                best_time = times

    return best_coefficients, best_time, [error_function(coeff) for coeff in best_coefficients.T]

def fista(X, y, alpha, learning_rate, n_coefficients, max_time, max_steps):
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    theta = Parameter(torch.zeros(X.shape[1]))
    optimizer = optim.SGD([theta], lr=learning_rate)
    l1_prox = L1(alpha=alpha)

    coefficients = np.zeros((n_coefficients, max_steps))
    coefficients[:, 0] = theta.detach().numpy()

    time_0 = time.time()
    times = [0]
    t = 0

    while time.time() - time_0 < max_time:
        optimizer.zero_grad()
        y_pred = X @ theta
        loss = ((y_pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            theta.data = l1_prox.prox(theta, learning_rate)
        times.append(time.time() - time_0)
        t += 1
        coefficients[:, t] = theta.detach().numpy()

    coefficients = coefficients[:, :t+1]
    return coefficients, times