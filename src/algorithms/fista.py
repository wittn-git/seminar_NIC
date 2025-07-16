import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.nn import Parameter
import numpy as np
import time
from proxtorch.operators import L1

def run_fista(X, y, error_function, args):

    # TODO implement alpha search

    n_coefficients, max_time, max_steps = args["n_coefficients"], args["max_time"], args["max_steps"]
    alpha, learning_rate = args["alpha"], args["learning_rate"]
    l1_prox = L1(alpha=alpha)
    coefficients, times, errors = fista(X, y, l1_prox, n_coefficients, learning_rate, max_time, max_steps, error_function)
    return coefficients, times, errors

def fista(X, y, l1_prox, n_coefficients, lr, max_time, max_steps, error_function):
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    theta = Parameter(torch.zeros(X.shape[1]))
    optimizer = optim.SGD([theta], lr=lr)

    coefficients = np.zeros((n_coefficients, max_steps))
    coefficients[:, 0] = theta.detach().numpy()

    time_0 = time.time()
    times = [0]
    t = 0
    errors = [error_function(coefficients[:, 0])]

    while time.time() - time_0 < max_time:
        optimizer.zero_grad()
        y_pred = X @ theta
        loss = ((y_pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            theta.data = l1_prox.prox(theta, lr)
        times.append(time.time() - time_0)
        t += 1
        coefficients[:, t] = theta.detach().numpy()
        errors.append(error_function(coefficients[:, t]))

    coefficients = coefficients[:, :t+1]
    return coefficients, times, errors