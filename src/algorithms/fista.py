import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.nn import Parameter
import numpy as np

from proxtorch.operators import L1

def run_fista(X, y, args):

    # TODO implement alpha search

    time_steps = args["time_steps"]
    coefficients = []
    alpha, learning_rate = args["alpha"], args["learning_rate"]
    l1_prox = L1(alpha=alpha)
    coefficients = fista(X, y, l1_prox, learning_rate, time_steps)
    return np.array(coefficients).T, {"alpha": alpha}

def fista(X, y, l1_prox, lr, n_iter):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    theta = Parameter(torch.zeros(X.shape[1]))
    optimizer = optim.SGD([theta], lr=lr)
    coefficients = [theta.detach().numpy()]

    for _ in range(n_iter):
        optimizer.zero_grad()
        y_pred = X @ theta
        loss = ((y_pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            theta.data = l1_prox.prox(theta, lr)
        coefficients.append(theta.detach().numpy())

    coefficients = np.array(coefficients)
    return coefficients