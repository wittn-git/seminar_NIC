import numpy as np
from scipy import linalg

# https://gist.github.com/agramfort/ac52a57dc6551138e89b

def run_ista(X, y, args):

    # TODO implement alpha search

    time_steps = args["time_steps"]
    coefficients = []
    alpha = args["alpha"]
    coefficients = ista(X, y, alpha, time_steps)
    return np.array(coefficients).T, {"alpha": alpha}

def soft_thresh(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0.)

def ista(A, b, l, maxit):
    x = np.zeros(A.shape[1])
    L = linalg.norm(A) ** 2 
    coefficients = []
    for _ in range(maxit):
        x = soft_thresh(x + np.dot(A.T, b - A.dot(x)) / L, l / L)
        coefficients.append(x.copy())

    return coefficients