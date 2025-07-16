from algorithms.fista import run_fista
from algorithms.ista import run_ista
from algorithms.coordinate_descent import run_coorddesc
from algorithms.slca import run_slca
from plotting.coeff_plot import save_lineplot
import numpy as np

def generate_data(n, p, n_fixed_coefficients):
    
    fixed_coefficients = np.random.uniform(0, 1, size=n_fixed_coefficients)

    X = []
    y = []
    
    zero_coefficients = np.zeros(p - n_fixed_coefficients)
    for _ in range(n):
        x_i = np.random.normal(size=p) 
        coefficients = np.concatenate((fixed_coefficients, zero_coefficients))
        y_i = np.dot(x_i, coefficients)
        X.append(x_i)
        y.append(y_i)
    
    X = np.array(X)
    y = np.array(y)
    
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (y - y.mean()) / y.std()
    
    return X, y, fixed_coefficients

def diff(estimated_coefficients, coefficients):
    return sum([a - b for a, b in zip(estimated_coefficients, coefficients)]) / len(coefficients)

if __name__ == "__main__":

    np.random.seed(23655)

    n_data_points = 25000
    n_fixed_coefficients = 2
    n_random_coefficients = 2

    args = {
        "time_steps": 500,
        "lambda": 10,
        "tau": 2.5,
        "learning_rate": 0.01,
        "n_coefficients": n_fixed_coefficients + n_random_coefficients,
        "alpha": 1.1
    }
    
    n_coefficients = n_fixed_coefficients + n_random_coefficients
    X, y, fixed_coefficients = generate_data(int(n_data_points), n_fixed_coefficients + n_random_coefficients, n_fixed_coefficients)
    true_coefficients = np.concatenate((fixed_coefficients, np.zeros(n_random_coefficients)))
    print("True Coefficients:", true_coefficients)

    algorithms = {
        "S-LCA": run_slca,
        #"Coordinate Descent": run_coorddesc,
        "Fista": run_fista,
        "Ista": run_ista
    }

    results = {}

    for name, algorithm in algorithms.items():
        print(f"Running {name}...")
        coeffs, params, times = algorithm(X, y, args)
        results[name] = {"coeffs": coeffs, "params": params, "times": times}
        print(f"{name} Coefficients:", results[name]["coeffs"][:, -1])
        print(f"{name} Diff:", diff(results[name]["coeffs"][:, -1], true_coefficients))
        print(f"{name} Parameters:", results[name]["params"])

    save_lineplot(results, n_fixed_coefficients, true_coefficients, args["time_steps"])
    # make error plot
    # make time plot
