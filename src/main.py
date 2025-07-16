from algorithms.fista import run_fista
from algorithms.slca import run_slca
from plotting.coeff_plot import save_lineplot
from plotting.time_plot import save_timeplot
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
    
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
    y_norm = (y - y.mean()) / y.std()
    
    return X, y, X_norm, y_norm, fixed_coefficients

def diff(estimated_coefficients, coefficients):
    return sum([a - b for a, b in zip(estimated_coefficients, coefficients)]) / len(coefficients)

def get_error(X, y, coefficients):
    y_pred = X @ coefficients
    return np.mean((y_pred - y) ** 2)

if __name__ == "__main__":

    # TODO impl. multi run experiments

    np.random.seed(23655)

    n_data_points = 25000
    n_fixed_coefficients = 2
    n_random_coefficients = 2

    args = {
        "max_time": 1,
        "max_steps": 20000,
        "lambda": 10,
        "tau": 2.5,
        "learning_rate": 0.01,
        "n_coefficients": n_fixed_coefficients + n_random_coefficients,
        "alpha": 1.6
    }
    
    n_coefficients = n_fixed_coefficients + n_random_coefficients
    X, y, X_norm, y_norm, fixed_coefficients = generate_data(int(n_data_points), n_fixed_coefficients + n_random_coefficients, n_fixed_coefficients)
    true_coefficients = np.concatenate((fixed_coefficients, np.zeros(n_random_coefficients)))
    print("True Coefficients:", true_coefficients)

    algorithms = {
        "S-LCA": run_slca,
        "Fista": run_fista
    }

    results = {}

    for name, algorithm in algorithms.items():
        print(f"Running {name}...")
        coeffs, times = algorithm(X_norm, y_norm, args)
        errors = [get_error(X, y, cs) for cs in coeffs.T]
        results[name] = {"coeffs": coeffs, "times": times, "errors": errors}
        print(f"{name} Coefficients:", results[name]["coeffs"][:, -1])
        print(f"{name} Diff:", diff(results[name]["coeffs"][:, -1], true_coefficients))
        print(f"{name} Error:", results[name]["errors"][-1])

    save_lineplot(results, n_fixed_coefficients, true_coefficients, max([result["coeffs"].shape[1] for result in results.values()]))
    save_timeplot(results)