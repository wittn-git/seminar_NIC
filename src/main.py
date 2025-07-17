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

def print_results(true_coefficients, results):
    print("True Coefficients:", true_coefficients)
    for name in results.keys():
        print(f"{name} Coefficients:", results[name]["coeffs"][:, -1])
        print(f"{name} Diff:", diff(results[name]["coeffs"][:, -1], true_coefficients))
        print(f"{name} Error:", results[name]["errors"][-1])
        print()

def run_experiment(seed, n_data_points, n_fixed_coefficients, n_random_coefficients, algorithms, args):
    np.random.seed(seed)

    X, y, X_norm, y_norm, fixed_coefficients = generate_data(int(n_data_points), n_fixed_coefficients + n_random_coefficients, n_fixed_coefficients)
    true_coefficients = np.concatenate((fixed_coefficients, np.zeros(n_random_coefficients)))
    error_func = lambda coeffs: get_error(X, y, coeffs)

    results = {}
    for name, algorithm in algorithms.items():
        coeffs, times, errors = algorithm(X_norm, y_norm, error_func, args)
        results[name] = {"coeffs": coeffs, "times": times, "errors": errors}
    
    return results
    
if __name__ == "__main__":

    seeds = [23655]

    n_data_points = 10000
    n_fixed_coefficients = 2
    n_random_coefficients = 2
    n_coefficients = n_fixed_coefficients + n_random_coefficients

    algorithms = {
        # TODO add third method
        "S-LCA": run_slca,
        "Fista": run_fista
    }

    args = {
        "max_time": 1,
        "max_steps": 25000,
        "n_coefficients": n_fixed_coefficients + n_random_coefficients
    }

    result_collection = []
    for seed in seeds:
        results = run_experiment(seed, n_data_points, n_fixed_coefficients, n_random_coefficients, algorithms, args)
        result_collection.append(results)    

    save_timeplot(result_collection, args["max_time"], algorithms.keys())