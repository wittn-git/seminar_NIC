from algorithms.fista import run_fista
from algorithms.ista import run_ista
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

def record_results(result_collection, true_coefficients_collection, seeds, algorithms, file_name):
    with open(file_name, "w") as f:
        for i, results in enumerate(result_collection):
            f.write(f"Run {i + 1} (Seed: {seeds[i]}):\n")
            f.write(f"  True Coefficients: {true_coefficients_collection[i]}\n")
            for name in algorithms.keys():
                f.write(f"  {name} Coefficients: {results[name]['coeffs'][:, -1]}\n")
                f.write(f"  {name} Error: {results[name]['errors'][-1]}\n")
            f.write("\n")
        f.write("Average and Standard Deviation of Last Errors:\n")
        for name in algorithms.keys():
            last_errors = [result[name]['errors'][-1] for result in result_collection]
            avg_error = np.mean(last_errors)
            std_error = np.std(last_errors)
            f.write(f"  {name}: Avg Error: {avg_error}, Std Error: {std_error}\n")

def run_experiment(seed, n_data_points, n_fixed_coefficients, n_random_coefficients, algorithms, args):
    np.random.seed(seed)

    X, y, X_norm, y_norm, fixed_coefficients = generate_data(int(n_data_points), n_fixed_coefficients + n_random_coefficients, n_fixed_coefficients)
    true_coefficients = np.concatenate((fixed_coefficients, np.zeros(n_random_coefficients)))
    error_func = lambda coeffs: get_error(X, y, coeffs)

    results = {}
    for name, algorithm in algorithms.items():
        coeffs, times, errors = algorithm(X_norm, y_norm, error_func, args)
        results[name] = {"coeffs": coeffs, "times": times, "errors": errors}
    
    return results, true_coefficients

def get_seed(i, j):
    return (i + 1) * (j + 5) * 150 + 16 * i + 4
    
if __name__ == "__main__":

    # experimental setup
    n_runs = 10
    coefficient_pairs = [
        (2, 2),
        (2, 0),
        (0, 2),
        (4, 4)
    ]
    
    # other parameters
    n_data_points = 10000
    algorithms = {
        "S-LCA": run_slca,
        "FISTA": run_fista,
        "ISTA": run_ista,
    }
    args = {
        "max_time": 0.5,
        "max_steps": 50000
    }

    result_collections, titles = [], []
    for i, (n_fixed_coefficients, n_random_coefficients) in enumerate(coefficient_pairs):
        result_collection, true_coefficients_collection, seeds = [], [], []
        for j in range(n_runs):
            seed = get_seed(i, j)
            seeds.append(seed)
            args["n_coefficients"] = n_fixed_coefficients + n_random_coefficients
            results, true_coefficients = run_experiment(seed, n_data_points, n_fixed_coefficients, n_random_coefficients, algorithms, args)
            result_collection.append(results) 
            true_coefficients_collection.append(true_coefficients)
        result_collections.append(result_collection)
        titles.append(f"{n_fixed_coefficients} fixed, {n_random_coefficients} random coefficients")
        record_results(result_collection, true_coefficients_collection, seeds, algorithms.keys(), f'results_{n_fixed_coefficients}_{n_random_coefficients}')
    save_timeplot(result_collections, titles, args["max_time"], algorithms.keys(), 2)
