from algorithms.fista import run_fista
from algorithms.ista import run_ista
from algorithms.slca import run_slca
from plotting.coeff_plot import save_lineplot
from plotting.time_plot import save_timeplot
import numpy as np
import os
import pandas as pd

def generate_data(n, p, n_fixed_coefficients):
    
    fixed_coefficients = np.random.uniform(0, 0.5, size=n_fixed_coefficients)

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
    
    X_std = np.std(X, axis=0)
    X_norm = (X - np.mean(X, axis=0)) / np.where(X_std == 0, 1, X_std)
    y_std = np.std(y)
    y_norm = (y - np.mean(y)) / (y_std if y_std != 0 else 1)
    
    return X, y, X_norm, y_norm, fixed_coefficients

def diff(estimated_coefficients, coefficients):
    return sum([a - b for a, b in zip(estimated_coefficients, coefficients)]) / len(coefficients)

def get_error(X, y, coefficients):
    y_pred = X @ coefficients
    return np.mean((y_pred - y) ** 2)

def record_results(result_collection, seeds, algorithm_names, file_name):
    result_folder = "results"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    with open(f"{result_folder}/{file_name}.txt", "w") as f:
        for i, results in enumerate(result_collection):
            f.write(f"Run {i + 1} (Seed: {seeds[i]}):\n")
            for name in algorithm_names:
                f.write(f"  {name} Error: {results[name]['errors'][-1]}\n")
            f.write("\n")
        f.write("Average and Standard Deviation of Last Errors:\n")
        for name in algorithm_names:
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
        _, times, errors = algorithm(X_norm, y_norm, error_func, args)
        results[name] = {"times": times, "errors": errors}
    
    return results, true_coefficients

def get_seed(i, j):
    return (i + 1) * (j + 5) * 150 + 16 * i + 4

def to_csv(result_collections, coefficients):
    data = []
    for i, result_collection in enumerate(result_collections):
        for j, results in enumerate(result_collection):
            for algorithm, result in results.items():
                for k, time in enumerate(result['times']):
                    data.append({
                        "run": j,
                        "seed": get_seed(i, j),
                        "algorithm": algorithm,
                        "time": time,
                        "error": result['errors'][k]
                    })
    if not os.path.exists("results"):
        os.makedirs("results")
    df = pd.DataFrame(data)
    df.to_csv("results/results.csv", index=False)
    
if __name__ == "__main__":

    # experimental setup
    n_runs = 25
    coefficient_pairs = [
        (5, 500),
        (10, 500),
        (5, 1000),
        (25, 1000)
    ]
    
    # other parameters
    n_data_points = 10000
    algorithms = {
        "S-LCA": run_slca,
        "FISTA": run_fista,
        "ISTA": run_ista,
    }
    args = {
        "max_time": 0.05,
        "max_steps": 50000
    }

    result_collections, titles = [], []
    for i, (n_fixed_coefficients, n_random_coefficients) in enumerate(coefficient_pairs):
        result_collection, seeds = [], []
        for j in range(n_runs):
            seed = get_seed(i, j)
            seeds.append(seed)
            args["n_coefficients"] = n_fixed_coefficients + n_random_coefficients
            results, true_coefficients = run_experiment(seed, n_data_points, n_fixed_coefficients, n_random_coefficients, algorithms, args)
            result_collection.append(results) 
        result_collections.append(result_collection)
        titles.append(f"{n_fixed_coefficients} fixed, {n_random_coefficients} random coefficients")
        record_results(result_collection, seeds, algorithms.keys(), f'results_{n_fixed_coefficients}_{n_random_coefficients}')
    save_timeplot(result_collections, titles, args["max_time"], algorithms.keys(), 2, True)
    to_csv(result_collections, coefficient_pairs)