import numpy as np
import matplotlib.pyplot as plt
from fista import run_fista
from coordinate_descent import run_coorddesc
from slca import run_slca

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

def save_lineplot(results, n_fixed_coefficients, true_coefficients, timesteps, file_name='spike_rates_plot.png', show=True):

    n_coefficients = len(true_coefficients)
    fig, axes = plt.subplots(nrows=max(n_fixed_coefficients, len(true_coefficients) - n_fixed_coefficients), ncols=2, figsize=(10*1.75, 3 * n_coefficients), sharex=True)

    axes = axes.flatten() if n_coefficients > 1 else [axes]

    plt.rcParams.update({'font.size': 18})
    
    for i in range(n_coefficients):
        axes[i].tick_params(axis='both', which='major', labelsize=14)
        axes[i].axhline(y=true_coefficients[i], color='r', linestyle='--', label='True Coefficient')
        if i == n_coefficients - 1 or i == n_fixed_coefficients:
            axes[i].set_xlabel('Time Steps', fontsize=18)
        if i % 2 == 0:
            axes[i].set_ylabel('Spike Rate', fontsize=18)
        axes[i].set_ylim(-0.1, 1.1)
        axes[i].set_xticks(np.arange(0, timesteps, timesteps/10))
        axes[i].set_title(f'Coefficient {i+1}', fontsize=18)
        for name, coefficients in results.items():
            axes[i].plot(coefficients[i, :], label=name)
        axes[i].legend(fontsize=12, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(file_name)
    if show:
        plt.show()
    plt.close(fig)

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
        "n_coefficients": n_fixed_coefficients + n_random_coefficients
    }
    
    n_coefficients = n_fixed_coefficients + n_random_coefficients
    X, y, fixed_coefficients = generate_data(int(n_data_points), n_fixed_coefficients + n_random_coefficients, n_fixed_coefficients)
    true_coefficients = np.concatenate((fixed_coefficients, np.zeros(n_random_coefficients)))
    print("True Coefficients:", true_coefficients)

    algorithms = {
        "S-LCA": run_slca,
        "Coordinate Descent": run_coorddesc,
        "Fista": run_fista
    }

    results = {}

    for name, algorithm in algorithms.items():
        print(f"Running {name}...")
        results[name] = algorithm(X, y, args)
        print(f"{name} Coefficients:", results[name][:, -1])
        print(f"{name} Diff:", diff(results[name][:, -1], true_coefficients))

    save_lineplot(results, n_fixed_coefficients, true_coefficients, args["time_steps"], f'plots/results.png', show=False)