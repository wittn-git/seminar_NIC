import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import os
import math
import matplotlib.pyplot as plt

def process_results(result_collection, max_time, algorithms, exclude_first):
    common_time = np.linspace(0, max_time, num=500)
    processed_results = {}

    for name in algorithms:
        interpolated_runs = []

        for result in result_collection:
            run_times = result[name]["times"]
            run_errors = result[name]["errors"]

            f = interp1d(run_times, run_errors, bounds_error=False, fill_value='extrapolate')
            interpolated_errors = f(common_time)
            interpolated_runs.append(interpolated_errors)

        interpolated_runs = np.array(interpolated_runs)
        mean_errors = np.mean(interpolated_runs, axis=0)

        common_time_ = common_time
        if exclude_first:
            common_time_ = common_time_[1:]
            mean_errors = mean_errors[1:]

        processed_results[name] = {
            "times": common_time_,
            "errors": mean_errors,
        }

    return processed_results

def save_timeplot(result_collections, titles, max_time, algorithms, n_cols, exclude_first):

    n_plots = len(result_collections)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 5 * n_rows))
    plt.rcParams.update({'font.size': 20})

    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for idx, (result_collection, title) in enumerate(zip(result_collections, titles)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        processed_results = process_results(result_collection, max_time, algorithms, exclude_first)

        for name, result in processed_results.items():
            ax.plot(result["times"], result["errors"], label=name)

        #ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Time (s)', fontsize=18)
        ax.set_ylabel('MSE', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_title(title)
        ax.legend()

    for idx in range(n_plots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row][col])

    plt.tight_layout()
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(f'plots/time_plots_grid.png')
    plt.savefig(f'plots/time_plots_grad.svg')
    plt.close()