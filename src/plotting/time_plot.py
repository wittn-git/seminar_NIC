import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def process_results(result_collection, max_time, algorithms):
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

        processed_results[name] = {
            "times": common_time,
            "errors": mean_errors
        }

    return processed_results

def save_timeplot(result_collection, max_time, algorithms):

    processed_results = process_results(result_collection, max_time, algorithms)

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 18})

    for name, result in processed_results.items():
        plt.plot(result["times"], result["errors"], label=name)

    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('MSE')
    plt.title('MSE vs Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/time_plot.png')
    plt.close()