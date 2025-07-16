import numpy as np
import matplotlib.pyplot as plt

def save_timeplot(results):
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 18})
    for name, result in results.items():
        plt.plot(result["times"], result["errors"], label=name)
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('MSE')
    plt.title('MSE vs Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/time_plot.png')
    plt.close()