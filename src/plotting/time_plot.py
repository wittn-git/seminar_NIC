import numpy as np
import matplotlib.pyplot as plt

def save_timeplot(results):
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 18})
    for name, result in results.items():
        plt.plot(result["times"], result["errors"], label=name)
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.title('Prediction Error vs Time on Training Data')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/time_plot.png')
    plt.close()