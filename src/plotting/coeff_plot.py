import numpy as np
import matplotlib.pyplot as plt

def save_lineplot(results, n_fixed_coefficients, true_coefficients, timesteps):

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
        for name, result in results.items():
            axes[i].plot(result["coeffs"][i, :], label=name)
        axes[i].legend(fontsize=12, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('plots/coeff_plot.png')
    plt.close(fig)