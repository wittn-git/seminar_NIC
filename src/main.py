import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from fista import Fista

def get_data(n_data_points, n_fixed_coefficients, n_random_coefficients):

    fixed_coefficients = np.random.uniform(0, 1, size=n_fixed_coefficients)
    fixed_coefficients = fixed_coefficients / np.sum(fixed_coefficients)

    X, y = [], []
    for _ in range(n_data_points):
        x = np.random.uniform(0, 1, size=n_fixed_coefficients + n_random_coefficients)
        x = x / np.sum(x)
        X.append(x)
        y.append(sum(c * x_val for c, x_val in zip(fixed_coefficients, x[:len(fixed_coefficients)])))
    return np.array(X), np.array(y), fixed_coefficients

def get_spike_rates(X, y, n_coefficients, time_steps, lambda_=0.2, tau=10):

    X = X / np.linalg.norm(X)

    v_reset = 0
    v_threshold=1

    spikes = np.zeros((n_coefficients, time_steps))
    filtered_spikes = np.zeros((n_coefficients, time_steps))

    b = X.T @ y
    w = X.T @ X    

    v = np.zeros(n_coefficients)
    for t in range(time_steps):
        if t > 0:
            filtered_spikes[:, t] = filtered_spikes[:, t - 1] * np.exp(-1 / tau) + spikes[:, t - 1]
        mu = b - w @ filtered_spikes[:, t]
        v += np.clip(mu - lambda_, 0, None)
        spikes[:, t] = (v >= v_threshold).astype(float)
        v[spikes[:, t] == 1] = v_reset

    spike_rates = np.zeros((n_coefficients, time_steps))
    for i in range(n_coefficients):
        for t in range(time_steps):
            spike_rates[i, t] = np.sum(spikes[i, :t + 1]) / (t + 1) if t > 0 else spikes[i, t]
            
    return spike_rates

def save_lineplot(spike_rates, n_fixed_coefficients, true_coefficients, file_name='spike_rates_plot.png', show=True):

    n_coefficients = spike_rates.shape[0]
    fig, axes = plt.subplots(nrows=max(n_fixed_coefficients, len(true_coefficients) - n_fixed_coefficients), ncols=2, figsize=(10*1.75, 2 * n_coefficients), sharex=True)

    axes = axes.flatten() if n_coefficients > 1 else [axes]

    plt.rcParams.update({'font.size': 18})
    
    for i in range(n_coefficients):
        axes[i].tick_params(axis='both', which='major', labelsize=14)
        axes[i].plot(spike_rates[i, :], label=f'Coefficient {i+1}')
        if i == 0:
            axes[i].axhline(y=true_coefficients[i], color='r', linestyle='--', label='True Coefficient')
        else:
            axes[i].axhline(y=true_coefficients[i], color='r', linestyle='--')
        if i == n_coefficients - 1 or i == n_fixed_coefficients:
            axes[i].set_xlabel('Time Steps', fontsize=18)
        if i % 2 == 0:
            axes[i].set_ylabel('Spike Rate', fontsize=18)
        axes[i].set_ylim(-0.1, 1.1)
        axes[i].set_xticks(np.arange(0, spike_rates.shape[1], spike_rates.shape[1]/10))
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(file_name)
    if show:
        plt.show()
    plt.close(fig)

def error(estimated_coefficients, test_X, test_y):
    predictions =  test_X @ estimated_coefficients
    return np.mean((predictions - test_y) ** 2)

if __name__ == "__main__":

    np.random.seed(23655)

    n_data_points = 10000
    n_fixed_coefficients = 2
    n_random_coefficients = 2
    time_steps = 500
    lambda_ = 6
    tau = 25

    n_coefficients = n_fixed_coefficients + n_random_coefficients
    X, y, fixed_coefficients = get_data(int(n_data_points * 1.1), n_fixed_coefficients, n_random_coefficients)
    true_coefficients = np.concatenate((fixed_coefficients, np.zeros(n_random_coefficients)))
    train_X, train_y = X[:n_data_points], y[:n_data_points]
    test_X, test_y = X[n_data_points:], y[n_data_points:]

    spike_rates = get_spike_rates(train_X, train_y, n_coefficients, time_steps, lambda_=lambda_, tau=tau)

    save_lineplot(spike_rates, n_fixed_coefficients, true_coefficients, f'results.png', show=False)
    print("True Coefficients:", true_coefficients)
    print("SLCA coefficients:", spike_rates[:, -1])
    print("SLCA Error:", error(spike_rates[:, -1], test_X, test_y))

    # run fista
    #fista = Fista(loss='squared-hinge', penalty='l11', lambda_=lambda_, n_iter=50)
    #fista.fit(X, y)
    #print("FISTA coefficients:", fista.coef_)
    #print("FISTA Error:", error(fista.coef_, true_coefficients))

    # use sklearn
    model = Lasso(alpha=lambda_)
    model.fit(X, y)
    print("SKL coefficients:", model.coef_)
    print("SKL Error:", error(model.coef_, test_X, test_y))