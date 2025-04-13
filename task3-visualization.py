import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_process import ArmaProcess

# Generate Fibonacci series with noise
def generate_fibonacci_with_noise(length, noise_ratio=0.05):
    fibonacci_series = [0, 1]
    for i in range(2, length):
        fibonacci_series.append(fibonacci_series[-1] + fibonacci_series[-2])
    fibonacci_series = np.array(fibonacci_series, dtype=float)
    noise = np.random.normal(0, noise_ratio * np.std(fibonacci_series), length)
    noisy_series = fibonacci_series + noise
    return fibonacci_series, noisy_series

# Statistical analysis
def statistical_analysis(series, name):
    mean = np.mean(series)
    std_dev = np.std(series)
    variance = np.var(series)
    print(f'{name} - Mean: {mean}, Standard Deviation: {std_dev}, Variance: {variance}')

# Stationarity Check
def check_stationarity(series, name):
    result = adfuller(series)
    print(f'{name} - ADF Statistic: {result[0]}, p-value: {result[1]}')
    if result[1] <= 0.05:
        print(f'{name} is stationary.')
    else:
        print(f'{name} is not stationary.')

# Visualization
def visualize_data(noisy_series):
    # Line Plot
    plt.figure(figsize=(12, 10))
    plt.subplot(221)
    plt.plot(noisy_series, color='blue')
    plt.title('Line Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Histogram
    plt.subplot(222)
    plt.hist(noisy_series, bins=20, color='purple', alpha=0.7)
    plt.title('Histogram')

    # Density Plot
    plt.subplot(223)
    pd.Series(noisy_series).plot(kind='density', color='green')
    plt.title('Density Plot')

    # Box Plot
    plt.subplot(224)
    plt.boxplot(noisy_series, vert=False)
    plt.title('Box Plot')

    plt.tight_layout()
    plt.savefig('pics/task3_line_plot.png', dpi=350)
    plt.show()

# Lag Plots
def draw_lag_plots(noisy_series):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    lag_plot(pd.Series(noisy_series), lag=1)
    plt.title('Lag-1 Plot')

    plt.subplot(122)
    lag_plot(pd.Series(noisy_series), lag=2)
    plt.title('Lag-2 Plot')

    plt.tight_layout()
    plt.savefig('pics/task3_lag_plot.png', dpi=350)
    plt.show()

# ACF and PACF Plots
def draw_acf_pacf(noisy_series):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plot_acf(noisy_series, lags=20, ax=plt.gca())
    plt.title('ACF Plot')

    plt.subplot(122)
    plot_pacf(noisy_series, lags=20, ax=plt.gca(), method='ols')
    plt.title('PACF Plot')

    plt.tight_layout()
    plt.savefig('pics/task3_ACF_plot.png', dpi=350)
    plt.show()

# Main Execution
length = 50
fibonacci_series, noisy_series = generate_fibonacci_with_noise(length)

# Statistical Analysis
statistical_analysis(noisy_series, 'Noisy Fibonacci Series')

# Check Stationarity
check_stationarity(noisy_series, 'Noisy Fibonacci Series')

# Visualizations
visualize_data(noisy_series)
draw_lag_plots(noisy_series)
draw_acf_pacf(noisy_series)