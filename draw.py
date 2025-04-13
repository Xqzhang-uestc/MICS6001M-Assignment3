import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def draw_plots(data, lags=40, name="time_series"):
    """
    Perform exploratory data analysis on time-series data and save plots.

    Parameters:
        data (numpy array or pandas Series): The time-series data.
        lags (int): Number of lags for ACF and PACF plots.
        name (str): Base name for the plots.

    Returns:
        None
    """
    # Create a folder to save the plots
    output_dir = os.path.join("pics", name)
    os.makedirs(output_dir, exist_ok=True)

    # Convert data to pandas Series if it's not already
    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    # Line Plot
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Time Series", color="blue")
    plt.title(f"{name} - Line Plot")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{name}_line_plot.png"))
    plt.close()

    # Box Plot
    plt.figure(figsize=(6, 6))
    sns.boxplot(data=data, color="orange")
    plt.title(f"{name} - Box Plot")
    plt.savefig(os.path.join(output_dir, f"{name}_box_plot.png"))
    plt.close()

    # Histogram
    plt.figure(figsize=(12, 6))
    sns.histplot(data, kde=False, bins=30, color="purple")
    plt.title(f"{name} - Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, f"{name}_histogram.png"))
    plt.close()

    # Density Plot
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data, color="green")
    plt.title(f"{name} - Density Plot")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.savefig(os.path.join(output_dir, f"{name}_density_plot.png"))
    plt.close()

    # Lag-1 Plot
    plt.figure(figsize=(6, 6))
    pd.plotting.lag_plot(data, lag=1)
    plt.title(f"{name} - Lag-1 Plot")
    plt.savefig(os.path.join(output_dir, f"{name}_lag1_plot.png"))
    plt.close()

    # ACF Plot
    plt.figure(figsize=(12, 6))
    plot_acf(data, lags=lags, alpha=0.05)
    plt.title(f"{name} - Autocorrelation Function (ACF)")
    plt.savefig(os.path.join(output_dir, f"{name}_acf_plot.png"))
    plt.close()

    # PACF Plot
    plt.figure(figsize=(12, 6))
    plot_pacf(data, lags=lags, alpha=0.05, method='ywm')
    plt.title(f"{name} - Partial Autocorrelation Function (PACF)")
    plt.savefig(os.path.join(output_dir, f"{name}_pacf_plot.png"))
    plt.close()

if __name__ == "__main__":
    # Example time-series data (replace with your own data)
    import numpy as np
    np.random.seed(42)
    time_series_data = np.random.randn(1000).cumsum()  # Random walk
    
    # Perform exploratory data analysis
    draw_plots(time_series_data, lags=40, name="random_walk")