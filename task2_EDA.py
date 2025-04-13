from draw import draw_plots
from statistical_test import perform_statistical_tests
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
# Set random seed for reproducibility
np.random.seed(42)
# Generate White Noise
def generate_white_noise(n=1000):
    return np.random.normal(0, 1, n)

# Generate Random Walk
def generate_random_walk(n=1000):
    return np.cumsum(np.random.normal(0, 1, n))

# Generate ARMA(2, 2) Process

def generate_arma_process(n=1000):
    # ARMA(2, 2) coefficients (ensure stationarity)
    ar = np.array([1, -0.7, 0.25])  # AR coefficients
    ma = np.array([1, 0.4, -0.3])   # MA coefficients
    arma_process = ArmaProcess(ar, ma)
    return arma_process.generate_sample(nsample=n)

# Generate datasets
white_noise = generate_white_noise()
random_walk = generate_random_walk()
arma_process = generate_arma_process()

# # Plot the generated datasets
# draw_plots(white_noise, name="White Noise", lags=40)
# draw_plots(random_walk, name="Random Walk", lags=40)
draw_plots(arma_process, name="New ARMA Process", lags=40)


# Perform statistical tests
perform_statistical_tests(white_noise, lags=40, name="White Noise")
perform_statistical_tests(random_walk, lags=40, name="Random Walk")
perform_statistical_tests(arma_process, lags=40, name="ARMA Process")
