from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
def perform_statistical_tests(data, lags=20, name="Time Series"):
    """
    Perform Ljung-Box (LB) Test and Augmented Dickey-Fuller (ADF) Test on time-series data.

    Parameters:
        data (numpy array or pandas Series): The time-series data.
        lags (int): Number of lags for the LB test.

    Returns:
        None
    """
    # ADF Test
    print(f"Performing Augmented Dickey-Fuller (ADF) Test for {name}...")
    adf_result = adfuller(data)
    print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")
    print("Critical Values:")
    for key, value in adf_result[4].items():
        print(f"   {key}: {value}")
    if adf_result[1] < 0.05:
        print("The series is likely stationary (reject null hypothesis).")
    else:
        print("The series is likely non-stationary (fail to reject null hypothesis).")
    print("-" * 50)

    # Ljung-Box Test
    print(f"Performing Ljung-Box (LB) Test for {name}...")
    lb_result = acorr_ljungbox(data, lags=[lags], return_df=True)
    print(lb_result)
    if lb_result['lb_pvalue'].iloc[0] < 0.05:
        print("The residuals are likely not white noise (reject null hypothesis).")
    else:
        print("The residuals are likely white noise (fail to reject null hypothesis).")
    print("-" * 50)

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    t = np.linspace(0, 20, 500)
    data = np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, len(t))

    # 调用函数进行统计测试
    perform_statistical_tests(data, lags=20)
