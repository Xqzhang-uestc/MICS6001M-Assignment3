import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Input
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import tensorflow as tf
from statsmodels.tsa.arima_process import ArmaProcess
from pmdarima import auto_arima

# --- Utility Functions ---
def generate_fibonacci(length):
    fib = [0, 1]
    for i in range(2, length):
        fib.append(fib[-1] + fib[-2])
    return np.array(fib, dtype=float)

def add_noise(signal, noise_ratio):
    noise = np.random.normal(0, noise_ratio * np.std(signal), len(signal))
    return signal + noise

def create_supervised_data(series, input_len):
    X, y = [], []
    for i in range(len(series) - input_len):
        X.append(series[i:i + input_len])
        y.append(series[i + input_len])
    return np.array(X), np.array(y)

def evaluate(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return mse, mae, mape

def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(8, 4))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Predicted")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Main Execution ---
length = 50
input_len = 30
noise_levels = [0.05, 0.1, 0.2]
models = ["MLP", "RNN", "LSTM", "ARIMA"]
metrics = {model: {"MSE": [], "MAE": [], "MAPE": []} for model in models}

fib = generate_fibonacci(length)

for noise_ratio in noise_levels:
    # Add noise and scale data
    fib_noisy = add_noise(fib, noise_ratio)
    scaler = MinMaxScaler()
    fib_scaled = scaler.fit_transform(fib_noisy.reshape(-1, 1)).flatten()
    X, y = create_supervised_data(fib_scaled, input_len)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    X_train_nn = X_train.reshape((X_train.shape[0], input_len, 1))
    X_test_nn = X_test.reshape((X_test.shape[0], input_len, 1))

    # --- MLP ---#
    mlp = Sequential([Dense(64, activation='relu', input_shape=(input_len,)),
                      Dense(128, activation='relu'),
                      Dense(1)])
    mlp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    mlp_history = mlp.fit(X_train, y_train, epochs=200, verbose=0)
    y_pred_mlp = mlp.predict(X_test).flatten()
    mse, mae, mape = evaluate(y_test, y_pred_mlp, "MLP")
    print("MLP", mse, mae, mape)
    metrics["MLP"]["MSE"].append(mse)
    metrics["MLP"]["MAE"].append(mae)
    metrics["MLP"]["MAPE"].append(mape)
    plot_predictions(y_test, y_pred_mlp, f"MLP Prediction (Noise {noise_ratio*100:.0f}%)")

    # --- RNN ---#
    rnn = Sequential([
        SimpleRNN(64, activation='tanh', return_sequences=True, input_shape=(input_len, 1)),
        SimpleRNN(64, activation='tanh'),
        Dense(1)
    ])
    rnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse')
    rnn_history = rnn.fit(X_train_nn, y_train, epochs=200, verbose=0)
    y_pred_rnn = rnn.predict(X_test_nn).flatten()
    mse, mae, mape = evaluate(y_test, y_pred_rnn, "RNN")
    print("RNN", mse, mae, mape)
    metrics["RNN"]["MSE"].append(mse)
    metrics["RNN"]["MAE"].append(mae)
    metrics["RNN"]["MAPE"].append(mape)
    plot_predictions(y_test, y_pred_rnn, f"RNN Prediction (Noise {noise_ratio * 100:.0f}%)")

    # --- LSTM ---#
    lstm = Sequential([
        LSTM(64, return_sequences=True, input_shape=(input_len, 1)),
        LSTM(128),
        Dense(1)
    ])
    lstm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse')
    lstm_history = lstm.fit(X_train_nn, y_train, epochs=200, verbose=0)
    y_pred_lstm = lstm.predict(X_test_nn).flatten()
    mse, mae, mape = evaluate(y_test, y_pred_lstm, "LSTM")
    print("LSTM", mse, mae, mape)
    metrics["LSTM"]["MSE"].append(mse)
    metrics["LSTM"]["MAE"].append(mae)
    metrics["LSTM"]["MAPE"].append(mape)
    plot_predictions(y_test, y_pred_lstm, f"LSTM Prediction (Noise {noise_ratio * 100:.0f}%)")

    # --- ARIMA ---#
    # --- Stationarity Check ---#
    def check_stationarity(series, name):
        result = adfuller(series)
        print(f'{name} - ADF Statistic: {result[0]}, p-value: {result[1]}')
        if result[1] <= 0.05:
            print(f'{name} is stationary.')
        else:
            print(f'{name} is not stationary. Consider differencing.')

    # Convert to Series
    series_arima = pd.Series(fib_noisy)

    # --- Step 1: Log-transform ---#
    log_series = np.log(np.clip(series_arima, a_min=1e-6, a_max=None))  # 避免 log(0) 或负值
    check_stationarity(log_series, "Log Series")

    # --- Step 2: Differencing ---#
    log_diff = log_series.diff().dropna()
    check_stationarity(log_diff, "Log Differenced Series")
    plt.plot(log_diff)

    # --- Step 3: Plot ACF/PACF ---#
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_acf(log_diff, lags=20, ax=axes[0])
    axes[0].set_title('ACF of Log-Differenced Series')
    plot_pacf(log_diff, lags=20, ax=axes[1])
    axes[1].set_title('PACF of Log-Differenced Series')
    plt.savefig(f'pics/task3_logACF_plot(Noise {noise_ratio*100:.0f}%).png', dpi=350)
    plt.show()

    # --- Step 4: Fit ARIMA on log-transformed series ---#
    model_arima = ARIMA(log_series[:train_size + input_len], order=(2, 1, 1))
    fit_arima = model_arima.fit()

    # --- Step 5: Forecast & inverse log-transform ---#
    forecast_log = fit_arima.forecast(steps=len(y_test))
    forecast_arima = np.exp(forecast_log)  # 还原回原始尺度

    # --- Step 6: Scale & Evaluate ---#
    scaled_arima = scaler.transform(forecast_arima.values.reshape(-1, 1)).flatten()

    mse, mae, mape = evaluate(y_test, scaled_arima, "ARIMA")
    print("ARIMA", mse, mae, mape)
    metrics["ARIMA"]["MSE"].append(mse)
    metrics["ARIMA"]["MAE"].append(mae)
    metrics["ARIMA"]["MAPE"].append(mape)
    plot_predictions(y_test, scaled_arima, f"ARIMA Prediction (Noise {noise_ratio * 100:.0f}%)")

    # --- Plot Training Curves ---#
    plt.figure(figsize=(10, 5))
    plt.plot(mlp_history.history['loss'], label="MLP Loss")
    plt.plot(rnn_history.history['loss'], label="RNN Loss")
    plt.plot(lstm_history.history['loss'], label="LSTM Loss")
    plt.title(f"Training Loss Curves (Noise {noise_ratio*100:.0f}%)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'pics/task3_training_loss(Noise {noise_ratio*100:.0f}%).png', dpi=350)
    plt.show()

    # --- Plot Predictions vs True Values ---#
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='True', color='black')
    plt.plot(y_pred_mlp, '--', label='MLP', color='blue')
    plt.plot(y_pred_rnn, '--', label='RNN', color='green')
    plt.plot(y_pred_lstm, '--', label='LSTM', color='orange')
    plt.plot(scaler.transform(forecast_arima.values.reshape(-1, 1)), '--', label='ARIMA', color='red')
    plt.legend()
    plt.title(f"Model Predictions vs True Values (Noise {noise_ratio*100:.0f}%)")
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Values")
    plt.grid(True)
    plt.savefig(f'pics/task3_model_predictions(Noise {noise_ratio*100:.0f}%).png', dpi=350)
    plt.show()

# --- Plot Performance Metrics ---#
for metric in ["MSE", "MAE", "MAPE"]:
    plt.figure(figsize=(10, 5))
    for model in models:
        plt.plot([n * 100 for n in noise_levels], metrics[model][metric], label=model)
    plt.title(f"{metric} vs Noise Level")
    plt.xlabel("Noise Level (%)")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'pics/task3_{metric}_vs_noise_level.png', dpi=350)
    plt.show()