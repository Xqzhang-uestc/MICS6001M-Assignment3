import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense
from sklearn.model_selection import train_test_split
from reshape import reshape_to_supervised
from draw import draw_plots
from statistical_test import perform_statistical_tests

# Generate a sinusoidal wave dataset with white noise
def generate_noisy_sinusoidal_wave(period=20, sample_rate=100, num_periods=3, noise_amplitude=0.1):
    """
    Generate a sinusoidal wave dataset with added white noise.

    Parameters:
        period (float): Period of the sinusoidal wave in seconds.
        sample_rate (int): Sampling rate in Hz.
        num_periods (int): Number of periods to generate.
        noise_amplitude (float): Amplitude of the added white noise.

    Returns:
        t (numpy array): Time array.
        data (numpy array): Generated sinusoidal wave data with noise.
    """
    t = np.linspace(0, num_periods * period, num_periods * sample_rate * period, endpoint=False)
    clean_data = np.sin(2 * np.pi * t / period)
    noise = np.random.normal(0, 1, len(clean_data)) * noise_amplitude
    noisy_data = clean_data + noise
    return t, noisy_data

# Generate the noisy sinusoidal wave data
t, data = generate_noisy_sinusoidal_wave(period=20, sample_rate=100, num_periods=4, noise_amplitude=0.1)

# Perform statistical tests on the generated data
draw_plots(data, lags=40, name="noisy_sinusoidal_wave")  # Draw plots for the generated data
perform_statistical_tests(data, lags=40)  # Perform statistical tests on the generated data

# Reshape the data for supervised learning
n_input = 8  # Input vector size (number of time steps)
n_output = 2  # Output vector size (two-step prediction)
X, y = reshape_to_supervised(data, n_input, n_output)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set activation function and epoch as variables
activation_function = "sigmoid"  # Activation function
epoch = 100  # Number of epochs

# Design the RNN model
def build_rnn_model(n_input, n_output, activation_function):
    model = Sequential([
        SimpleRNN(50, activation=activation_function, input_shape=(n_input, 1)),
        Dense(n_output)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Design the LSTM model
def build_lstm_model(n_input, n_output, activation_function):
    model = Sequential([
        LSTM(50, activation=activation_function, input_shape=(n_input, 1)),
        Dense(n_output)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train and evaluate the RNN model
rnn_model = build_rnn_model(n_input, n_output, activation_function)
rnn_model.fit(X_train, y_train, epochs=epoch, batch_size=16, verbose=1)
rnn_loss, rnn_mae = rnn_model.evaluate(X_test, y_test, verbose=0)
print(f"RNN - Activation: {activation_function}, Epochs: {epoch}, Test Loss (MSE): {rnn_loss}, Test MAE: {rnn_mae}")

# Train and evaluate the LSTM model
lstm_model = build_lstm_model(n_input, n_output, activation_function)
lstm_model.fit(X_train, y_train, epochs=epoch, batch_size=16, verbose=1)
lstm_loss, lstm_mae = lstm_model.evaluate(X_test, y_test, verbose=0)
print(f"LSTM - Activation: {activation_function}, Epochs: {epoch}, Test Loss (MSE): {lstm_loss}, Test MAE: {lstm_mae}")

# Save results to a txt file
output_file = "1-4-output.txt"
with open(output_file, "a") as f:
    f.write(f"RNN - Activation: {activation_function}, Epochs: {epoch}, Test Loss (MSE): {rnn_loss}, Test MAE: {rnn_mae}\n")
    f.write(f"LSTM - Activation: {activation_function}, Epochs: {epoch}, Test Loss (MSE): {lstm_loss}, Test MAE: {lstm_mae}\n")
    f.write("-" * 50 + "\n")