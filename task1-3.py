import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense
from sklearn.model_selection import train_test_split
from reshape import reshape_to_supervised
import matplotlib.pyplot as plt
from draw import draw_plots
from statistical_test import perform_statistical_tests

# Generate a sinusoidal wave dataset
def generate_sinusoidal_wave(period=20, sample_rate=100, num_periods=3):
    """
    Generate a sinusoidal wave dataset.

    Parameters:
        period (float): Period of the sinusoidal wave in seconds.
        sample_rate (int): Sampling rate in Hz.
        num_periods (int): Number of periods to generate.

    Returns:
        data (numpy array): Generated sinusoidal wave data.
    """
    t = np.linspace(0, num_periods * period, num_periods * sample_rate * period, endpoint=False)
    data = np.sin(2 * np.pi * t / period)
    return t, data

# Generate the sinusoidal wave data
t, data = generate_sinusoidal_wave(period=20, sample_rate=100, num_periods=4)

# Perform statistical tests on the generated data
draw_plots(data, lags=40, name="sinusoidal_wave")  # Draw plots for the generated data
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

# # Train and evaluate the LSTM model
# lstm_model = build_lstm_model(n_input, n_output, activation_function)
# lstm_model.fit(X_train, y_train, epochs=epoch, batch_size=16, verbose=1)
# lstm_loss, lstm_mae = lstm_model.evaluate(X_test, y_test, verbose=0)
# print(f"LSTM - Activation: {activation_function}, Epochs: {epoch}, Test Loss (MSE): {lstm_loss}, Test MAE: {lstm_mae}")

rnn_model.save("sine_wave_rnn_model.h5")
# Save results to a txt file
output_file = "1-3-output.txt"
with open(output_file, "a") as f:
    f.write(f"RNN - Activation: {activation_function}, Epochs: {epoch}, Test Loss (MSE): {rnn_loss}, Test MAE: {rnn_mae}\n")
    # f.write(f"LSTM - Activation: {activation_function}, Epochs: {epoch}, Test Loss (MSE): {lstm_loss}, Test MAE: {lstm_mae}\n")
    f.write("-" * 50 + "\n")

# # In-Sample Prediction
y_train_pred_rnn = rnn_model.predict(X_train)
# # y_train_pred_lstm = lstm_model.predict(X_train)

# # Out-of-Sample Prediction
y_test_pred_rnn = rnn_model.predict(X_test)
# # y_test_pred_lstm = lstm_model.predict(X_test)

# Plot In-Sample Prediction (RNN)
plt.figure(figsize=(12, 6))
plt.plot(y_train[:50, 0], label="True Value (Step 1)", color="blue")
plt.plot(y_train_pred_rnn[:50, 0], label="RNN Prediction (Step 1)", color="orange", linestyle="--")
plt.plot(y_train[:50, 1], label="True Value (Step 2)", color="green")
plt.plot(y_train_pred_rnn[:50, 1], label="RNN Prediction (Step 2)", color="red", linestyle="--")
plt.title("RNN In-Sample Prediction")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend()
plt.savefig("pics/prediction/sine_wave_in_sample.png")
# plt.show()
plt.close()

# Plot Out-of-Sample Prediction (RNN)
plt.figure(figsize=(12, 6))
plt.plot(y_test[:50, 0], label="True Value (Step 1)", color="blue")
plt.plot(y_test_pred_rnn[:50, 0], label="RNN Prediction (Step 1)", color="orange", linestyle="--")
plt.plot(y_test[:50, 1], label="True Value (Step 2)", color="green")
plt.plot(y_test_pred_rnn[:50, 1], label="RNN Prediction (Step 2)", color="red", linestyle="--")
plt.title("RNN Out-of-Sample Prediction")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend()
plt.savefig("pics/prediction/sine_wave_out_sample.png")
# plt.show()
plt.close()

# # Plot In-Sample Prediction (LSTM)
# plt.figure(figsize=(12, 6))
# plt.plot(y_train[:50, 0], label="True Value (Step 1)", color="blue")
# plt.plot(y_train_pred_lstm[:50, 0], label="LSTM Prediction (Step 1)", color="orange", linestyle="--")
# plt.plot(y_train[:50, 1], label="True Value (Step 2)", color="green")
# plt.plot(y_train_pred_lstm[:50, 1], label="LSTM Prediction (Step 2)", color="red", linestyle="--")
# plt.title("LSTM In-Sample Prediction")
# plt.xlabel("Sample Index")
# plt.ylabel("Value")
# plt.legend()
# plt.show()

# # Plot Out-of-Sample Prediction (LSTM)
# plt.figure(figsize=(12, 6))
# plt.plot(y_test[:50, 0], label="True Value (Step 1)", color="blue")
# plt.plot(y_test_pred_lstm[:50, 0], label="LSTM Prediction (Step 1)", color="orange", linestyle="--")
# plt.plot(y_test[:50, 1], label="True Value (Step 2)", color="green")
# plt.plot(y_test_pred_lstm[:50, 1], label="LSTM Prediction (Step 2)", color="red", linestyle="--")
# plt.title("LSTM Out-of-Sample Prediction")
# plt.xlabel("Sample Index")
# plt.ylabel("Value")
# plt.legend()
# plt.show()