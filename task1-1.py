import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from reshape import reshape_to_supervised
from split import split_data
from model_structure import build_model_from_structure
from draw import draw_plots
from statistical_test import perform_statistical_tests
import matplotlib.pyplot as plt
# Generate the equal-difference series
data = np.linspace(0, 1, 200, endpoint=False)  # 200 points, step = 0.005

# Perform statistical tests on the generated data
draw_plots(data, lags=40, name="equal_difference_series")  # Draw plots for the generated data

perform_statistical_tests(data, lags=40)  # Perform statistical tests on the generated data

# Reshape the data for supervised learning
n_input = 4  # Input vector size
n_output = 1  # Output vector size
X, y = reshape_to_supervised(data, n_input, n_output)
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
# print("X_train:", X_train)
# print("X_test:", X_test)

# Design the MLP model
activation_function = "sigmoid"  # Activation function
model_structure = "64-32"  # Model structure
epoch = 100  # Number of epochs

# Build the model structure
model = build_model_from_structure(model_structure, activation_function, n_input, n_output)
# model = Sequential([
#     Dense(32, activation='relu', input_dim=n_input),  # Hidden layer with 64 neurons
#     Dense(32, activation='relu'),                    # Hidden layer with 32 neurons
#     # Dense(32, activation='relu'),                    # Hidden layer with 16 neurons
#     Dense(n_output)                                  # Output layer with 1 neuron
# ])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=epoch, batch_size=16, verbose=1)

# Evaluate the model on the training set (in-sample prediction)
train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
print(f"In-Sample Loss: {train_loss}, In-Sample MAE: {train_mae}")

# Evaluate the model on the testing set (out-of-sample prediction)
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Out-of-Sample Loss: {test_loss}, Out-of-Sample MAE: {test_mae}")

# Make predictions
# In-sample prediction
sample_in_sample = X_train[0].reshape(1, -1)  # Reshape a sample input from the training set
predicted_in_sample = model.predict(sample_in_sample)
print(f"In-Sample Prediction - Input: {sample_in_sample}, Predicted Output: {predicted_in_sample}")

# Out-of-sample prediction
sample_out_sample = X_test[0].reshape(1, -1)  # Reshape a sample input from the testing set
predicted_out_sample = model.predict(sample_out_sample)
print(f"Out-of-Sample Prediction - Input: {sample_out_sample}, Predicted Output: {predicted_out_sample}")

# Save results to a txt file in append mode
output_file = "1-1-output.txt"
with open(output_file, "a") as f:  # Use "a" mode to append to the file
    f.write(f"Activation Function: {activation_function}\n")
    f.write(f"Model Structure: {model_structure}\n")
    f.write(f"In-Sample Loss (MSE): {train_loss}\n")
    f.write(f"In-Sample MAE: {train_mae}\n")
    f.write(f"Out-of-Sample Loss (MSE): {test_loss}\n")
    f.write(f"Out-of-Sample MAE: {test_mae}\n")
    f.write(f"Epochs: {epoch}\n")
    f.write("-" * 50 + "\n")  # Add a separator for clarity

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
# Plot In-Sample Prediction

plt.figure(figsize=(12, 6))
plt.plot(y_train[:50], label="True Value", color="blue")
plt.plot(y_train_pred[:50], label="Predicted Value", color="orange", linestyle="--")
plt.title("In-Sample Prediction")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend()
plt.savefig("pics/prediction/equal_difference_in_sample.png")
# plt.show()
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(y_test[:50], label="True Value", color="blue")
plt.plot(y_test_pred[:50], label="Predicted Value", color="orange", linestyle="--")
plt.title("Out-of-Sample Prediction")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend()
plt.savefig("pics/prediction/equal_difference_out_sample.png")
# plt.show()
plt.close()

