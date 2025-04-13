from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_process import ArmaProcess
import os
from tensorflow.keras.optimizers import Adam 
from sklearn.model_selection import train_test_split
from reshape import reshape_to_supervised
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
    ar = np.array([1, -0.5, 0.25])  # AR coefficients
    ma = np.array([1, 0.4, -0.3])   # MA coefficients
    arma_process = ArmaProcess(ar, ma)
    return arma_process.generate_sample(nsample=n)

def plot_predictions(y_train, y_test, y_train_pred, y_test_pred, file_prefix):
    """
    Plot in-sample and out-of-sample predictions.

    Parameters:
        y_train (numpy array): True values for the training set.
        y_test (numpy array): True values for the test set.
        y_train_pred (numpy array): Predicted values for the training set.
        y_test_pred (numpy array): Predicted values for the test set.
        file_prefix (str): Prefix for saving the plot.
    """
    plt.figure(figsize=(12, 6))
    
    # 检查并创建目录
    output_dir = 'pics/prediction/'
    os.makedirs(output_dir, exist_ok=True)

    # Plot in-sample predictions
    plt.plot(range(len(y_train)), y_train, label='In-Sample True', color='blue')
    plt.plot(range(len(y_train)), y_train_pred, label='In-Sample Prediction', color='orange', linestyle='--')
    
    # Plot out-of-sample predictions
    plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, label='Out-of-Sample True', color='green')
    plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_test_pred, label='Out-of-Sample Prediction', color='red', linestyle='--')
    
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title('In-Sample and Out-of-Sample Predictions')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/{file_prefix}_predictions.png')  # Save the plot
    plt.close()  # Close the plot to free memory

def plot_loss_curve(history, structure, activation, file_prefix, epochs, data_name='data', data_amplitude=1):
    """
    绘制训练和验证损失随 Epoch 的变化曲线。

    Parameters:
        history: Keras 训练返回的 history 对象。
        structure: 模型结构（如 [32, 32]）。
        activation: 激活函数（如 'relu'）。
        file_prefix: 保存图像的文件前缀。
    """
    train_loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', None)
    # val_loss = history.history.get('val_loss', None)  # 如果没有验证集，val_loss 可能为 None
    train_loss = np.array(train_loss) / data_amplitude
    if val_loss is not None and len(val_loss) > 0:
        val_loss = np.array(val_loss) / data_amplitude
    

    # 检查并创建目录
    output_dir = 'pics/loss/'
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss', color='blue')
    if val_loss is not None and len(val_loss) > 0:
        plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title(f'MSE Convergence - Structure: {structure}, Activation: {activation}')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.title(f'MSE Convergence - Structure: {structure}, Activation: {activation}, Data: {data_name}')
    plt.savefig(f'{output_dir}{data_name}_{file_prefix}_loss_curve_structure_{structure}_activation_{activation}_epochs_{epochs}.png')
    plt.close()  # 关闭图像以释放内存

def build_rnn_model(n_input, n_output, activation, learning_rate=0.001):
    """
    Build an RNN model.

    Parameters:
        n_input (int): Number of input features (time steps).
        n_output (int): Number of output features.
        activation (str): Activation function for RNN layers.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        model: Compiled RNN model.
    """
    model = Sequential([
        Input(shape=(n_input, 1)),  # Input layer with time steps
        SimpleRNN(100, activation=activation, return_sequences=True),
        SimpleRNN(50, activation=activation),
        Dense(n_output)  # Output layer
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def train_rnn(data, n_input, n_output, file, activations, epochs_list, learning_rate=0.001, data_name='data'):
    """
    Train and evaluate an RNN model.

    Parameters:
        data: Time series data.
        n_input: Number of input features (time steps).
        n_output: Number of output features.
        file: File to save results.
        activations: List of activation functions.
        epochs_list: List of epochs to train.
        learning_rate: Learning rate for the optimizer.
        data_name: Name of the dataset (for saving results).
    """
    X, y = reshape_to_supervised(data, n_input, n_output)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape input data for RNN
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Calculate data amplitude for normalization
    data_amplitude = np.max(data) - np.min(data)
    for activation in activations:
        for epochs in epochs_list:
            print(f"Training RNN - Activation: {activation}, Epochs: {epochs}")
            model = build_rnn_model(n_input, n_output, activation, learning_rate=learning_rate)
            
            # Train the model
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0, validation_data=(X_test, y_test))
            
            # Plot loss curve
            plot_loss_curve(history, structure="RNN", activation=activation, file_prefix='rnn', epochs=epochs, data_name=data_name, data_amplitude=data_amplitude)

            # Evaluate the model
            loss, mae = model.evaluate(X_test, y_test, verbose=0)

            # Predict and calculate residuals
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            residuals = y_test.flatten() - y_test_pred.flatten()

            # Plot predictions
            plot_predictions(y_train, y_test, y_train_pred, y_test_pred, file_prefix=f'rnn_{data_name}_{activation}_{epochs}')

            # Ljung-Box test
            try:
                lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
                lb_pvalue = lb_test['lb_pvalue'].values[0]
                is_white_noise = lb_pvalue > 0.05
            except Exception as e:
                lb_pvalue = None
                is_white_noise = False
                print(f"Error in Ljung-Box test: {e}")

            # Save results to file
            with open(file, 'a') as f:
                f.write(f"RNN - Activation: {activation}, Epochs: {epochs}, Test Loss (MSE): {loss}, Test MAE: {mae}\n")
                if lb_pvalue is not None:
                    f.write(f"Ljung-Box p-value: {lb_pvalue:.4f}, Residuals White Noise: {is_white_noise}\n")
                else:
                    f.write("Ljung-Box test failed.\n")

if __name__ == "__main__":
    # Parameters
    learning_rate = 0.001
    activations = ['relu', 'tanh', 'sigmoid']
    # activations = ['relu']
    epochs_list = [50, 100]

    # Generate datasets
    white_noise = generate_white_noise()
    random_walk = generate_random_walk()
    arma_process = generate_arma_process()

    # Train and save results
    # train_rnn(white_noise, n_input=16, n_output=1, file='white_noise_results.txt', activations=activations, epochs_list=epochs_list, learning_rate=learning_rate, data_name='white_noise')
    # train_rnn(random_walk, n_input=16, n_output=1, file='random_walk_results.txt', activations=activations, epochs_list=epochs_list, learning_rate=learning_rate, data_name='random_walk')
    train_rnn(arma_process, n_input=16, n_output=1, file='arma_process_results.txt', activations=activations, epochs_list=epochs_list, learning_rate=learning_rate, data_name='arma_process')