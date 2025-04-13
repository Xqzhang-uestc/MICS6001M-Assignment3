import numpy as np
from reshape import reshape_to_supervised
from sklearn.model_selection import train_test_split
from model_structure import build_model_from_structure
from tensorflow.keras.optimizers import Adam 
from statsmodels.stats.diagnostic import acorr_ljungbox
# from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
import os
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

def train_mlp(data, n_input, n_output, file, structures, activations, epochs_list, learning_rate=0.001, data_name='data'):
    X, y = reshape_to_supervised(data, n_input, n_output)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 计算数据集的幅值
    data_amplitude = np.max(data) - np.min(data)

    for structure in structures:
        for activation in activations:
            for epochs in epochs_list:
                model = build_model_from_structure('-'.join(map(str, structure)), activation, n_input, n_output, learning_rate=learning_rate)
                # model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0)
                
                # 训练模型并记录历史
                history = model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0, validation_data=(X_test, y_test))
                
                # 绘制损失曲线
                plot_loss_curve(history, structure, activation, file_prefix='mlp', epochs=epochs, data_name=data_name, data_amplitude=data_amplitude)

                loss, mae = model.evaluate(X_test, y_test, verbose=0)

                # 预测训练集和测试集
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # 绘制预测结果
                plot_predictions(y_train, y_test, y_train_pred, y_test_pred, file_prefix=f'mlp_{data_name}_{structure}_{activation}_{epochs}')

                # 计算残差
                residuals = y_test.flatten() - y_test_pred.flatten()

                # Normalize MSE and MAE
                normalized_loss = loss / data_amplitude
                normalized_mae = mae / data_amplitude

                # Ljung-Box 检验
                try:
                    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
                    lb_pvalue = lb_test['lb_pvalue'].values[0]
                    is_white_noise = lb_pvalue > 0.05  # 判断残差是否为白噪声
                except Exception as e:
                    lb_pvalue = None
                    is_white_noise = False
                    print(f"Error in Ljung-Box test: {e}")
                # Save results to file
                with open(file, 'a') as f:
                    f.write(f"MLP - Structure: {structure}, Activation: {activation}, Epochs: {epochs}, Test Loss (MSE): {loss}, Test MAE: {mae}\n")
                    # f.write(f"MLP - Structure: {structure}, Activation: {activation}, Epochs: {epochs}, Normalized Loss (MSE): {normalized_loss}, Normalized MAE: {normalized_mae}\n")
                    if lb_pvalue is not None:
                        f.write(f"Ljung-Box p-value: {lb_pvalue:.4f}, Residuals White Noise: {is_white_noise}\n")
                    else:
                        f.write("Ljung-Box test failed.\n")
                    # f.write(f"Ljung-Box p-value: {lb_pvalue:.4f}, Residuals White Noise: {is_white_noise}\n")


if __name__ == "__main__":
    # from task2 import generate_white_noise, generate_random_walk, generate_arma_process

    # Parameters
    learning_rate = 0.001
    # mlp_structures = [[32, 32], [64, 32], [64, 32, 16], [32, 32, 16]]
    # mlp_structures = [[32, 32], [16, 32], [8, 16]]
    mlp_structures = [[32, 32, 16]]
    activations = ['relu']
    # activations = ['relu']
    epochs_list = [100]
    # epochs_list = [100]
    # Generate datasets
    white_noise = generate_white_noise()
    random_walk = generate_random_walk()
    arma_process = generate_arma_process()

    # calculate the amplitude of data
    # print("Amplitude of Arma process: ", np.max(arma_process) - np.min(arma_process))   
    # print("Amplitude of White Noise: ", np.max(white_noise) - np.min(white_noise))
    # Calculate the amplitude of data
    # print("Amplitude of White Noise: ", np.max(random_walk) - np.min(random_walk))
    # Train and save results
    # train_mlp(white_noise, n_input=16, n_output=1, file='white_noise_results.txt', structures=mlp_structures, activations=activations, epochs_list=epochs_list, learning_rate=learning_rate, data_name='white_noise')   
    train_mlp(random_walk, n_input=16, n_output=1, file='random_walk_results.txt', structures=mlp_structures, activations=activations, epochs_list=epochs_list, learning_rate=learning_rate, data_name='random_walk')
    # train_mlp(arma_process, n_input=16, n_output=1, file='arma_process_results.txt', structures=mlp_structures, activations=activations, epochs_list=epochs_list, learning_rate=learning_rate, data_name='arma_process') 