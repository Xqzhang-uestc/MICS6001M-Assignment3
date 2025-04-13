import numpy as np

def reshape_to_supervised(data, n_input, n_output):
    """
    Reshape a time series dataset into a supervised learning dataset.

    Parameters:
        data (list or numpy array): The time series data.
        n_input (int): Number of input time steps.
        n_output (int): Number of output time steps.

    Returns:
        X (numpy array): Input features of shape (samples, n_input).
        y (numpy array): Output labels of shape (samples, n_output).
    """
    data = np.array(data)
    X, y = [], []
    for i in range(len(data) - n_input - n_output + 1):
        X.append(data[i:i + n_input])
        y.append(data[i + n_input:i + n_input + n_output])
    return np.array(X), np.array(y)

# Example usage
if __name__ == "__main__":
    # Example time series data
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    n_input = 4
    n_output = 1

    X, y = reshape_to_supervised(data, n_input, n_output)
    print("Input (X):")
    print(X)
    print("Output (y):")
    print(y)