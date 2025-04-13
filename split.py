from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
        X (numpy array): Input features.
        y (numpy array): Output labels.
        test_size (float): Proportion of the dataset to include in the test split (default is 0.2).
        random_state (int): Random seed for reproducibility (default is 42).

    Returns:
        X_train (numpy array): Training input features.
        X_test (numpy array): Testing input features.
        y_train (numpy array): Training output labels.
        y_test (numpy array): Testing output labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Example usage
if __name__ == "__main__":
    import numpy as np

    # Example data
    X = np.random.rand(100, 4)  # 100 samples, 4 features
    y = np.random.rand(100, 1)  # 100 samples, 1 output
    print("Original data size:", X.shape, y.shape)
    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training set size:", X_train.shape, y_train.shape)
    print("Testing set size:", X_test.shape, y_test.shape)