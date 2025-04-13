from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# def build_model_from_structure(model_structure, activation_function, n_input, n_output):
#     """
#     Build a Sequential model based on the given model structure.

#     Parameters:
#         model_structure (str): A string representing the model structure, e.g., "64-32-16".
#         activation_function (str): The activation function to use in each layer, e.g., "relu".
#         n_input (int): The size of the input vector.
#         n_output (int): The size of the output vector.

#     Returns:
#         model (Sequential): A Keras Sequential model with the specified structure.
#     """
#     # Parse the model structure string into a list of integers
#     layer_sizes = list(map(int, model_structure.split('-')))
    
#     # Initialize the Sequential model
#     model = Sequential()
    
#     # Add the first layer with input_dim
#     model.add(Dense(layer_sizes[0], activation=activation_function, input_dim=n_input))
    
#     # Add the remaining hidden layers
#     for size in layer_sizes[1:]:
#         model.add(Dense(size, activation=activation_function))
    
#     # Add the output layer
#     model.add(Dense(n_output))
    
#     return model

def build_model_from_structure(model_structure, activation_function, n_input, n_output, learning_rate=0.001):
    """
    Build an MLP model based on the given structure and activation function.
    """
    """
    Build an MLP model based on the given structure and activation function.
    """
    model = Sequential()
    layers = list(map(int, model_structure.split('-')))
    model.add(Input(shape=(n_input,)))  # Use Input layer
    for units in layers:
        model.add(Dense(units, activation=activation_function))
    model.add(Dense(n_output))  # Output layer
    optimizer = Adam(learning_rate=learning_rate)  # Example learning rate
    model.compile(optimizer="adam", loss='mse', metrics=['mae'])
    return model

# Example usage
if __name__ == "__main__":
    model_structure = "64-32-32-32-16"
    activation_function = "relu"
    n_input = 4
    n_output = 1   
    learning_rate = 0.001  # 例如，设置为 0.001
    # optimizer = Adam(learning_rate=learning_rate)
    model = build_model_from_structure(model_structure, activation_function, n_input, n_output, learning_rate=learning_rate)
    model.summary()  # Print the model summary