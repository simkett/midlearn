import numpy as np



# Convolution filters
conv_filters = {
'identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),        # Retains information without change
'sobel': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),        # Edge detection
'prewitt': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),      # Edge detection
'gaussian': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]),        # Blur
'box': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),             # Blur
'sharpening': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),  # Sharpening
'embossing': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])     # Sharpening
}

# Activation functions
def activation_func(x: int, func_name: str) -> int:
    if func_name == 'relu':
        return np.maximum(0, x)
    if func_name == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    if func_name == 'tanh':
        return np.tanh(x)
    if func_name == 'softmax':
        e_x = np.exp(x - np.max(x))  # Subtracting the max value for numerical stability
        return e_x / np.sum(e_x, axis=0)
