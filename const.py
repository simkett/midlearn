import numpy as np



# Convolution filters
identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])          # Retains information without change
sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])          # Edge detection
prewitt = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])        # Edge detection
gaussian = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])          # Blur
box = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])               # Blur
sharpening = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])    # Sharpening
embossing = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])      # Sharpening