import numpy as np

arr = np.array(([0, 0, 1, 1]))
bias = np.array([[200, 200, 200, 200]])
arr2 = arr + bias
print(arr2)