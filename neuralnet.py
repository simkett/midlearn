import numpy as np
from scipy import signal
from PIL import Image
from skimage.measure import block_reduce
import const



class CNN:
    def __init__(self, color_dimensions=None):
        self.color_dimensions = color_dimensions
        self.input_data = None
        self.layers = []

    def load_data(self, data: np.ndarray) -> None:
        self.input_data = data
    
    def add_convolution_layer(self, n_filters: int, filter_size: int, filter_stride: int, activation_func_name: str) -> None:
        self.layers.append(ConvolutionLayer(n_filters, filter_size, filter_stride,  activation_func_name, self.color_dimensions))

    def add_pooling_layer(self, pooling_option: str, pool_size: int=2, flatten: bool=False) -> None:
        self.layers.append(PoolingLayer(pooling_option, pool_size, flatten))

    def add_fully_connected_layer(self, size: int=None) -> None:
        self.layers.append(FullyConnectedLayer(size))

    def run_network(self) -> None:
        data_layer = self.input_data   # Input data of first layer is input data of whole network
        for layer in self.layers:
            data_layer = layer.run(data_layer)
            print(f'   {type(layer)} --> {data_layer.shape}')



class ConvolutionLayer:
    def __init__(self, n_filters: int, filter_size: int, filter_stride: int, activation_func_name: str, num_color_dimensions: int):
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.filter_stride = filter_stride
        self.activation_func_name = activation_func_name
        self.filters = []  # list of np.ndarray
        self.bias = []
        self.color_dimensions = num_color_dimensions
        self.initialize_filter_bias()
        self.output = None

    def initialize_filter_bias(self) -> None:
        for i in range(self.n_filters):
            self.filters.append(np.random.randn(self.filter_size, self.filter_size))
            self.bias.append(np.random.randn())

    def convolution(self, image_arr: np.ndarray, filter_no: int) -> np.ndarray:
        convolution_layers_lst = []
        for image_dimension in image_arr:  # Convolution for every color dimension (e.g. RGB)
            image_dimension = image_dimension[::self.filter_stride, ::self.filter_stride]
            convolution_layers_lst.append(signal.convolve2d(image_dimension, self.filters[filter_no], mode='same')+self.bias[filter_no]) # Here: Same filter for all channels
        sum_conv_layers = np.sum(convolution_layers_lst, axis=0)
        return sum_conv_layers
    
    def apply_activation_func(self, image_arr: np.ndarray) -> np.ndarray:
        image_arr = const.activation_func(image_arr, self.activation_func_name)
        return image_arr

    def run(self, input_data: np.ndarray) -> np.ndarray:
        output = []
        for i in range(self.n_filters):
            conv_image = self.convolution(input_data, i)
            conv_image = self.apply_activation_func(conv_image)
            output.append(conv_image)
        output = np.array(output)
        self.output = output
        return output



class PoolingLayer:
    def __init__(self, pooling_option: str, pool_size: tuple, flatten_array: bool):
        self.pooling_option = pooling_option
        self.pool_size = pool_size
        self.flatten_array = flatten_array
        self.output = None

    def pooling(self, image_arr: np.ndarray) -> np.ndarray:
        if self.pooling_option == 'max':
            pooled_img = block_reduce(image_arr, block_size=self.pool_size, func=np.max)
        elif self.pooling_option == 'avg':
            pooled_img = block_reduce(image_arr, block_size=self.pool_size, func=np.mean)
        return pooled_img
    
    def flatten_func(self, input_data: np.ndarray) -> np.ndarray:
        return input_data.flatten()

    def run(self, input_data: np.ndarray) -> np.ndarray:
        output = []
        for layer in input_data:
            pooled_layer = self.pooling(layer)
            output.append(pooled_layer)
        output = np.array(output)
        if self.flatten_array == True:
            output = self.flatten_func(output)
            len_output = len(output)
            output = np.reshape(output, (1, len_output))
        self.output = output
        return output
    


class FullyConnectedLayer:
    def __init__(self, size: int):
        self.size = size
        self.weights = None # weight matrix (2D)
        self.bias = None    # bias vector (1D)
        self.output = None  # neuron vector (1D)
    
    def initialize_weights_bias(self, input_data_shape: tuple) -> None:
        n_neurons_input = input_data_shape[1]   # Number of neurons of previous layer
        if self.size:
            n_neurons_output = self.size
        else:
            n_neurons_output = n_neurons_input      # Number of neurons of this layer
        self.weights = np.random.randn(n_neurons_input, n_neurons_output)

        self.bias = np.random.randn(1, n_neurons_output)

    def run(self, input_data: np.ndarray) -> np.ndarray:
        if not self.weights:
            self.initialize_weights_bias(input_data.shape)
        self.output = np.dot(input_data, self.weights) + self.bias
        print(self.output)
        return self.output
