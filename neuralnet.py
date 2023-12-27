import numpy as np
from scipy import signal
from PIL import Image
import const



class NeuralNet:
    def __init__(self):
        self.input_data = None
        self.layers = []

    def load_data(self, data: np.ndarray) -> None:
        self.input_data = data
    
    def add_convolution_layer(self, filter_name: str, activation_func_name: str) -> None:
            self.layers.append(ConvolutionLayer(filter_name, activation_func_name))

    def run_network(self) -> None:
        input_data_layer = self.input_data   # Input data of first layer is input data of whole network
        for layer in self.layers:
            input_data_layer = layer.run(input_data_layer)
                   


class ConvolutionLayer:
    def __init__(self, filter_name: str, activation_func_name: str):
        self.filter_name = filter_name
        self.activation_func_name = activation_func_name
        self.filter = None  # np.ndarray
        self.set_filter()
        self.output = None

    def set_filter(self) -> None:
        self.filter = const.conv_filters[self.filter_name]

    def convolution(self, image_arr) -> np.ndarray:
        convolution_layers_lst = []
        for image_dimension in image_arr:  # Convolution for every color dimension (e.g. RGB)
            convolution_layers_lst.append(signal.convolve2d(image_dimension, self.filter, mode='same'))
        return np.array(convolution_layers_lst)
    
    def apply_activation_func(self, image_arr) -> np.ndarray:
        if self.activation_func_name == 'relu':
            image_arr = const.relu(image_arr)
        if self.activation_func_name == 'sigmoid':
            image_arr = const.sigmoid(image_arr)
        if self.activation_func_name == 'tanh':
            image_arr = const.tanh(image_arr)
        return image_arr

    def run(self, input_data) -> None:
        conv_image = self.convolution(input_data)
        conv_image = self.apply_activation_func(conv_image)
        self.output = conv_image
        self.visualize(self.output)
        
    def visualize(self, input_data) -> None:
        img_pillow_arr = np.dstack((input_data[0].astype(np.uint8), input_data[1].astype(np.uint8), input_data[2].astype(np.uint8)))
        img_pillow = Image.fromarray(img_pillow_arr)
        img_pillow.show()