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
    
    def add_convolution_layer(self, filter_name: str) -> None:
            self.layers.append(ConvolutionLayer(filter_name))

    def run_network(self) -> None:
        input_data_layer = self.input_data   # Input data of first layer is input data of whole network
        for layer in self.layers:
            input_data_layer = layer.run(input_data_layer)
                   


class ConvolutionLayer:
    def __init__(self, filter_name: str):
        self.filter_name = filter_name
        self.filter = None
        self.set_filter()

    def set_filter(self) -> None:
        self.filter = eval(f'const.{self.filter_name}')

    def convolution(self, input_image) -> np.ndarray:
        convolution_layers_lst = []
        for image_dimension in input_image:  # Convolution for every color dimension (e.g. RGB)
            convolution_layers_lst.append(signal.convolve2d(image_dimension, self.filter))
        return np.array(convolution_layers_lst)

    def run(self, input_data) -> None:
        conv_image = self.convolution(input_data)
        img_pillow_arr = np.dstack((conv_image[0].astype(np.uint8), conv_image[1].astype(np.uint8), conv_image[2].astype(np.uint8)))
        img_pillow = Image.fromarray(img_pillow_arr)
        img_pillow.show()