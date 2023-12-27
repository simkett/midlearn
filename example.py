from load import ImageLoader
from neuralnet import NeuralNet
import os



# Load images
directory = os.path.dirname(__file__)
dirName = os.path.dirname(directory) + '/midlearn'

img_loader = ImageLoader()
img_loader.load_image(f'{dirName}/dog_image.jpeg', 200, 200)

# Create neural net
network = NeuralNet()
network.add_convolution_layer('sharpening')

# Feed data to neural net
network.load_data(img_loader.images_arr[0])

# Test stuff
network.run_network()