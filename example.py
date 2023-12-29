from load import ImageLoader
from neuralnet import CNN
import os



# Load images
directory = os.path.dirname(__file__)
dirName = os.path.dirname(directory) + '/midlearn'

img_loader = ImageLoader()
img_loader.load_image(f'{dirName}/dog_image.jpeg', 50, 50)

# Create neural net
network = CNN(color_dimensions=3)
network.add_convolution_layer(n_filters=5, filter_size=3, filter_stride=1, activation_func_name='relu')
network.add_pooling_layer(pooling_option='max', pool_size=2)
network.add_pooling_layer(pooling_option='max', pool_size=2, flatten=True)
network.add_fully_connected_layer()
network.add_fully_connected_layer(size=5)

# Feed data to neural net
network.load_data(img_loader.images_arr[0])

# Test stuff
network.run_network()



'''
ToDos:
- Data scaling
- Backpropagation
- Hyperparameter optimization
'''