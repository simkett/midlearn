import numpy as np
from PIL import Image
import os



class ImageLoader:
    def __init__(self):
        self.images_arr = []   # Image stored as array
        self.labels = []       # Labels stored as array of strings
        self.feature_vector = None    # One hot encoded feature vector
        self.classes = None
    
    def load_data(self, dir_path: str, resize_width: int, resize_height: int) -> None:
        for file in os.listdir(f'{dir_path}'):
            self.load_image(f'{dir_path}/{file}', resize_width, resize_height) # Images have to be names like this: 'label_no' -> example: 'dog_01'
        self.scale_image_data()     # Scale image data
        self.create_feature_vector()    # Create one hot encoded feature vector
    
    def load_image(self, image_path: str, resize_width: int, resize_height: int) -> None:
        image = Image.open(image_path).convert('RGB')
        image = self.resize_image(image, resize_width, resize_height)   # Resized image without maintainig aspect ratio
        image_arr = np.array(image)
        image_arr = self.fix_channels(image_arr)
        self.images_arr.append(image_arr)   # Store image data in array
        self.labels.append(os.path.basename(image_path.split('.')[0].split('_')[0]).lower())    # Store label
        
    def resize_image(self, image: Image.Image, width: int, height: int) -> Image.Image:
        # Resize to given resolution while maintaining aspect ratio
        image.thumbnail((width, height))
        # Add padding with zeros, if the image does not have enough pixels in one (or both) directions
        fixed_image = Image.new('RGB', (width, height), (0, 0, 0))
        fixed_image.paste(image, (0, 0))
        return fixed_image
    
    def fix_channels(self, input_image_arr: np.array) -> np.ndarray:
        # Every channel needs to represent one color
        channels = ['r', 'g', 'b']
        image_arr_rgb_channels = []
        for i_channel, channel in enumerate(channels):
            image_arr_rgb_channels.append(input_image_arr[:, :, i_channel])
        return np.array(image_arr_rgb_channels)
    
    def scale_image_data(self) -> None:
        # Standardization is used
        for index_image, image in enumerate(self.images_arr):
            mean = np.mean(image, axis=(0, 1), keepdims=True)
            std = np.std(image, axis=(0, 1), keepdims=True)
            image_scaled = (image - mean) / std
            self.images_arr[index_image] = image_scaled
    
    def create_feature_vector(self) -> None:
        self.classes = list(set(self.labels))
        feature_vector = np.zeros((len(self.labels), len(self.classes)))
        for index_image, label in enumerate(self.labels):
            index_class = self.classes.index(label)
            feature_vector[index_image][index_class] = 1
