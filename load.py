import numpy as np
from PIL import Image



class ImageLoader:
    def __init__(self):
        self.images_arr = []   # Image stored as array
    
    def load_image(self, image_path: str, resize_width: int, resize_height: int) -> None:
        image = Image.open(image_path).convert('RGB')
        image = self.resize_image(image, resize_width, resize_height)   # Resized image without maintainig aspect ratio
        image_arr = np.array(image)
        image_arr = self.fix_channels(image_arr)
        self.images_arr.append(image_arr)
        
    def resize_image(self, image: Image.Image, width: int, height: int) -> Image.Image:
        # Resize to given resolution while maintaining aspect ratio
        image.thumbnail((width, height))
        # Add padding with zeros, if the image does not have enough pixels in one (or both) directions
        fixed_image = Image.new('RGB', (width, height), (0, 0, 0))
        fixed_image.paste(image, (0, 0))
        return fixed_image
    
    def fix_channels(self, input_image_arr) -> np.ndarray:
        # Every channel needs to represent one color
        channels = ['r', 'g', 'b']
        image_arr_rgb_channels = []
        for i_channel, channel in enumerate(channels):
            image_arr_rgb_channels.append(input_image_arr[:, :, i_channel])
        return np.array(image_arr_rgb_channels)
