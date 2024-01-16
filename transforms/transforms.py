from torchvision import transforms, utils
import torch
from common.utils import get_config
import numpy as np

class Resize(object):
    def __init__(self, output_size = 256):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
            
    def __call__(self, image):
        image = transforms.functional.resize(image, self.output_size)
            
        return image
    
class RandomCrop(object):
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        
    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        top = torch.randint(0, h - new_h + 1)
        left = torch.randint(0, w - new_w + 1)
        image = image[top: top + new_h, left: left + new_w]
        
        return image

class CenterCrop(object):
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    
    def __call__(self, image):
        assert isinstance(image, torch.Tensor)
        image = transforms.functional.center_crop(image, self.output_size)

        return image
        
class ToTensor(object):
    
    def __call__(self, image):
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image) / 255
        return image
            
            