from torchvision import transforms, utils
import torch
from common.utils import get_config

class Resize(object):
    def __init__(self, output_size = 224, transform_y = False):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.transform_y = transform_y
        
    def __call__(self, sample):
        config = get_config()
        X_key, y_key = config['X_key'], config['y_key']
        image = sample[X_key]
        
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
              
        new_h, new_w = int(new_h), int(new_w)
        image = transforms.resize(image, (new_h, new_w))
        sample[X_key] = image
        if self.transform_y:
            image = sample[y_key]
            image = transforms.resize(image, (new_h, new_w))
            sample[y_key] = image
             
        return sample
    
class RandomCrop(object):
    
    def __init__(self, output_size, transform_y = False):
        assert isinstance(output_size, (int, tuple))
        
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.transform_y = transform_y
        
    def __call__(self, sample):
        config = get_config()
        X_key, y_key = config['X_key'], config['y_key']
        image = sample[X_key]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        top = torch.randint(0, h - new_h + 1)
        left = torch.randint(0, w - new_w + 1)
        image = image[top: top + new_h, left: left + new_w]
        sample[X_key] = image
        if self.transform_y:
            image = sample[y_key]
            image = image[top: top + new_h, left: left + new_w]
            sample[y_key] = image

        return sample

class CenterCrop(object):
    
    def __init__(self, output_size, transform_y = False):
        assert isinstance(output_size, (int, tuple))
        
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.transform_y = transform_y
    
    def __call__(self, sample):
        config = get_config()
        X_key, y_key = config['X_key'], config['y_key']
        image = sample[X_key]
        assert isinstance(image, torch.tensor)
        image = transforms.functional.center_crop(image)
        sample[X_key] = image
        if self.transform_y:
            image = sample[y_key]
            assert isinstance(image, torch.tensor)
            image = transforms.functional.center_crop(image)
            sample[y_key] = image

        return sample
        
class ToTensor(object):
    
    def __init__(self, transform_y = False):
        self.transform_y = transform_y
    
    def __call__(self, sample):
        config = get_config()
        X_key, y_key = config['X_key'], config['y_key']
        image = sample[X_key]
        image = image.transpose(2, 0, 1)
        sample[X_key] = image
        if self.transform_y:
            image = sample[y_key]
            image = image.transpose(2, 0, 1)
            sample[y_key] = image
        return sample
            
            