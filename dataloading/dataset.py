#Python file to define custom dataset for your project

from torch.utils.data import Dataset
import pandas as pd
import os
from skimage import io, transform
from common.utils import get_config
import torch
from PIL import Image
import numpy as np

class CropDataset(Dataset):

    def __init__(self, label_csv_path, name2numlblmap, is_test = False):
        config = get_config()
        self.data_dir = config['data_dir']
        self.img_dir = config['img_dir']
        self.image_labels = pd.read_csv(os.path.join(self.data_dir, label_csv_path))
        self.name2numlblmap = name2numlblmap
        self.is_test = is_test

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_tensor_path = os.path.join(self.img_dir, self.image_labels.loc[idx, 'filename'])
        #img = io.imread(img_tensor_path)
        img = np.array(Image.open(img_tensor_path))
        label = self.name2numlblmap[self.image_labels.loc[idx, 'damage']] if not(self.is_test) else ''
        idint = self.image_labels.loc[idx, 'index']

        return idint, img, label 

