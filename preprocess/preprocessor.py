import os
from common.utils import get_config
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_image

class Preprocessor:

    def __init__(self, data_filesuffix):
        cfg = get_config()
        self.image_dir = cfg["img_dir"]
        self.train_labels = pd.read_csv(os.path.join(cfg["data_dir"], "input/Train.csv"))
        self.test_labels = pd.read_csv(os.path.join(cfg["data_dir"], "input/Test.csv"))
        self.processed_img_dir = os.path.join(cfg["data_dir"], "processed_input")
        self.X_key = cfg['X_key']
        self.y_key = cfg['y_key']
        self.data_filesuffix = data_filesuffix
        
        
    def transform_input(self, transform):
        train_files = self.train_labels["filename"].tolist()
        test_files = self.test_labels["filename"].tolist()
        train_data = np.empty((1, self.data_filesuffix, self.data_filesuffix, 3))
        test_data = np.empty((1, self.data_filesuffix, self.data_filesuffix, 3))
        print('Iterating through train files')
        for i, tf in enumerate(train_files):
            img = read_image(os.path.join(self.image_dir, tf))
            sample = { self.X_key: img }
            img = transform(sample)[self.X_key]
            img = img.cpu().detach().numpy()
            img = np.transpose(img, (1, 2, 0))
            train_data = np.concatenate((train_data, np.expand_dims(img, 0)))
        train_data = train_data[1:]
        np.savez_compressed(os.path.join(self.processed_img_dir, f"train_{self.data_filesuffix}"),  train_data)
        del train_data
        print('\nIterating through test files')
        for i, tf in enumerate(test_files):
            img = read_image(os.path.join(self.image_dir, tf))
            sample = { self.X_key: img }
            img = transform(sample)[self.X_key]
            img = img.cpu().detach().numpy()
            img = np.transpose(img, (1, 2, 0))
            test_data = np.concatenate((test_data, np.expand_dims(img, 0)))
        test_data = test_data[1:]
        np.savez_compressed(os.path.join(self.processed_img_dir, f"test_{self.data_filesuffix}"), test_data)
        del test_data
    def make_label_csv(self):
        ## This function is for implementing code that constructs a csv file
        ## listing labels of all images. The csv file will have 4 columsn - image file name, label (encoded),
        ## original label and full path
        pass

