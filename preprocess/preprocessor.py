import os
from common.utils import get_config
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_image

class Preprocessor:

    def __init__(self):
        cfg = get_config()
        self.image_dir = cfg["img_dir"]
        self.train_labels = pd.read_csv(os.path.join(cfg["data_dir"], "input/Train.csv"))
        self.test_labels = pd.read_csv(os.path.join(cfg["data_dir"], "input/Test.csv"))
        self.processed_img_dir = os.path.join(cfg["data_dir"], "processed_input")
        self.X_key = cfg['X_key']
        self.y_key = cfg['y_key']
        
        
    def transform_input(self, transform):
        train_files = self.train_labels["filename"].tolist()
        test_files = self.test_labels["filename"].tolist()
        train_data = np.array([])
        test_data = np.array([])
        print('Iterating through train files')
        for i, tf in enumerate(train_files):
            print(f'\tFile {i}')
            img = read_image(os.path.join(self.image_dir, tf))
            sample = { self.X_key: img }
            img = transform(sample)[self.X_key]
            np.append(train_data, img)
        print('\nIterating through test files')
        for i, tf in enumerate(test_files):
            print(f'\tFile {i}')
            img = read_image(os.path.join(self.image_dir, tf))
            sample = { self.X_key: img }
            img = transform(sample)[self.X_key]
            np.append(test_data, img)
        with open(os.path.join(self.processed_img_dir, "train.npy"), 'w') as fp:
            np.save(fp, train_data)
        with open(os.path.join(self.processed_img_dir, "test.npy"), 'w') as fp:
            np.save(fp, test_data)
        all_data = np.concatenate((train_data, test_data))
        with open(os.path.join(self.processed_img_dir, "data.npy"), 'w') as fp:
            np.save(fp, all_data)
            
    def make_label_csv(self):
        ## This function is for implementing code that constructs a csv file
        ## listing labels of all images. The csv file will have 4 columsn - image file name, label (encoded),
        ## original label and full path
        pass

