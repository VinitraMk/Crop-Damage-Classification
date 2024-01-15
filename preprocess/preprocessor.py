import os
from common.utils import get_config, get_exp_params, dump_json
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_image
from random import shuffle
from torch.utils.data import DataLoader, Subset

class Preprocessor:

    def __init__(self):
        cfg = get_config()
        self.image_dir = cfg["img_dir"]
        '''
        self.train_labels = pd.read_csv(os.path.join(cfg["data_dir"], "input/Train.csv"))
        self.test_labels = pd.read_csv(os.path.join(cfg["data_dir"], "input/Test.csv"))
        self.processed_img_dir = os.path.join(cfg["data_dir"], "processed_input")
        '''
        self.X_key = cfg['X_key']
        self.y_key = cfg['y_key']
        self.root_dir = cfg["root_dir"]
        self.exp_params = get_exp_params()
        
    def __get_metric(self, dataloader):
        pop_mean = []
        pop_std0 = []
        pop_std1 = []
        for i, data in enumerate(dataloader):
            # shape (batch_size, 3, height, width)
            numpy_image = data[self.X_key].numpy()
            #print(numpy_image.shape)
            # shape (3,)
            batch_mean = np.mean(numpy_image, axis=(0,2,3))
            batch_std0 = np.std(numpy_image, axis=(0,2,3))
            batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
            
            pop_mean.append(batch_mean)
            pop_std0.append(batch_std0)
            pop_std1.append(batch_std1)
            del data[self.X_key]
        
        # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
        pop_mean = np.array(pop_mean).mean(axis=0)
        pop_mean = [x/255 for x in pop_mean]
        pop_std0 = np.array(pop_std0).mean(axis=0)
        pop_std0 = [x/255 for x in pop_std0]
        pop_std1 = np.array(pop_std1).mean(axis=0)
        pop_std1 = [x/255 for x in pop_std1]
        return pop_mean, pop_std0, pop_std1

    def get_dataset_metrics(self, dataset, method = 'k-fold'):
        if method == 'k-fold':
            k = self.exp_params['train']['k']
            fl = len(dataset)
            fr = list(range(fl))
            shuffle(fr)
            vlen = fl // k
            si = 0
            vset_ei = fl // k
            val_eei = list(range(vset_ei, fl, vlen))
            preop = Preprocessor()
            all_folds_metrics = {}
            
            for vi, ei in enumerate(val_eei):
                print(f"\tCalculating metric for split {vi} starting with {si}, ending with {ei}")
                val_idxs = fr[si:ei]
                tr_idxs = [fi for fi in fr if fi not in val_idxs]
                train_dataset = Subset(dataset, tr_idxs)
                train_loader = DataLoader(train_dataset,
                    batch_size = 128,
                    shuffle = False,
                    num_workers=1
                )
                all_folds_metrics[vi] = self.__get_metric(train_loader)
                si = ei
                jpath = os.path.join(self.root_dir, 'models/checkpoints/all_folds_metrics.json')
                dump_json(all_folds_metrics, jpath)
            return all_folds_metrics
        else:
            train_loader = DataLoader(dataset,
                batch_size = 128,
                shuffle = False,
                num_workers=1,
                persistent_workers=True
            )
            jpath = os.path.join(self.root_dir, 'models/checkpoints/all_folds_metrics.json')
            dump_json(all_folds_metrics, jpath)
            return self.__get_metric(train_loader)

    def make_label_csv(self):
        ## This function is for implementing code that constructs a csv file
        ## listing labels of all images. The csv file will have 4 columsn - image file name, label (encoded),
        ## original label and full path
        pass

