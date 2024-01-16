import os
from common.utils import get_config, get_exp_params, dump_json, image_collate
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_image
from random import shuffle
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose

class Preprocessor:

    def __init__(self):
        self.exp_params = get_exp_params()
        cfg = get_config()
        self.root_dir = cfg['root_dir']
        
    def __get_metric(self, dataloader, data_transform):
        pop_mean = []
        pop_std0 = []
        for i, data in enumerate(dataloader):
            # shape (batch_size, 3, height, width)
            print(f'\t\tGetting metrics for batch {i}')
            _, imgs , _ = data
            img_batch = list(map(data_transform, imgs))
            numpy_image = np.stack(img_batch, 0)
            #print(numpy_image.shape)
            # shape (3,)
            batch_mean = np.mean(numpy_image, axis=(0,2,3))
            batch_std0 = np.std(numpy_image, axis=(0,2,3))
            
            pop_mean.append(batch_mean)
            pop_std0.append(batch_std0)
            del data
        
        # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
        pop_mean = np.array(pop_mean).mean(axis=0)
        pop_mean = [x/255 for x in pop_mean]
        pop_std0 = np.array(pop_std0).mean(axis=0)
        pop_std0 = [x/255 for x in pop_std0]
        return { 'mean': pop_mean, 'std0': pop_std0 }

    def get_dataset_metrics(self, dataset, data_transform):
        data_transform = Compose(data_transform)
        if self.exp_params['train']['val_split_method'] == 'k-fold':
            k = self.exp_params['train']['k']
            fl = len(dataset)
            fr = list(range(fl))
            shuffle(fr)
            vlen = fl // k
            si = 0
            vset_ei = fl // k
            val_eei = list(range(vset_ei, fl, vlen))
            if val_eei[-1] <= fl and (val_eei[-1] + vlen) <= (fl + vlen):
                val_eei.append(val_eei[-1] + vlen)
            preop = Preprocessor()
            all_folds_metrics = {}
            
            for vi, ei in enumerate(val_eei):
                print(f"\tCalculating metric for split {vi} starting with {si}, ending with {ei}")
                val_idxs = fr[si:ei]
                tr_idxs = [fi for fi in fr if fi not in val_idxs]
                train_dataset = Subset(dataset, tr_idxs)
                train_loader = DataLoader(train_dataset,
                    batch_size = self.exp_params['train']['batch_size'],
                    shuffle = False,
                    collate_fn=image_collate
                )
                all_folds_metrics[vi] = self.__get_metric(train_loader, data_transform)
                si = ei
                jpath = os.path.join(self.root_dir, 'models/checkpoints/all_folds_metrics.json')
                dump_json(all_folds_metrics, jpath)
                print(f'\tSaved metrics of fold {vi}\n')
            print('\n\nSaved all metrics')
            return all_folds_metrics
        elif self.exp_params['train']['val_split_method'] == 'fixed-split':
            all_folds_metrics = {}
            train_loader = DataLoader(dataset,
                batch_size = self.exp_params['train']['batch_size'],
                shuffle = False,
                collate_fn=image_collate
            )
            jpath = os.path.join(self.root_dir, 'models/checkpoints/all_folds_metrics.json')
            all_folds_metrics[0] = self.__get_metric(train_loader, data_transform)
            dump_json(all_folds_metrics, jpath)
            print('\n\nSaved all metrics')
            return all_folds_metrics
        else:
            raise SystemExit('Error: Invalid val split method name passed! Check run.yaml')

    def make_label_csv(self):
        ## This function is for implementing code that constructs a csv file
        ## listing labels of all images. The csv file will have 4 columsn - image file name, label (encoded),
        ## original label and full path
        pass


