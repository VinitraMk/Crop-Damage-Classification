from argparse import ArgumentParser
from torchvision.transforms import Compose
from transforms.transforms import Resize, CenterCrop, ToTensor
from preprocess.preprocessor import Preprocessor
from common.utils import init_config, get_config, save2config
import numpy as np
import torch
import os
from PIL import Image
import time
from dataloading.dataset import CropDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--resize_dim", type = int, required = True)
    parser.add_argument("--crop_dim", type = int, required = True)
    args = parser.parse_args()
    
    init_config()
    cfg = get_config()
    save2config('X_key', 'image')
    save2config('y_key', 'label')
    print(cfg)
    label_dict = {
        'DR': 0,
        'G': 1,
        'ND': 2,
        'WD': 3,
        'other': 4
    }
    
    data_transform = Compose([ToTensor(), Resize(args.resize_dim), CenterCrop(args.crop_dim)])
    ftr_dataset = CropDataset('input/Train.csv', label_dict, False)
    loader = DataLoader(ftr_dataset, batch_size = 1, shuffle = False)
    '''
    preop = Preprocessor(args.crop_dim)
    preop.transform_input(data_transform)
    
    a = np.load("data/processed_input/train_224.npz")
    train_224 = a['arr_0']
    test_224 = a['arr_0']
    print(train_224.shape, test_224.shape)
    '''
    img_files = os.listdir(cfg['img_dir'])[:10]
    sf = time.time()
    for imgfn in img_files:
        si = time.time()
        img = Image.open(os.path.join(cfg['img_dir'], imgfn))
        print('time taken to open image', (time.time() - si))
    print('time taken to load 10 images', (time.time() - sf))

    sf = time.time()
    print()
    for i, batch in enumerate(loader):
        si = time.time()
        #print(f'Processing batch {i}')
        print('Time taken to load 1 images', (time.time() - sf))
        if i == 9:
            break
    #print('time taken to load 2 batches', (time.time() - sf))
