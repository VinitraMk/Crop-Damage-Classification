#custom imports
import config
from transforms.transforms import ToTensor
from dataloading.datareader import DataReader
from dataloading.dataset import CustomDataset
from common.utils import get_exp_params

#py imports
import random
import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

if __name__ == "__main__":
    
    # read experiment parameters
    exp_params = get_exp_params()
       
    # initialize directories and config data
    config.root_dir = os.getcwd()
    config.use_gpu = torch.cuda.is_available()
    config.data_dir = os.path.join(os.getcwd(), 'data')

    #initialize randomness seed
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    #preprocess data or load preprocessed data
    dr = DataReader()
    ds = dr.get_split_data()
    Ltr, ABtr, ftr_len = ds['Ltr'], ds['ABtr'], ds['ftr_len']
    Lte, ABte, te_len = ds['Lte'], ds['ABte'], ds['te_len']

    #transform data
    composed_transforms =  transforms.Compose([
        ToTensor(True)
    ])
    
    #convert to dataset
    ftr_dataset = CustomDataset(Ltr, ABtr, ftr_len, composed_transforms)
    te_dataset = CustomDataset(Lte, ABte, te_len, composed_transforms)
    
    #load data
    ftr_loader = DataLoader(ftr_dataset, batch_size = exp_params['data_params']['batch_size'])
    te_loader = DataLoader(te_dataset, batch_size = exp_params['data_params']['batch_size'])
    
    
    