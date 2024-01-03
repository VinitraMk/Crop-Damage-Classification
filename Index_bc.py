# %%
#mount drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
!ls

# %%
# move into project directory
repo_name = "Image-Colorization"
%cd /content/drive/MyDrive/Personal-Projects/$repo_name
!ls

# %%
# set up environment
!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install matplotlib numpy pandas pyyaml opencv-python

# %%
# this cell is for downloading data.
# as of yet data is not hosted and is available in the private data folder


# %%
# setup some imports
#custom imports
from transforms.transforms import ToTensor
from dataloading.datareader import DataReader
from dataloading.dataset import CustomDataset
from common.utils import get_exp_params, init_config, get_config

#py imports
import random
import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# %%
# read experiment parameters
exp_params = get_exp_params()
print('Experiment parameters\n')
print(exp_params)

# %%
# initialize directories and config data
init_config()
config = get_config()
# %%
#initialize randomness seed
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# %%
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

# %%



