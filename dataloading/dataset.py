#Python file to define custom dataset for your project

from torch.utils.data import Dataset
import pandas as pd
import os
from skimage import io, transform


class CustomDataset(Dataset):

    def __init__(self, label_csv_filepath, root_dir, name2numlblmap, transforms = None):
        self.image_labels = pd.read_csv(os.path.join(root_dir, label_csv_filepath))
        self.root_dir = root_dir
        self.transforms = transforms
        self.name2numlblmap = name2numlblmap

    def __len__(self):
        return len(self.image_labels)

    def __get_item__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_tensor_path = os.path.join(self.root_dir, self.image_labels[idx, 0])
        img_tensor = io.imread(img_tensor_path)
        label = name2numlblmap[self.image_labels.loc[idx,1]]
        sample = { 'image': img_tensor, 'label': label }

        if self.transforms:
            samle = self.transforms(sample)

        return sample

