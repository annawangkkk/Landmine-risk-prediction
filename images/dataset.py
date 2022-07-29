import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

class Landmine(Dataset):
    def __init__(self, root, test_name, split, map_type, transform=None):
        self.root = root
        self.transform = transform
        self.map_type = map_type
        self.image_root = f'/home/siqiz/social_good/data/{self.map_type}'
        self.split = split
        self.test_name = test_name # three splits
        if self.split == "train_labeled":
            self.df = pd.read_csv(self.root + f'/processed_dataset/{test_name}/train/train_labeled.csv',index_col=0)
        elif self.split == "train_unlabeled":
            self.df = pd.read_csv(self.root + f'/processed_dataset/{test_name}/train/train_unlabeled.csv',index_col=0)
        elif self.split == "test_labeled":
            self.df = pd.read_csv(self.root + f'/processed_dataset/{test_name}/test/test_labeled.csv',index_col=0)
        elif self.split == "test_unlabeled":
            self.df = pd.read_csv(self.root + f'/processed_dataset/{test_name}/test/test_unlabeled.csv',index_col=0)
        elif self.split == "all_labeled":
            self.df = pd.read_csv(self.root + f'/processed_dataset/{test_name}/all/all_labeled.csv',index_col=0)
        elif self.split == "all_unlabeled":
            self.df = pd.read_csv(self.root + f'/processed_dataset/{test_name}/all/all_unlabeled.csv',index_col=0)
        self.img_paths = np.array(self.image_root + '/' + self.df['LATITUD_Y'].astype(str) + '_' + self.df['LONGITUD_X'].astype(str) + '.jpg')
        self.targets = np.array(self.df['mines_outcome'])
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self,idx):
        img_name = self.img_paths[idx]
        img = Image.open(self.img_paths[idx]).crop((0,0,224,224)) # remove "Google"
        tar = self.targets[idx]
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(tar).long()