import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from wb_data_copy import WaterBirdsDataset


class WaterBirdsDatasetCustom:

    def __init__(self, basedir, batch_size, label, test_transform=None, train_transform=None):
        self.test_transform = test_transform
        self.train_transform = train_transform
        self.label = label
        self.train = _WaterBirdsDatasetSlice(basedir, batch_size, split="train", transform=self.train_transform)
        self.test = _WaterBirdsDatasetSlice(basedir, batch_size, split="test", transform=self.test_transform)
        self.val = _WaterBirdsDatasetSlice(basedir, batch_size, split="val", transform=self.test_transform)


class _WaterBirdsDatasetSlice(Dataset):

    def __init__(self, basedir, batch_size, split="train", transform=None):
        # init
        self.basedir = basedir
        self.transform = transform
        self.batch_size = batch_size

        # metadata
        metadata_df = pd.read_csv(os.path.join(basedir, "metadata.csv"))
        split_i = ["train", "val", "test"].index(split)
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]

        # targets
        self.y_array = self.metadata_df['y'].values
        self.n_classes = np.unique(self.y_array).size

        # places
        self.p_array = self.metadata_df['place'].values
        self.n_places = np.unique(self.p_array).size

        # groups (waterbird+land, waterbird+water, landbird+land, landbird+water
        self.group_array = (self.y_array * self.n_places + self.p_array).astype('int')
        self.n_groups = self.n_classes * self.n_places

        # num of items in each group, class, place
        self.group_counts = (
                torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()

        # list of filenames
        self.filename_array = self.metadata_df['img_filename'].values

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]

        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, y, g

    def filter(self, ids):
        # retain the given indices
        self.y_array = self.y_array[ids]
        self.group_array = self.group_array[ids]
        self.p_array = self.p_array[ids]
        self.filename_array = self.filename_array[ids]
        self.metadata_df = self.metadata_df.iloc[ids]

        # recalculate sizes
        self.n_classes = np.unique(self.y_array).size
        self.n_places = np.unique(self.p_array).size
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (
                torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()

    def loader(self):
        return DataLoader(self, batch_size=self.batch_size, pin_memory=True, shuffle=True)


class WaterBirdsDatasetCustom2:

    def __init__(self, basedir, label, test_transform=None, train_transform=None):
        self.test_transform = test_transform
        self.train_transform = train_transform
        self.label = label
        self.train = WaterBirdsDataset(basedir, split="train", transform=self.train_transform)
        self.test = WaterBirdsDataset(basedir, split="test", transform=self.test_transform)
        self.val = WaterBirdsDataset(basedir, split="val", transform=self.test_transform)
