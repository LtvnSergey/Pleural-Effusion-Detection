import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class ImageDataset(Dataset):
    def __init__(self, dir_input, test_size, is_valid=None, normalization=None):
        """
        Initialize images and masks datasets
        :param dir_input: directory with input data
        :param test_size: size of validation dataset
        :param is_valid: defines validation or training set
        :param normalization: normalization to be applied to input images
        """

        # Read images
        self.images = np.load(os.path.join(dir_input, 'images.npz'))

        # Read masks
        self.masks = np.load(os.path.join(dir_input, 'masks.npz'))

        # Read meta-file
        self.meta = pd.read_csv(os.path.join(dir_input,
                                             'meta_file.csv'), sep='\t')

        # Split data into train and validation sets if necessary
        if is_valid:
            idx_train, idx_val, _, _ = train_test_split(self.meta['index'].values,
                                                        self.meta['Effusion'].values,
                                                        test_size=test_size,
                                                        random_state=42)
            if is_valid == True:
                self.meta = self.meta.iloc[idx_val]
            else:
                self.meta = self.meta.iloc[idx_train]

        if normalization:
            self.normalization = T.Normalize(std=443, mean=-720)

    def __getitem__(self, idx):
        """
        Get item from dataset according to index from meta-file
        :param idx: index
        """
        index = 'arr_' + str(self.meta.iloc[idx]['index'])

        mask = T.ToTensor()(self.masks[index]).type(torch.float)
        image = T.ToTensor()(self.images[index]).type(torch.float)

        if self.normalization:
            image = self.normalization(image)

        return image, mask

    def __len__(self):
        """
        Get length of dataset
        """
        return len(self.meta)
