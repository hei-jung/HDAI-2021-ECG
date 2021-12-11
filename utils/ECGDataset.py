import numpy as np
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.X = np.load(data_path, allow_pickle=True)
        self.y = np.load(label_path, allow_pickle=True)

        assert len(self.X) == len(self.y), "length should be same between input and label"

        self.X = torch.FloatTensor(self.X)  # cpu tensor
        self.y = torch.FloatTensor(self.y)  # cpu tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return {
            'X': X,
            'y_target': y
        }
