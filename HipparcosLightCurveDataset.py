import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

class HipparcosLightCurveDataset(Dataset):
    def __init__(self, file_dir, labels_dict=None, transform=None, max_len=500):
        """
        Args:
          file_dir (str): Path to folder with .tbl files
          labels_dict (dict): Optional dict mapping filename to label (0/1)
          transform (callable): Optional transform to apply on data
          max_len (int): Max length of sequences; shorter sequences are padded
        """
        self.file_dir = file_dir
        self.file_list = [f for f in os.listdir(file_dir) if f.endswith('.tbl')]
        self.labels_dict = labels_dict
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        filepath = os.path.join(self.file_dir, filename)

        # Read the tbl file (skip comments)
        df = pd.read_csv(filepath, delim_whitespace=True, comment='#', 
                         names=['JD', 'Hp', 'Hp_err', 'Flag'])

        # Preprocess: extract magnitude, normalize (example)
        mag = df['Hp'].values.astype(np.float32)
        mag = (mag - mag.mean()) / mag.std()  # normalize
        
        # Pad or trim to max_len
        if len(mag) < self.max_len:
            mag = np.pad(mag, (0, self.max_len - len(mag)), 'constant')
        else:
            mag = mag[:self.max_len]

        mag = torch.tensor(mag).unsqueeze(1)  # shape (max_len, 1)

        # Get label if available
        label = -1  # default unknown
        if self.labels_dict and filename in self.labels_dict:
            label = self.labels_dict[filename]
        label = torch.tensor(label).long()

        if self.transform:
            mag = self.transform(mag)

        return mag, label
