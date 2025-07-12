import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class PansharpeningDataset(Dataset):
    def __init__(self, h5_file_path, norm, dr):
        self.h5_file_path = h5_file_path
        with h5py.File(h5_file_path, 'r') as file:
            gt = file['gt'][:]
            gt = np.array(gt, dtype=np.float32)
            self.gt_data = torch.from_numpy(gt)

            lms = file['lms'][:]
            lms = np.array(lms, dtype=np.float32)
            self.lms_data = torch.from_numpy(lms)

            ms = file['ms'][:]
            ms = np.array(ms, dtype=np.float32)
            self.ms_data = torch.from_numpy(ms).to(torch.float32)

            pan = file['pan'][:]
            pan = np.array(pan, dtype=np.float32)
            self.pan_data = torch.from_numpy(pan).to(torch.float32)

            if norm == True:
                self.gt_data = self.gt_data / dr
                self.lms_data = self.lms_data / dr
                self.ms_data = self.ms_data / dr
                self.pan_data = self.pan_data / dr

    def __len__(self):
        return self.gt_data.shape[0]

    def __getitem__(self, idx):
        gt = self.gt_data[idx].float()
        lms = self.lms_data[idx].float()
        ms = self.ms_data[idx].float()
        pan = self.pan_data[idx].float()

        sample = {'gt': gt, 'lms': lms, 'ms': ms, 'pan': pan}

        return sample

