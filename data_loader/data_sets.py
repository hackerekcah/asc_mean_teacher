from torch.utils.data import Dataset
from data_manager.dcase18_taskb import Dcase18TaskbData
from data_manager.taskb_standrizer import TaskbStandarizer
import numpy as np


class TaskbDevSet (Dataset):
    def __init__(self, mode='train', device='a', norm_device=None, transform=None):
        super(TaskbDevSet, self).__init__()

        if not norm_device:
            norm_device = device

        # x.shape(Bath, Hight, Width)
        self.x, self.y = TaskbStandarizer(data_manager=Dcase18TaskbData()).\
            load_dev_standrized(mode=mode, device=device, norm_device=norm_device)

        self.x = np.expand_dims(self.x, axis=1)

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = (self.x[idx], self.y[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample


class TaskbDevDoubleSet (Dataset):
    def __init__(self, mode='train', device='a', norm_device=None, transform=None):
        super(TaskbDevDoubleSet, self).__init__()

        if not norm_device:
            norm_device = device

        # x.shape(Bath, Hight, Width)
        self.x, self.y = TaskbStandarizer(data_manager=Dcase18TaskbData()).\
            load_dev_standrized(mode=mode, device=device, norm_device=norm_device)

        self.x = np.expand_dims(self.x, axis=1)

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample1 = (self.x[idx], self.y[idx])
        sample2 = np.copy(sample1)
        if self.transform:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)
        return sample1, sample2
