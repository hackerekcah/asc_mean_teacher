from data_manager.dcase18_taskb import Dcase18TaskbData
from data_manager.taskb_standrizer import TaskbStandarizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class TaskbDevSet (Dataset):
    def __init__(self, mode='train', device='a', transform=None):
        super(TaskbDevSet, self).__init__()

        # x.shape(Bath, Hight, Width)
        self.x, self.y = TaskbStandarizer(data_manager=Dcase18TaskbData()).\
            load_dev_standrized(mode=mode, device=device, norm_device=device)

        self.x = np.expand_dims(self.x, axis=1)

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = (self.x[idx], self.y[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):

    def __call__(self, sample):
        x, y = torch.from_numpy(sample[0]), torch.from_numpy(sample[1])
        x, y = x.type(torch.FloatTensor), y.type(torch.LongTensor)
        return x, y


class ASCDevLoader:
    def __init__(self, device='a'):
        self.device = device
        self.train_set = TaskbDevSet(mode='train', device=device, transform=ToTensor())
        self.val_set = TaskbDevSet(mode='test', device=device, transform=ToTensor())

    def train_val(self, batch_size=128, shuffle=True):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle)

        # Not need to shuffle validation data
        val_loader = DataLoader(dataset=self.val_set, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
