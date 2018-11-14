from torch.utils.data import Dataset, DataLoader
from data_loader.data_sets import *
from data_loader.transformer import *
import torch


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


class Exp1Loader (object):

    def __init__(self):
        self.trainA = TaskbDevSet(mode='train', device='A', norm_device='A', transform=ToTensor())
        self.valp = TaskbDevSet(mode='test', device='p', norm_device='A', transform=ToTensor())
        self.valb = TaskbDevSet(mode='test', device='b', norm_device='b', transform=ToTensor())

    def train(self, batch_size=128, shuffle=True):
        train_loader = DataLoader(dataset=self.trainA, batch_size=batch_size, shuffle=shuffle)
        return train_loader

    def val(self, batch_size=128):
        loaders = {}
        valp_loader = DataLoader(dataset=self.valp, batch_size=batch_size, shuffle=False)
        valb_loader = DataLoader(dataset=self.valb, batch_size=batch_size, shuffle=False)
        loaders['p'] = valp_loader
        loaders['b'] = valb_loader

        return loaders

    def train_val(self, batch_size=128):
        return self.train(batch_size=batch_size), self.val(batch_size=batch_size)

