import torch


class ToTensor(object):

    def __call__(self, sample):
        x, y = torch.from_numpy(sample[0]), torch.from_numpy(sample[1])
        x, y = x.type(torch.FloatTensor), y.type(torch.LongTensor)
        return x, y
