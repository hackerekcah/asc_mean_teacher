import torch
import torch.nn as nn
from torch.nn import functional as F


class BaseConv(nn.Module):
    """
    dcase2018 baseline
    """
    def __init__(self, filters=32, is_drop=False, is_bn=False):
        super(BaseConv, self).__init__()

        self.is_drop = is_drop
        self.is_bn = is_bn
        self.filters = filters

        # padding same
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=filters, kernel_size=(7, 7), padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=filters)
        self.relu1 = nn.ReLU()
        # TODO: try nn.Dropout2d() in the early layers
        self.pool1 = nn.MaxPool2d((5, 5))
        self.drop1 = nn.Dropout(p=0.3)

        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters * 2, kernel_size=(7, 7), padding=3)
        self.bn2 = nn.BatchNorm2d(num_features=filters * 2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((4, 100))
        self.drop2 = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(in_features=self.filters * 4, out_features=100)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        # conv bn relu drop pool, (batch, 1, 40, 500)
        x = self.conv1(x)
        if self.is_bn:
            x = self.bn1(x)
        x = self.relu1(x)
        if self.is_drop:
            x = self.drop1(x)
        x = self.pool1(x)

        # conv bn relu drop pool, (batch, filters, 8, 100)
        x = self.conv2(x)
        if self.is_bn:
            x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        if self.is_drop:
            x = self.drop2(x)

        # reshape, (batch, filters*2, 2, 1) ->(batch, filters*4)
        x = x.view(-1, self.filters * 4)

        # fc1 relu fc2
        x = self.relu3(self.fc1(x))
        logits = self.fc2(x)

        # (batch, 10)
        return logits

    # def forward(self, x):
    #     # conv bn relu drop pool, (batch, 1, 40, 500)
    #     x = self.conv1(x)
    #     if self.is_bn:
    #         x = self.bn1(x)
    #     x = F.relu(x)
    #     if self.is_drop:
    #         x = self.drop1(x)
    #     x = self.pool1(x)
    #
    #     # conv bn relu drop pool, (batch, filters, 8, 100)
    #     x = self.conv2(x)
    #     if self.is_bn:
    #         x = self.bn2(x)
    #     x = F.relu(x)
    #     x = self.pool2(x)
    #     if self.is_drop:
    #         x = self.drop2(x)
    #
    #     # reshape, (batch, filters*2, 2, 1) ->(batch, filters*4)
    #     x = x.view(-1, self.filters * 4)
    #
    #     # fc1 relu fc2
    #     x = F.relu(self.fc1(x))
    #     logits = self.fc2(x)
    #
    #     # (batch, 10)
    #     return logits

# TODO: 1.BatchNorm2d vs 1d 2.multi-class NLL Loss


