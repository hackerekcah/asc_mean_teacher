import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from data_loader.loader import ASCDevLoader
from data_manager.dcase18_taskb import Dcase18TaskbData
import os

learning_rate = 0.001
EPOCH = 60
BATCH_SIZE=64
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,\
                      kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (5, 5), 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 10 * 125, 128)
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1,  32 * 10 * 125)
        x = F.relu(self.fc(x))
        x = self.out(x)
        return x

def main():
    torch.manual_seed(1)
    #os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device('cuda')
    cnn = CNN()
    cnn.to(device)
    print(cnn)
    cnn.train(mode=True)
    # load data
    train_a, val_a = ASCDevLoader(device='a').train_val(batch_size=BATCH_SIZE)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()
    print(len(train_a))
    for epoch in range(EPOCH):
        running_loss = 0.0
        acc = 0
        for step, (b_x, b_y) in enumerate(train_a):

            b_x = b_x.to(device)
            b_y = b_y.to(device)
            _, labels = b_y.max(dim=1)
            output = cnn(b_x)
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # compute test accuracy
        test_loss = 0.0
        with torch.no_grad():
            for step, (test_x, test_y) in enumerate(val_a):
                test_x, test_y = test_x.to(device), test_y.to(device)
                print(test_y.size())
                _, labels = test_y.max(dim=1)
                output = cnn(test_x)
                loss = loss_func(output, labels)
                test_loss += loss.item()
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                acc += pred.eq(labels.view_as(pred)).sum().item()

        print('epoch: ', epoch, 'loss: ', 'test loss: ', test_loss / len(val_a.dataset), 'accuracy: ', acc / len(val_a.dataset))


if __name__ == '__main__':
    main()