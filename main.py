import torch
import os
import net_archs
from data_loader.loader import ASCDevLoader
import torch.optim as optim
from utils.check_point import CheckPoint


def train(train_loader, model, optimizer, device, epoch):
    model.train(mode=True)
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)

        # inside CE: combined LogSoftmax and NLLLoss
        criterion = torch.nn.CrossEntropyLoss()

        _, labels = y.max(dim=1)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch{}:[{}/{} ({:.0f}%)]\tBatchLoss:{:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(x)))


def val(test_loader, model, device, epoch):

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, target = target.max(dim=1)
            logits = model(data)
            # inside CE: combined LogSoftmax and NLLLoss
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(logits, target)

            test_loss += loss.item() # sum up batch loss
            pred = logits.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nEpoch{},Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return {'loss':test_loss, 'acc': correct / len(test_loader.dataset)}


def main():

    torch.manual_seed(1)

    # set up cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device = torch.device('cuda')

    # laod input to cuda
    train_a, val_a = ASCDevLoader(device='a').train_val()

    # load model to cuda
    model = net_archs.BaseConv(filters=32, is_bn=True, is_drop=True)
    model.to(device)

    from torchsummary import summary
    summary(model, input_size=(1, 40, 500))

    # optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=1e-4)

    # checkpoint
    ckpter = CheckPoint(model=model, optimizer=optimizer, path='./ckpt', prefix='Run01', interval=1, save_num=1)

    for epoch in range(10):
        train(train_a, model, optimizer, device, epoch)
        loss_acc = val(val_a, model, device, epoch)
        ckpter.check_on(epoch=epoch, monitor='loss', loss_acc=loss_acc)


if __name__ == '__main__':
    main()
