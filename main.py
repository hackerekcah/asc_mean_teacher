import torch
import os
import net_archs
from data_loader.loader import ASCDevLoader
import torch.optim as optim
from engine import train_model, train_mixup, eval_model
from utils.check_point import CheckPoint
torch.manual_seed(0)


def main():

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
    ckpter = CheckPoint(model=model, optimizer=optimizer, path='./ckpt', prefix='Run01,Mixup', interval=3, save_num=3)

    from utils.history import History
    train_hist = History(name='train')
    val_hist = History(name='val')
    for epoch in range(10):
        train_mixup(train_a, model, optimizer, device)
        train_loss_acc = eval_model(train_a, model, device)
        val_loss_acc = eval_model(val_a, model, device)
        print("Epoch{}train".format(epoch), train_loss_acc)
        print("Epoch{}val".format(epoch), val_loss_acc)
        ckpter.check_on(epoch=epoch, monitor='loss', loss_acc=val_loss_acc)
        train_hist.add(train_loss_acc, epoch)
        val_hist.add(val_loss_acc, epoch)


if __name__ == '__main__':
    main()
