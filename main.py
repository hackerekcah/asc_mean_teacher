import torch
import os
import net_archs
from data_loader.loader import *
import torch.optim as optim
from engine import *
from utils.check_point import CheckPoint
from utils.history import History
import numpy as np
import logging
torch.manual_seed(0)
np.random.seed(0)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def exp1(device='3', ckpt_prefix='Run01', run_epochs=1000, lr=1e-3):

    # setup logging and save kwargs
    kwargs = locals()
    log_file = '{}/ckpt/exp1/{}.log'.format(ROOT_DIR, ckpt_prefix)
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info(str(kwargs))

    # set up cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    device = torch.device('cuda')

    # load input to cuda
    train_loader, val_loaders = Exp1Loader().train_val()

    # load model to cuda
    model = net_archs.BaseConv(filters=32, is_bn=True, is_drop=True)
    model.to(device)

    # from torchsummary import summary
    # summary(model, input_size=(1, 40, 500))

    # optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    train_hist, val_histp, val_histb = History(name='train/A'), History(name='val/p'), History(name='val/b')

    # checkpoint after new History, order matters
    ckpter = CheckPoint(model=model, optimizer=optimizer, path='{}/ckpt/exp1'.format(ROOT_DIR),
                        prefix=ckpt_prefix, interval=2, save_num=2)

    for epoch in range(1, run_epochs):
        train_hist.add(
            logs=train_model(train_loader, model, optimizer, device),
            epoch=epoch
        )
        val_histp.add(
            logs=eval_model(val_loaders['p'], model, device),
            epoch=epoch
        )

        val_histb.add(
            logs=eval_model(val_loaders['b'], model, device),
            epoch=epoch
        )

        train_hist.clc_plot()
        val_histp.plot()
        val_histb.plot()
        logging.info("Epoch{:04d},{:15},{}".format(epoch, train_hist.name, str(train_hist.recent)))
        logging.info("Epoch{:04d},{:15},{}".format(epoch, val_histp.name, str(val_histp.recent)))
        logging.info("Epoch{:04d},{:15},{}".format(epoch, val_histb.name, str(val_histb.recent)))

        ckpter.check_on(epoch=epoch, monitor='acc', loss_acc=val_histb.recent)
    # explicitly save last
    ckpter.save(epoch=run_epochs-1, monitor='acc', loss_acc=val_histb.recent)


if __name__ == '__main__':
    exp1()
