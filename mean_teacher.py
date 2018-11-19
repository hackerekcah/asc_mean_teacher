import torch
import os
import net_archs
from data_loader.loader import *
import torch.optim as optim
from engine import *
from utils.check_point import CheckPoint
from utils.history import History
import numpy as np
from role import *
import timeit
import logging
torch.manual_seed(0)
np.random.seed(0)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def run(device='3', ckpt_prefix='Run01', rampup_epochs=80, run_epochs=1000,
        lr=1e-3, teacher_weight=3, teacher_ema_alhpa=0.999, log_level='DEBUG'):

    # setup logging and save kwargs
    kwargs = locals()
    log_file = '{}/ckpt/mean_teacher/{}.log'.format(ROOT_DIR, ckpt_prefix)
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    logging.basicConfig(filename=log_file, level=getattr(logging, log_level.upper(), None))
    logging.info(str(kwargs))

    # set up cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    device = torch.device('cuda')

    uda_loader = UdaLoader()

    # load train
    src_loader, dst_double_loader = uda_loader.train(batch_size=128, shuffle=True)

    # load train for eval
    src_loader_eval, dst_loader_eval = uda_loader.train_for_eval(batch_size=128, shuffle=False)

    # load val
    val_loaders = uda_loader.val(batch_size=128, shuffle=False)

    # load model to cuda
    student_model = net_archs.BaseConv(filters=32, is_bn=True, is_drop=True)
    student_model.to(device)

    teacher_model = net_archs.BaseConv(filters=32, is_bn=True, is_drop=True)
    teacher_model.to(device)

    student = Student(student_model, lr=lr, teacher_weight=teacher_weight)
    teacher = Teacher(teacher_model).bind(student, teacher_alpha=teacher_ema_alhpa)

    train_hist = dict()
    train_hist['T/A'] = History(name='teacher/train/A')
    train_hist['T/b'] = History(name='teacher/train/b')
    train_hist['S/A'] = History(name='student/train/A')
    train_hist['S/b'] = History(name='student/train/b')

    val_hist = dict()
    val_hist['T/p'] = History(name='teacher/val/p')
    val_hist['T/b'] = History(name='teacher/val/b')
    val_hist['S/p'] = History(name='student/val/p')
    val_hist['S/b'] = History(name='student/val/b')

    # checkpoint after new History, order matters
    teacher_ckpter = CheckPoint(model=teacher.model, optimizer=None, path='{}/ckpt/mean_teacher/teacher'.format(ROOT_DIR),
                                prefix=ckpt_prefix, interval=2, save_num=2)
    teacher_ckpter.bind_histories([train_hist['T/A'], train_hist['T/b'], val_hist['T/p'], val_hist['T/b']])

    student_ckpter = CheckPoint(model=student.model, optimizer=student.optimizer, path='{}/ckpt/mean_teacher/student'.format(ROOT_DIR),
                                prefix=ckpt_prefix, interval=2, save_num=2)
    student_ckpter.bind_histories([train_hist['S/A'], train_hist['S/b'], val_hist['S/p'], val_hist['S/b']])

    # rampup 80 epochs
    rampup = RampUp(rampup_epochs=rampup_epochs)

    for epoch in range(1, run_epochs):

        ts_train(src_loader, dst_double_loader, student, teacher, device, rampup.get_weight())

        train_hist['T/A'].add(
            logs=eval_model(src_loader_eval, teacher.model, device),
            epoch=epoch
        )
        train_hist['T/b'].add(
            logs=eval_model(dst_loader_eval, teacher.model, device),
            epoch=epoch
        )
        val_hist['T/p'].add(
            logs=eval_model(val_loaders['p'], teacher.model, device),
            epoch=epoch
        )
        val_hist['T/b'].add(
            logs=eval_model(val_loaders['b'], teacher.model, device),
            epoch=epoch
        )

        train_hist['S/A'].add(
            logs=eval_model(src_loader_eval, student.model, device),
            epoch=epoch
        )
        train_hist['S/b'].add(
            logs=eval_model(dst_loader_eval, student.model, device),
            epoch=epoch
        )

        val_hist['S/p'].add(
            logs=eval_model(val_loaders['p'], student.model, device),
            epoch=epoch
        )
        val_hist['S/b'].add(
            logs=eval_model(val_loaders['b'], student.model, device),
            epoch=epoch
        )

        train_hist['T/A'].clear()

        for key in train_hist.keys():
            # train_hist[key].plot()
            logging.info("Epoch{:04d},{:15},{}".format(epoch, train_hist[key].name, str(train_hist[key].recent)))

        for key in val_hist.keys():
            val_hist[key].plot()
            logging.info("Epoch{:04d},{:15},{}".format(epoch, val_hist[key].name, str(val_hist[key].recent)))

        teacher_ckpter.check_on(epoch=epoch, monitor='acc', loss_acc=val_hist['T/b'].recent)
        student_ckpter.check_on(epoch=epoch, monitor='acc', loss_acc=val_hist['S/b'].recent)

    # explicitly save the last run
    teacher_ckpter.save(epoch=run_epochs-1, monitor='acc', loss_acc=val_hist['T/b'].recent)
    student_ckpter.save(epoch=run_epochs-1, monitor='acc', loss_acc=val_hist['T/b'].recent)
