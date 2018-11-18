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
torch.manual_seed(0)
np.random.seed(0)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def run():

    # set up cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device = torch.device('cuda')

    # load input to cuda
    src_loader, dst_double_loader = UdaLoader().train(batch_size=128, shuffle=True)

    # load for eval
    src_loader_eval, dst_loader_eval = UdaLoader().train_for_eval(batch_size=128, shuffle=False)

    # load model to cuda
    student_model = net_archs.BaseConv(filters=32, is_bn=True, is_drop=True)
    student_model.to(device)

    teacher_model = net_archs.BaseConv(filters=32, is_bn=True, is_drop=True)
    teacher_model.to(device)

    student = Student(student_model)
    teacher = Teacher(teacher_model).bind(student)

    train_hist = dict()
    train_hist['A/T'] = History(name='teacher/train/A')
    train_hist['b/T'] = History(name='teacher/train/b')
    train_hist['A/S'] = History(name='student/train/A')
    train_hist['b/S'] = History(name='student/train/b')

    # checkpoint after new History, order matters
    teacher_ckpter = CheckPoint(model=teacher.model, optimizer=None, path='{}/ckpt/teacher'.format(ROOT_DIR),
                                prefix='Run01', interval=2, save_num=2)
    teacher_ckpter.bind_histories([train_hist['A/T'], train_hist['b/T']])

    student_ckpter = CheckPoint(model=student.model, optimizer=student.optimizer, path='{}/ckpt/student'.format(ROOT_DIR),
                                prefix='Run01', interval=2, save_num=2)
    student_ckpter.bind_histories([train_hist['A/S'], train_hist['b/S']])

    # rampup 80 epochs
    rampup = RampUp(rampup_epochs=200)

    for epoch in range(1, 1000):
        ts_train(src_loader, dst_double_loader, student, teacher, device, rampup.get_weight())
        train_hist['A/T'].add(
            logs=eval_model(src_loader_eval, teacher.model, device),
            epoch=epoch
        )
        train_hist['b/T'].add(
            logs=eval_model(dst_loader_eval, teacher.model, device),
            epoch=epoch
        )
        train_hist['A/S'].add(
            logs=eval_model(src_loader_eval, student.model, device),
            epoch=epoch
        )
        train_hist['b/S'].add(
            logs=eval_model(dst_loader_eval, student.model, device),
            epoch=epoch
        )
        train_hist['A/T'].clc_plot()
        train_hist['b/T'].plot()
        train_hist['A/S'].plot()
        train_hist['b/S'].plot()
        teacher_ckpter.check_on(epoch=epoch, monitor='acc', loss_acc=train_hist['b/T'].recent)
        student_ckpter.check_on(epoch=epoch, monitor='acc', loss_acc=train_hist['b/S'].recent)
