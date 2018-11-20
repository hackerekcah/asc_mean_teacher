import torch
import math
import logging


class Student (object):
    def __init__(self, model, lr=1e-4, teacher_weight=3):
        self.model = model
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        self.cls_criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        self.unsup_criterion = torch.nn.MSELoss(reduction='elementwise_mean')

        self.total_loss = 0
        self.cls_loss = None
        self.consistency_loss = None
        self.teacher_weight = teacher_weight

        self._step = 0

    def feed_labeled(self, xl, y):
        self.model.train()
        logits = self.model(xl)
        self.cls_loss = self.cls_criterion(logits, y)
        self.total_loss += self.cls_loss
        logging.debug("Step:{},cls_loss:{}".format(self._step, self.cls_loss.item()))
        return self.cls_loss.item()

    def _consistency_loss(self, student_logits, teacher_logits):
        student_prob = torch.nn.functional.softmax(student_logits, dim=1)
        teacher_prob = torch.nn.functional.softmax(teacher_logits, dim=1)
        return self.unsup_criterion(student_prob, teacher_prob)

    def learn_from(self, xu, teacher, rampup_weight):
        self.model.train()
        student_logits = self.model(xu)
        self.consistency_loss = self._consistency_loss(student_logits, teacher.logits)
        weighted_consistency_loss = self.teacher_weight * rampup_weight * self.consistency_loss
        self.total_loss += weighted_consistency_loss
        logging.debug("Step:{},consistency_loss:{}".format(self._step, self.consistency_loss.item()))
        logging.debug("Step:{},weighted_consistency_loss:{}".format(self._step, weighted_consistency_loss.item()))
        return weighted_consistency_loss.item()

    def update(self):
        self.optimizer.zero_grad()
        self.total_loss.backward()
        self.optimizer.step()

        # reset total loss to zero, otherwise will accumulate loss over epochs
        self.total_loss = 0
        self._step += 1


class Teacher (object):
    def __init__(self, model):
        self.model = model

        # for saving memory
        for param in list(self.model.parameters()):
            param.requires_grad = False

        self.optimizer = None
        self.logits = None

    def bind(self, student, teacher_alpha=0.99):
        logging.info("WeightEMA teacher_alpha:{}".format(teacher_alpha))
        self.optimizer = WeightEMA(self.model.parameters(), student.model.parameters(), alpha=teacher_alpha)
        return self

    def feed_unlabeled(self, ux):
        self.model.train()
        self.logits = self.model(ux)
        return self

    def update(self):
        self.optimizer.step()


class WeightEMA (object):
    """
    Exponential moving average weight optimizer for mean teacher model
    """
    def __init__(self, params, src_params, alpha=0.99):
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)


class RampUp (object):

    def __init__(self, rampup_epochs=80):
        self.rampup_epochs = rampup_epochs
        self._rampup_counter = 0

    def get_weight(self):
        self._rampup_counter += 1
        if self._rampup_counter <= self.rampup_epochs:
            p = max(0.0, float(self._rampup_counter)) / float(self.rampup_epochs)
            p = 1.0 - p
            weight = math.exp(-p * p * 5.0)
        else:
            weight = 1.0
        logging.debug("Epoch{},rampup_weight{:.3f}".format(self._rampup_counter, weight))
        return weight


