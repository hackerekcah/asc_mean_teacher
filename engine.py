import torch
from data_loader.mixup import mixup_data, mixup_criterion


def ts_train(src_loader, dst_loader, student, teacher, device, rampup_weight):
    for batch_idx, ((xl, y), ((xu, _), (xu_, _))) in enumerate(zip(src_loader, dst_loader)):
        xl, y, xu, xu_ = xl.to(device), y.to(device), xu.to(device), xu_.to(device)
        # onehot to int
        _, y = y.max(dim=1)
        student.feed_labeled(xl, y)
        student.learn_from(xu=xu, teacher=teacher.feed_unlabeled(xu_), rampup_weight=rampup_weight)
        student.update()
        teacher.update()


def train_mixup(train_loader, model, optimizer, device):
    model.train(mode=True)
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        # onehot to int
        _, y = y.max(dim=1)

        x, y1, y2, lam = mixup_data(x, y, alpha=1, use_cuda=True)

        logits = model(x)

        # inside CE: combined LogSoftmax and NLLLoss
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')

        loss = mixup_criterion(criterion, logits, y1, y2, lam)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_model(train_loader, model, optimizer, device):
    model.train(mode=True)
    train_loss = 0
    correct = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        _, y = y.max(dim=1)

        logits = model(x)

        # inside CE: combined LogSoftmax and NLLLoss
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        with torch.no_grad():
            pred = logits.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)
    return {'loss': train_loss, 'acc': train_acc}


def eval_model(test_loader, model, device):

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, target = target.max(dim=1)
            logits = model(data)
            # inside CE: combined LogSoftmax and NLLLoss
            criterion = torch.nn.CrossEntropyLoss(reduction='sum')
            loss = criterion(logits, target)

            test_loss += loss.item() # sum up batch loss
            pred = logits.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    # print('\nEpoch{},Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     epoch, test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    return {'loss': test_loss, 'acc': test_acc}