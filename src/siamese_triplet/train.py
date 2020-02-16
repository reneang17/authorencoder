#Within the present repo, this is a modified version 
#of the computer vision repo https://github.com/adambielski/siamese-triplet
#by Adam Bielski


import torch
import numpy as np
import numpy
from sklearn.neighbors import KNeighborsClassifier as KNC


def simplified_fit(train_loader, val_loader, model, loss_fn, optimizer, n_epochs, is_cuda_available, metrics=[],
        start_epoch=0, scheduler = None, log_interval=1):
    """
    TODO
    """
    train_list = []
    valid_list = []
    log_interval = len(train_loader)//2
    if scheduler != None:
        for epoch in range(0, start_epoch):
            scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        if scheduler != None:
            scheduler.step()

        # Train stage
        train_loss, _metrics = train_epoch(train_loader, model, loss_fn, optimizer, is_cuda_available, log_interval, metrics)
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in _metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            train_list.append(metric.value())



        # Validation stage
        if val_loader != None:
            val_loss, _metrics = test_epoch(val_loader, model, loss_fn, is_cuda_available, metrics)
            val_loss /= len(val_loader)
            message += '\nEpoch: {}/{}. Validation set: Avg loss: {:.4f}'.format(epoch + 1, n_epochs,val_loss)
            for metric in _metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())
                valid_list.append(metric.value())
        

    return (train_list, valid_list)



def train_epoch(train_loader, model, loss_fn, optimizer, is_cuda_available, log_interval, metrics):
    """
    TODO
    """
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    outs=[];targs=[]
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if is_cuda_available:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)
        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)
        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0][0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []
    total_loss /= (batch_idx + 1)
    return total_loss, metrics

def test_epoch(val_loader, model, loss_fn, is_cuda_available, metrics):
    """
    TODO
    """
    outs=[];targs=[]

    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for (data, target) in val_loader:
            

            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if is_cuda_available:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)
            outs.append(outputs.detach())
            targs.append(target.detach())

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()
            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
