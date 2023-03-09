import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from main_ce import set_loader
from trades import trades_loss
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from utils_advCL import eval_adv_test
from networks.resnet_big import SupConResNet, LinearClassifier
from adv_train import PGDAttack

import sys
import time


mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).cuda()
std = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).cuda()

def normalize(X):
        return (X - mu)/std

def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder, device_ids = [0,1])
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt, freeze = True):
    """one epoch training"""
    model.eval()
    if not freeze:
        model.train()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        if freeze:
            with torch.no_grad():
                features = model.encoder(images)
        else:
            features = model.encoder(images)
        output = classifier(features.detach())
        # loss = criterion(output, labels)

        loss = trades_loss( model, classifier,
                            images,
                            labels,
                            optimizer,
                            step_size = 2./255,
                            epsilon=8./255.,
                            perturb_steps=10,
                            beta=1.0,
                            distance='l_inf')

        # update metric
        losses.update(loss.item(), bsz)
        acc1 = accuracy(output, labels, topk=(1,))[0]
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def PGDtrain(train_loader, model, classifier, criterion, optimizer, epoch, opt, attack):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        adv_images = attack(images, labels)
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            adv_features = model.encoder(adv_images)
            features = model.encoder(images)
        adv_output = classifier(adv_features.detach())
        output = classifier(features.detach())
        # loss = criterion(adv_output, labels)
        loss = (criterion(adv_output, labels) + criterion(output, labels)) / 2


        # update metric
        losses.update(loss.item(), bsz)
        acc1 = accuracy(adv_output, labels, topk=(1,))[0]
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg



def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    correct = 0
    total = 0

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            # images = images.float().cuda()
            images = images.cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # update metric
            losses.update(loss.item(), bsz)
            acc1 = accuracy(output, labels, topk=(1,))[0]
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    print('standard acc: ', 100 * correct // total)
    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def adv_validate(val_loader, model, classifier, criterion, opt, attack):
    """adv validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(val_loader):
        images = images.float().cuda()
        labels = labels.cuda()
        bsz = labels.shape[0]
        input_adv = attack(images, labels)

        # forward
        with torch.no_grad():
            output = classifier(model.encoder(input_adv))
            loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1 = accuracy(output, labels, topk=(1,))[0]
        top1.update(acc1[0], bsz)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            print('adv Test: [{0}/{1}]\t'
                    'adv Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'adv Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'adv Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1))

    print('adv * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def adv_validate_con(val_loader, model, classifier, criterion, opt, attack):
    """adv validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    # losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(val_loader):
        images = images.float().cuda()
        labels = labels.cuda()
        bsz = labels.shape[0]
        input_adv = attack(images, labels, criterion)[-1]

        # forward
        with torch.no_grad():
            features = model.encoder(input_adv)
            output = classifier(features)
            # loss = criterion(features, labels)

        # update metric
        # losses.update(loss.item(), bsz)
        acc1 = accuracy(output, labels, topk=(1,))[0]
        top1.update(acc1[0], bsz)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            print('adv con Test: [{0}/{1}]\t'
                    'adv con Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #'adv con Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'adv con Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time,top1=top1))

    print('adv con * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg