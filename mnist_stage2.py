import json
import argparse
import time
import math
from copy import deepcopy
import os
import sys

import tensorboard_logger as tb_logger
from torchvision import transforms, datasets
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from main_ce import set_loader
from util import adjust_learning_rate, warmup_learning_rate, accuracy, AverageMeter
from networks.resnet_big import SupConCNN, LinearClassifier

from adv_train import PGDAttack
from trades import trades_loss
from stage2_utils import validate, adv_validate

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='10,15',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='smallCNN')
    parser.add_argument('--dataset', type=str, default='mnist')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    
    parser.add_argument('--binary', action='store_false')
    

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    if opt.binary:
        opt.dataset = '{}_binary'.format(opt.dataset)

    

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    stage1_name = opt.ckpt[:,opt.ckpt.rfind('/')]
    stage1_name = stage1_name[stage1_name.rfind('/')+1:]
    opt.model_name = 'mnist_stage2_lr_{}_decay_{}_bsz_{}_ckpt{}'.\
        format(opt.learning_rate, opt.weight_decay,opt.batch_size, stage1_name)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate


    opt.tb_path = './save/SupCon/{}_tensorboard_stage2'.format(opt.dataset)
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    opt.model_path = './save/SupCon/{}_models_stage2'.format(opt.dataset)

    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)


    if opt.binary:
        opt.n_cls = 2
    else:
        opt.n_cls = 10

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def get_same_index(target, label_1, label_2):
    label_indices = []

    for i in range(len(target)):
        if target[i] == label_1:
            label_indices.append(i)
        if target[i] == label_2:
            label_indices.append(i)
    return label_indices

def set_loader(opt):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize,
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        # normalize,
    ])

    train_dataset = datasets.MNIST('../data', train=True, download=True,
                               transform=train_transform)
        
    val_dataset = datasets.MNIST('../data', train=False, download=True,
                               transform=val_transform)

    if opt.binary:
        idx_train = get_same_index(train_dataset.targets, 1, 2)
        train_dataset.targets = train_dataset.targets[idx_train] - 1
        train_dataset.data = train_dataset.data[idx_train]

        idx_val = get_same_index(val_dataset.targets, 1, 2)
        val_dataset.targets = val_dataset.targets[idx_val] - 1
        val_dataset.data = val_dataset.data[idx_val]

    # selected_dataset = torch.utils.data.TensorDataset(selected_data, selected_labels)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader,val_loader


def set_model(opt):
    model = SupConCNN()
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
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
                            step_size = 0.01,
                            epsilon=0.3,
                            perturb_steps=20,
                            beta=6.0,
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
    

def main():
    best_acc = 0
    best_adv_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    # optimizer = set_optimizer(opt, classifier)
    optimizer = optim.SGD(classifier.parameters(),
                            lr= opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)

    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    with open(opt.tb_folder + '/stage2_args.txt', 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    test_attack = PGDAttack(model, classifier, eps=0.3, alpha = 0.01, steps=40)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()

        loss, acc = train(train_loader, model, classifier, criterion, optimizer, epoch, opt)

        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        clean_loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc

        adv_loss, adv_val_acc = adv_validate(val_loader, model, classifier, criterion, opt, test_attack)
        if adv_val_acc > best_adv_acc:
            best_adv_acc = adv_val_acc
            best_state = deepcopy(classifier.state_dict()) 

        
        logger.log_value('clean loss', clean_loss, epoch)
        logger.log_value('adv loss', adv_loss, epoch)
        logger.log_value('clean acc', val_acc, epoch)
        logger.log_value('adv acc', adv_val_acc, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

    print('best accuracy: {:.2f}'.format(best_acc))
    print('best adv accuracy: {:.2f}'.format(best_adv_acc))
    save_file = os.path.join(
        opt.save_folder, 'best.pth')
    torch.save(best_state , save_file)


if __name__ == '__main__':
    main()