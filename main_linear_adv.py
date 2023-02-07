from __future__ import print_function
from distutils.log import error
from gc import freeze

import sys
import json
import argparse
import time
import math
from copy import deepcopy

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn


import tensorboard_logger as tb_logger

from main_ce import set_loader
from trades import trades_loss
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from utils_advCL import eval_adv_test
from networks.resnet_big import SupConResNet, LinearClassifier
from adv_train import PGDAttack
from stage2_utils import *

import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='7',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    
    parser.add_argument

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'stage2{}_{}_lr_{}_decay_{}_bsz_{}_without_ema'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

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

    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


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

    test_attack = PGDAttack(model, classifier, eps=8./255., alpha = 2./255., steps=50)
    train_attack = PGDAttack(model, classifier, eps=8./255., alpha = 2./255., steps=10)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        # loss, acc = PGDtrain(train_loader, model, classifier, criterion,
        #                   optimizer, epoch, opt, train_attack)

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
    torch.save(best_state , 'LOSSV2_satge2_7,8,9_10_resnet18_lr_0.1_decay_0.0005_bsz_200_temp_0.07_trial_0_cosine.pth')


if __name__ == '__main__':
    main()
    