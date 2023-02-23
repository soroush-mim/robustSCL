import os
import argparse
import time
import json
import math

import tensorboard_logger as tb_logger
from torchvision import transforms, datasets
import torch
import torch.backends.cudnn as cudnn

from util import adjust_learning_rate
from util import TwoCropTransform
from util import set_optimizer, save_model
from util import copy_parameters_from_model, copy_parameters_to_model

from adv_train import PGDCons, PGDConsMulti, PGDConsMultiWithOrg
from torch_ema import ExponentialMovingAverage

from networks.resnet_big import SupConCNN
from losses import SupConLoss

from stage1_utils import adv_train2

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=500,
                        help='save frequency')
    ##############
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    parser.add_argument('--ADV_train', action='store_true',
                    help='using adversarial training or not')

    # optimization
    ###########
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')


    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.05,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_false',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    
    parser.add_argument('--model', type=str, default='smallCNN')
    
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--dataset', type=str, default='mnist_binary')
    
    parser.add_argument('--pgd_train_steps', type=int, default=20)
    
    parser.add_argument('--steps_to_use', type=str, default='9', #10 is clean example, 9 is last iteration of PGD
                        help='which attack steps to use(starts from zero, 10 is clean example)')
    
    parser.add_argument('--ema', action='store_true',
                        help='using exponential moving avg')
    parser.add_argument('--ema_decay', type=float, default=0.996)
    

    opt = parser.parse_args()


    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'steps_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_epochs_{}_trial_{}_mnist_binary'.\
        format(opt.steps_to_use, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.epochs, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)
    if opt.ema:
        opt.model_name = '{}_ema{}'.format(opt.model_name, opt.ema_decay)

    # warm-up for large-batch training,
    if opt.batch_size >= 256:
        opt.warm = True
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

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

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

    transform = TwoCropTransform(train_transform)

    train_dataset = datasets.MNIST('../data', train=True, download=True,
                               transform=transform)

    idx_train = get_same_index(train_dataset.train_labels, 1, 3)
    train_dataset.targets = train_dataset.targets[idx_train] - 2
    train_dataset.data = train_dataset.data[idx_train]

    # selected_dataset = torch.utils.data.TensorDataset(selected_data, selected_labels)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader

def set_model(opt):
    model = SupConCNN()
    criterion = SupConLoss(temperature=opt.temp)#, V2=True)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    else:
        raise ValueError('NO CUDA!')

    return model, criterion

def main():
    opt = parse_option()

    train_loader = set_loader(opt)

    model, criterion = set_model(opt)

    optimizer = set_optimizer(opt, model)

    if opt.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=opt.ema_decay)
    else:
        ema=False

    with open(opt.tb_folder + '/stage1_args.txt', 'w') as f:
        json.dump(opt.__dict__, f, indent=2)
    
    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    
    atk = PGDConsMulti(model, eps=0.1, alpha=0.01, steps=opt.pgd_train_steps, random_start=True)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        # train for one epoch
        time1 = time.time()
        loss = adv_train2(train_loader, model, criterion, optimizer, epoch, opt, atk, ema)
        time2 = time.time()

        if ema:
            copy_of_model_parameters = copy_parameters_from_model(model)
            ema.copy_to(model.parameters())
        
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

        if ema:
            copy_parameters_to_model(copy_of_model_parameters, model)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)
    # compare_models(model,model1)

    
if __name__ == '__main__':
    main()
