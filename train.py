import argparse
import logging
import os
import time
import torch
import yaml
import shutil
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage

from warmup_scheduler import GradualWarmupScheduler

from tools import utils
from adv_train import *

scaler = torch.cuda.amp.GradScaler()


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str,
        default="configs/train/train_supcon_resnet18_cifar10_stage2.yml",
    )

    parser_args = parser.parse_args()

    with open(vars(parser_args)["config_name"], "r") as config_file:
        hyperparams = yaml.full_load(config_file)

    return hyperparams


if __name__ == "__main__":
    # parse hyperparameters
    hyperparams = parse_config()
    print(hyperparams)

    backbone = hyperparams["model"]["backbone"]
    ckpt_pretrained = hyperparams['model']['ckpt_pretrained']
    num_classes = hyperparams['model']['num_classes']
    amp = hyperparams['train']['amp']
    ema = hyperparams['train']['ema']
    ema_decay_per_epoch = hyperparams['train']['ema_decay_per_epoch']
    n_epochs = hyperparams["train"]["n_epochs"]
    logging_name = hyperparams['train']['logging_name']
    target_metric = hyperparams['train']['target_metric']
    stage = hyperparams['train']['stage']
    data_dir = hyperparams["dataset"]
    optimizer_params = hyperparams["optimizer"]
    scheduler_params = hyperparams["scheduler"]
    criterion_params = hyperparams["criterion"]
    device = hyperparams['device']
    warmup = hyperparams['warmup']

    batch_sizes = {
        "train_batch_size": hyperparams["dataloaders"]["train_batch_size"],
        'valid_batch_size': hyperparams['dataloaders']['valid_batch_size']
    }
    num_workers = hyperparams["dataloaders"]["num_workers"]

    torch.cuda.set_device(device)

    if not amp: scaler = None

    utils.seed_everything()
