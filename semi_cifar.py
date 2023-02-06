import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from PIL import Image



class psudoSoftLabel_CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, model, **kwds):
        super().__init__(**kwds)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.label = []
        # generate psudo label
        imgs = []
        for cnt, img in enumerate(self.data):
            img = Image.fromarray(img).convert('RGB')
            img = transform_test(img)
            img = img.cuda()
            imgs.append(img)

            if cnt % 100 == 99:
                imgs = torch.stack(imgs)
                print("generating psudo label {}/{}".format(cnt, len(self.data)))
                pred = model.eval()(imgs)
                self.label += pred.cpu().detach().numpy().tolist()
                imgs = []

        print("len self.label is {}".format(len(self.label)))

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        img = self.transform(img)

        psudoLabel = torch.FloatTensor(self.label[idx])
        real_label = self.targets[idx]

        return img, psudoLabel, real_label


# train_datasets_psudolabeled = psudoSoftLabel_CIFAR10(root=args.data, train=True, download=True, transform=transform_train, model=gene_net)
