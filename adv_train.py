from __future__ import print_function

import os
import sys
import argparse
import time
import math

from torchattacks import PGD

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).cuda()
std = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).cuda()

def normalize(X):
    return (X - mu)/std

class PGDConsMulti(PGD):
    def __init__(self, model, eps=0.3,
                alpha=2/255, steps=40, random_start=True):
      
      super().__init__(model, eps, alpha, steps, random_start)

    def forward(self, images, labels, loss):
      r"""
      Overridden.
      """
      # images = torch.cat([images[0], images[1]], dim=0)
      images = images.clone().detach().to(self.device)
      labels = labels.clone().detach().to(self.device)
      
      adv_images = images.clone().detach()
      bsz = labels.shape[0]

      if self.random_start:
          # Starting at a uniformly random point
          adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
          adv_images = torch.clamp(adv_images, min=0, max=1).detach()

      attacks = []
      for _ in range(self.steps):
          adv_images.requires_grad = True
          outputs = self.model(adv_images)
          f1, f2 = torch.split(outputs, [bsz, bsz], dim=0) #f1 and f2 -> torch.Size([bsz, 128]
          outputs = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) 

          # Calculate loss
          cost = loss(outputs, labels)

          # Update adversarial images
          grad = torch.autograd.grad(cost, adv_images,
                                      retain_graph=False, create_graph=False)[0]

          adv_images = adv_images.detach() + self.alpha*grad.sign()
          delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
          adv_images = torch.clamp(images + delta, min=0, max=1).detach()
          attacks.append(adv_images)

      return attacks


class PGDCons(PGD):
    def __init__(self, model, eps=0.3,
                alpha=2/255, steps=40, random_start=True):
      
      super().__init__(model, eps, alpha, steps, random_start)

    def forward(self, images, labels, loss):
      r"""
      Overridden.
      """
      # images = torch.cat([images[0], images[1]], dim=0)
      images = images.clone().detach().to(self.device)
      labels = labels.clone().detach().to(self.device)
      print('device: ', self.device)
      
      adv_images = images.clone().detach()
      bsz = labels.shape[0]
      
      if self.random_start:
          # Starting at a uniformly random point
          adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
          adv_images = torch.clamp(adv_images, min=0, max=1).detach()

      for _ in range(self.steps):
          adv_images.requires_grad = True
          outputs = self.model(adv_images)
          f1, f2 = torch.split(outputs, [bsz, bsz], dim=0) #f1 and f2 -> torch.Size([bsz, 128]
          outputs = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) 

          # Calculate loss
          cost = loss(outputs, labels)

          # Update adversarial images
          grad = torch.autograd.grad(cost, adv_images,
                                      retain_graph=False, create_graph=False)[0]

          adv_images = adv_images.detach() + self.alpha*grad.sign()
          delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
          adv_images = torch.clamp(images + delta, min=0, max=1).detach()

      return adv_images

class PGDAttack(PGD):
    def __init__(self, model, classifier, eps=0.3,
                alpha=2/255, steps=40, random_start=True):

      
      super().__init__(model, eps, alpha, steps, random_start)
      self.classifier = classifier

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)


        loss = torch.nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.classifier(self.model.encoder(adv_images))

            cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images