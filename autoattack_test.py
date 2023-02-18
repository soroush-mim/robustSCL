import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, datasets
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from networks.resnet_big import SupConResNet,LinearClassifier
from adv_train import PGDCons , PGDConsMulti

from autoattack import AutoAttack

import os
import numpy as np


def parse_option():
    parser = argparse.ArgumentParser('argument for autoattack test')

    parser.add_argument('--encoder_ckpt', type=str)
    parser.add_argument('--model_ckpt', type=str)

    opt = parser.parse_args()

    return opt
    
class ClassifierModel(nn.Module):
    "encoder + classifier"
    def __init__(self, encoder, linearClassifier):
        super(ClassifierModel, self).__init__()
        self.encoder = encoder
        self.linearClassifier = linearClassifier

    def forward(self, x):
        return self.linearClassifier(self.encoder(x))


def set_model_linear(encoder_ckpt, classifier_ckpt):
    model = SupConResNet(name='resnet18')
    classifier = LinearClassifier(name='resnet18', num_classes=10)

    ckpt = torch.load(encoder_ckpt,map_location='cpu')
    state_dict = ckpt['model']

    classifier_state = torch.load(classifier_ckpt, map_location='cpu' )

    if torch.cuda.is_available():
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
        classifier.load_state_dict(classifier_state)
    else:
        raise KeyError

    return model, classifier


def set_loader_linear(): #select only 1000 samples from cifar10
    cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True)

    val_transform = transforms.Compose([transforms.ToTensor()])

    # Set the random seed for reproducibility
    np.random.seed(42)

    # Select 1000 random indices from the test set
    random_indices = np.random.choice(len(cifar_testset), size=1000, replace=False)

    # Create a subset of the test set using the selected indices
    cifar_subset = torch.utils.data.Subset(cifar_testset, random_indices)

    # Create a DataLoader for the selected samples
    test_loader = torch.utils.data.DataLoader(cifar_subset, batch_size=100, shuffle=False, num_workers=8, pin_memory=True)

    return test_loader


if __name__ == '__main__':

    opt = parse_option()

    val_loader = set_loader_linear()

    model, classifier = set_model_linear(opt.encoder_ckpt, opt.classifier_ckpt)
    CModel = ClassifierModel(model.encoder, classifier)


    
    save_dir = './results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load attack    
    CModel.eval()
    adversary = AutoAttack(CModel, norm='Linf', eps=8./255., version='standard', log_path='./log_file.txt')

    l = [x for (x, y) in val_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in val_loader]
    y_test = torch.cat(l, 0)

    # run attack and save images
    with torch.no_grad():
            adv_complete = adversary.run_standard_evaluation(x_test, y_test,bs=1000)