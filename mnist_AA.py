import torch.nn as nn
from torchvision import transforms, datasets
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from networks.resnet_big import SupConCNN,LinearClassifier

from autoattack import AutoAttack

import os
import argparse
import numpy as np

def parse_option():
    parser = argparse.ArgumentParser('argument for autoattack test')

    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--classifier_ckpt', type=str)
    parser.add_argument('--binary', action='store_false')

    opt = parser.parse_args()

    if opt.binary:
        opt.n_cls = 2
    else:
        opt.n_cls = 10

    return opt
    

def get_same_index(target, label_1, label_2):
    label_indices = []

    for i in range(len(target)):
        if target[i] == label_1:
            label_indices.append(i)
        if target[i] == label_2:
            label_indices.append(i)
    return label_indices

class ClassifierModel(nn.Module):
    "encoder + classifier"
    def __init__(self, encoder, linearClassifier):
        super(ClassifierModel, self).__init__()
        self.encoder = encoder
        self.linearClassifier = linearClassifier

    def forward(self, x):
        return self.linearClassifier(self.encoder(x))


def set_model_linear(opt):
    model = SupConCNN()
    
    classifier = LinearClassifier(name='smallCNN', num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt,map_location='cpu')
    state_dict = ckpt['model']

    classifier_state = torch.load(opt.classifier_ckpt, map_location='cpu' )

    if torch.cuda.is_available():
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
        classifier.load_state_dict(classifier_state)
    else:
        raise KeyError

    return model, classifier


def set_loader_linear(opt): #select only 1000 samples from mnist if not binary
    
    val_transform = transforms.Compose([transforms.ToTensor()])

    cifar_testset = datasets.MNIST(root='../data', train=False, download=True, transform=val_transform)

    if opt.binary:

        idx_val = get_same_index(cifar_testset.targets, 1, 2)
        cifar_testset.targets = cifar_testset.targets[idx_val] - 1
        cifar_testset.data = cifar_testset.data[idx_val]


        test_loader = torch.utils.data.DataLoader(
        cifar_testset, batch_size=1000, shuffle=False,
        num_workers=8, pin_memory=True)

    else:

        # Set the random seed for reproducibility
        np.random.seed(42)

        # Select 1000 random indices from the test set
        random_indices = np.random.choice(len(cifar_testset), size=1000, replace=False)

        # Create a subset of the test set using the selected indices
        cifar_subset = torch.utils.data.Subset(cifar_testset, random_indices)

        # Create a DataLoader for the selected samples
        test_loader = torch.utils.data.DataLoader(cifar_subset, batch_size=1000, shuffle=False, num_workers=8, pin_memory=True)
    

    return test_loader



if __name__ == '__main__':

    opt = parse_option()

    val_loader = set_loader_linear(opt)

    model, classifier = set_model_linear(opt)
    CModel = ClassifierModel(model.encoder, classifier)

    save_dir = './results_mnist'
    if opt.binary:
        save_dir = save_dir + '_binary'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    stage1_name = opt.ckpt[:opt.ckpt.rfind('/')]
    stage1_name = stage1_name[stage1_name.rfind('/')+1:]

    # load attack    
    CModel.eval()
    adversary = AutoAttack(CModel, norm='Linf', eps=8./255., version='standard', log_path='{}/{}_log_file.txt'.format(save_dir,stage1_name))

    l = [x for (x, y) in val_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in val_loader]
    y_test = torch.cat(l, 0)

    # run attack and save images
    with torch.no_grad():
            adv_complete = adversary.run_standard_evaluation(x_test, y_test,bs=1000)
