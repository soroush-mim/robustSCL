import argparse
from tsne_torch import TorchTSNE as TSNE

from torchvision import transforms, datasets
import torch
import torch.backends.cudnn as cudnn

import numpy as np
import matplotlib.pyplot as plt



def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--binary', action='store_false')
    

    opt = parser.parse_args()



def set_loader(binary):

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

    if binary:
        idx_train = get_same_index(train_dataset.targets, 1, 3)
        train_dataset.targets = train_dataset.targets[idx_train] - 2
        train_dataset.data = train_dataset.data[idx_train]

        idx_val = get_same_index(val_dataset.targets, 1, 3)
        val_dataset.targets = val_dataset.targets[idx_train] - 2
        val_dataset.data = val_dataset.data[idx_train]
        

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1024, shuffle=True,
        num_workers=8, pin_memory=True)

    return train_loader,val_loader



def set_model(ckpt_path):
    model = SupConCNN()

    ckpt = torch.load(ckpt_path, map_location='cpu')
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
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model



def get_reps(val_loader, model):
    """validation"""
    model.eval()
    classifier.eval()

    outputs = []
    adv_outputs = []
    all_labels = []

    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            # images = images.float().cuda()
            images = images.cuda()
            labels = labels.cuda()
            input_adv = attack(images, labels)

            # forward
            outputs.append(model.encoder(images))
            adv_outputs.append(model.encoder(input_adv))
            all_labels.append(labels) 
    
    outputs = torch.stack(outputs,2)
    adv_outputs = torch.stack(adv_outputs,2)
    all_labels = torch.stack(all_labels)

    return outputs, adv_outputs, all_labels


opt = parse_option()

train_loader, val_loader = set_loader(opt.binary)

# build model and criterion
model = set_model(opt.ckpt)

reps, adv_reps, labels = get_reps(val_loader,model)

emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(reps)
adv_emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(adv_reps)
labels = labels.numpy()
# fig = plt.figure(figsize=(8,8))
plt.scatter(emb[:, 0], emb[:, 1], 20, labels)
plt.savefig('clean.png')
plt.clf()
plt.scatter(adv_emb[:, 0], adv_emb[:, 1], 20, labels)
plt.savefig('adv.png')



