#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""This file"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse, sys, os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from trainer import fit

import params
from metrics import AccumulatedAccuracyMetric
from networks import EmbeddingNet, ClassificationNet

cuda = torch.cuda.is_available()

__author__ = "TengQi Ye"
__copyright__ = "Copyright 2017-2019"
__credits__ = ["TengQi Ye"]
__license__ = ""
__version__ = "0.0.2"
__maintainer__ = "TengQi Ye"
__email__ = "yetengqi@gmail.com"
__status__ = "Research"


def parse_arguments(argv):
    """Command line parse."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='../data/MNIST', help='The path to MNIST dataset.')

    return parser.parse_args()



def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(10):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=params.colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(params.mnist_classes)
    plt.show()



def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels



def main(args):
    # Prepare dataset.
    train_dataset = MNIST(args.data_path, train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((params.mean,), (params.std,))
                          ]))
    test_dataset = MNIST(args.data_path, train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((params.mean,), (params.std,))
                         ]))


    # Baseline.
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size,
                                              shuffle=False, **kwargs)
    '''
    embedding_net = EmbeddingNet()
    model = ClassificationNet(embedding_net, n_classes=params.n_classes)
    if cuda:
        model.cuda()
    loss_fn = torch.nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, params.n_epochs, cuda, params.log_interval,
        metrics=[AccumulatedAccuracyMetric()])

    train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_loader, model)
    plot_embeddings(train_embeddings_baseline, train_labels_baseline)
    val_embeddings_baseline, val_labels_baseline = extract_embeddings(test_loader, model)
    plot_embeddings(val_embeddings_baseline, val_labels_baseline)
    '''

    # Triplet network
    from datasets import TripletMNIST

    triplet_train_dataset = TripletMNIST(train_dataset)  # Returns triplets of images
    triplet_test_dataset = TripletMNIST(test_dataset)
    batch_size = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True,
                                                       **kwargs)
    triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False,
                                                      **kwargs)

    # Set up the network and training parameters
    from networks import EmbeddingNet, TripletNet
    from losses import TripletLoss

    margin = 1.
    embedding_net = EmbeddingNet()
    model = TripletNet(embedding_net)
    if cuda:
        model.cuda()
    loss_fn = TripletLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer,
        scheduler, params.n_epochs, cuda, params.log_interval)

    train_embeddings_tl, train_labels_tl = extract_embeddings(train_loader, model)
    plot_embeddings(train_embeddings_tl, train_labels_tl)
    val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader, model)
    plot_embeddings(val_embeddings_tl, val_labels_tl)


    # torch.save(model.state_dict(), 'triplet.pt')



if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))