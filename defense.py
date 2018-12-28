#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""This file"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse, sys, os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from datasets import TripletMNIST
from losses import TripletLoss
from networks import EmbeddingNet, TripletNet
import params
from sklearn.metrics import accuracy_score
from trainer import test_epoch



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
    parser.add_argument('--fig_mode', type=str, default=None, help='Plot experiment figures.')

    return parser.parse_args()


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    # perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = image + epsilon * torch.rand(sign_data_grad.shape)
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image



class TripletInfer(object): # TODO
    """Inference using kNN"""
    def __init__(self, model, x, y):

        self.model = TripletNet(EmbeddingNet())
        self.model.load_state_dict(model)
        self.model.eval()

        # TODO: check training_set shape and type
        self.train_x = self.model.get_embedding(torch.from_numpy(x))

        self.knn = KNeighborsClassifier(n_neighbors=3) # TODO: cross-validation
        self.knn.fit(self.train_x.detach().numpy(), y)

    def __call__(self, x):
        embed = self.model.get_embedding(x).cpu().detach().numpy()
        y = self.knn.predict(embed)
        prob = self.knn.predict_proba(embed)
        return y, prob



epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "lenet_mnist_model.pth"
use_cuda=False



def parse_arguments(argv):
    """Command line parse."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='/home/tenger/research/metric-learning/data/MNIST',
                        help='The path to MNIST dataset.')

    return parser.parse_args()



cuda = torch.cuda.is_available()



def gen_hard(ti, data_loader, cls=None, n_examples=None, target_cls=None):
    """Generate adversarial examples from those with lowest confidence."""

    hard_examples = []
    for data, target in data_loader:
        pred_label, pred_conf = ti.model(*data)

        if pred_label == target and np.maximum(pred_conf) < 1: # positive examples
            if cls is not None and target == cls: # for a specific class
                if target_cls is not None and pred_conf[target_cls] > 0: # convert to another specific class
                    hard_examples.append((data, pred_conf))

    hard_examples.sort(key=lambda tup:tup[1])    

    return hard_examples[:n_examples]



def loader2numpy(loader):
    images, labels = [], []
    for data, target in loader:
        images.append(data)
        labels.append(target)

    images = np.concatenate(images)
    labels = np.concatenate(labels)
    return images, labels



def imshow(img):
    npimg = img.detach().numpy()
    npimg = np.squeeze(npimg)
    plt.imshow(npimg, cmap='gray')
    plt.show()



def test(args):
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

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size,
                                              shuffle=False, **kwargs)

    # kNN training
    train_images, train_labels = loader2numpy(train_loader)
    model = torch.load('/home/tenger/research/metric-learning/siamese-triplet/triplet.pt')
    ti = TripletInfer(model, train_images, train_labels)

    # Load test dataset.
    test_images, test_labels = loader2numpy(test_loader)

    # Get embeddings from training and test.
    train_embed, test_embed = ti.model.get_embedding(torch.from_numpy(train_images)), \
                              ti.model.get_embedding(torch.from_numpy(test_images))
    n_train, n_test = train_images.shape[0], test_images.shape[0]

    # Construct distance matrix.
    dis_mat = pairwise_distances(np.concatenate((train_embed.detach().numpy(),
                        test_embed.detach().numpy()))
                                 )

    hard_examples = []
    k=3
    loss_fn = TripletLoss(params.margin)

    cnt = 0
    for row_idx in range(n_train, n_train+n_test):
        row = dis_mat[row_idx, :n_train]

        # nearest neighbors and corresponding confidences
        nn_indices = row.argsort()[:k]
        nn_labels = train_labels[nn_indices]
        confidence = max(np.bincount(nn_labels))/k

        # Hard, positive examples.
        if np.argmax(np.bincount(nn_labels)) == test_labels[row_idx-n_train] \
                and confidence < 1:

            # hard_examples.append(row_idx)
            # Find the furthest example that is the same class as the hard example.
            indices = np.where(train_labels==test_labels[row_idx-n_train])
            fur_pos_idx = indices[0][np.argmax(row[indices])] # furthest example

            # Find nearest the negative example.
            neg_indices = nn_indices[np.where(nn_labels!=test_labels[row_idx-n_train])]
            neg_idx = neg_indices[np.argmin(row[neg_indices])]

            # Calculate the triplet loss.
            anchor_image = torch.from_numpy(test_images[row_idx-n_train])
            anchor_image.requires_grad = True
            anchor = ti.model.get_embedding(anchor_image.unsqueeze(0))

            loss = loss_fn(anchor,
                    train_embed[fur_pos_idx].unsqueeze(0),
                    train_embed[neg_idx].unsqueeze(0)
                    )

            # back-propagation
            ti.model.zero_grad()

            since = time.time()
            loss.backward(retain_graph=True)
            print('Cost =', time.time() - since)

            data_grad = anchor_image.grad.data

            # Call FGSM Attack
            perturbed_data = fgsm_attack(anchor_image, params.epsilon, data_grad)  # TODO: multiple epsilons

            # plot origin, added noise and sythetic image.
            imshow(anchor_image)
            imshow(params.epsilon * data_grad)
            imshow(perturbed_data)

            # Re-classify the perturbed image
            print(ti(perturbed_data.unsqueeze(0)), test_labels[row_idx-n_train],
                  ti(anchor_image.unsqueeze(0)))

            anchor_image.detach()

            cnt += 1
            if cnt == 30:
                break


    ''' 
    TODO: fabricate an adversarial attack.
    1. [x] Triplet loss is different from softmax loss. Perform correct loss.
    2. [x] Perform correct gradient and generate corresponding adversarial examples with labels.
    3. [ ] Exceptional case: (adversarial examples can be ordered).
    4. [ ] Show test result and sample some successful and failed adversarial examples.
    '''



def main(args):
    test(args)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))