#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""This file"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse, sys, os
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.model_selection import cross_val_score, train_test_split
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST, ImageFolder
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
    def __init__(self, net, x, y):
        """embedding_net is an object of nn.Module"""
        if isinstance(net, TripletNet):
            self.model = net
        else:
            self.model = TripletNet(net)
        self.model.eval()

        # TODO: check training_set shape and type
        # if next(self.model.parameters()).is_cuda:
        #     self.train_x = self.model.get_embedding(torch.from_numpy(x).cuda())
        # else:
        self.model.cpu()
        self.train_x = self.model.get_embedding(torch.from_numpy(x))

        # cross validation to determine the best k
        self.k_candidates = [1, 3, 5, 7, 9]
        scores = []
        for k in self.k_candidates:
            self.knn = KNeighborsClassifier(n_neighbors=k)
            scores.append(cross_val_score(self.knn, self.train_x.detach().numpy(), y, cv=4).mean())

        self.k = self.k_candidates[np.argmax(scores)] # k of knn
        self.knn.fit(self.train_x.detach().numpy(), y)


    def __call__(self, x):

        try:
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x).cuda()
            elif not x.is_cuda:
                x.cuda()

            self.model.cuda()
        except:
            self.model.cpu()
            x.cpu()

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


def defense(dataset, nn_model_path):
    # Prepare dataset.
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    loader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size,
                                         shuffle=True, **kwargs)

    images, labels = loader2numpy(loader)
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.33, random_state=42
    )

    # kNN training.
    model = torch.load(nn_model_path)
    model.eval()
    ti = TripletInfer(model, train_images, train_labels)

    # Save model.
    print('accuracy:')
    del images, labels, train_images, train_labels, dataset # Delete unnecessary variables.

    # Calculate result batch by batch.
    test_batch_size, batch_idx = 100, 0
    predict_array = []

    while batch_idx + test_batch_size < len(test_labels):
        predict_array.append(ti(torch.from_numpy(
            test_images[batch_idx: batch_idx+test_batch_size, :,:,:]))[0])
        batch_idx += test_batch_size

    predict_array.append(ti(torch.from_numpy(test_images[batch_idx:, :,:,:]))[0])
    predict_array = np.concatenate(predict_array)

    print(accuracy_score(test_labels, predict_array))

    # Save the model.
    with open('ti_infer_bb.pkl', 'wb') as output:
        pickle.dump(ti, output, pickle.HIGHEST_PROTOCOL)


def fgsm_attack(args):
    # TODO: run on all test examples.
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
    model = TripletNet(EmbeddingNet())
    model.load_state_dict(torch.load('/home/tenger/research/metric-learning/siamese-triplet/triplet.pt'))
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
    loss_fn = TripletLoss(params.margin)

    cnt = 0
    for row_idx in range(n_train, n_train+n_test): # Run only on test set.
        row = dis_mat[row_idx, :n_train]

        # Nearest neighbors and corresponding confidences
        nn_indices = row.argsort()[:ti.k]
        nn_labels = train_labels[nn_indices]
        confidence = max(np.bincount(nn_labels))/ti.k

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
    # fgsm_attack(args)
    train_dataset = ImageFolder(root=os.path.join(params.data_root, 'train'),
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((params.mean,), (params.std,))])
                                )

    defense(train_dataset, '/home/tenger/research/metric-learning/siamese-triplet/uaec-pre.pth')

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))