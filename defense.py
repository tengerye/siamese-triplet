#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""This file"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse, sys, os

from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.optim as optim

from datasets import TripletMNIST
from losses import TripletLoss
from networks import EmbeddingNet, TripletNet
import numpy as np
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
    perturbed_image = image + epsilon * torch.rand(sign_data_grad.shape).cuda()
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

'''
def test( model, device, test_loader, epsilon ):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
'''


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

    parser.add_argument('--data_path', type=str, default='../data/MNIST', help='The path to MNIST dataset.')

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

    triplet_train_dataset = TripletMNIST(train_dataset)  # Returns triplets of images
    triplet_test_dataset = TripletMNIST(test_dataset)

    # kNN training
    images, labels = [], []
    for data, target in train_loader:
        images.append(data)
        labels.append(target)

    images = np.concatenate(images)
    labels = np.concatenate(labels)
    model = torch.load('./triplet.pt')
    ti = TripletInfer(model, images, labels)

    '''
    # kNN inference data preparation
    y_pred, y_true = [], []

    for data, target in test_loader:
        y_pred.append(data) # Save predictions' confidence only.
        y_true.append(target.numpy())

    print(accuracy_score(np.concatenate(y_true), np.concatenate(y_pred)))
    '''

    # Accuracy counter2
    correct = 0
    adv_examples = []
    losses = []

    # Loop over all examples in test set
    ''' 
    TODO: fabricate an adversarial attack.
    1. [x] Triplet loss is different from softmax loss. Perform correct loss.
    2. [ ] Perform correct gradient and generate corresponding adversarial examples with labels.
    3. [ ] Exceptional case: (adversarial examples can be ordered).
    4. [ ] Show test result and sample some successful and failed adversarial examples.
    '''
    hard_examples = gen_hard(ti, test_loader, cls=None, n_examples=None, target_cls=None)

    for data, target in hard_examples:

        if cuda:
            data = tuple(d.unsqueeze(0).cuda() for d in data)
            data[0].requires_grad = True

        optimizer = optim.Adam(ti.model.parameters(), lr=params.lr)
        optimizer.zero_grad()
        ti.model.cuda()
        outputs = ti.model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_fn = TripletLoss(params.margin)
        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

        # Calculate the loss
        if loss.item() != 0:
            # loss = F.nll_loss(outputs, target)

            # Zero all existing gradients
            ti.model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = data[0].grad.data

            # Call FGSM Attack
            perturbed_data = fgsm_attack(data[0], params.epsilon, data_grad) # TODO: multiple epsilons

            # Re-classify the perturbed image
            print(ti(perturbed_data))

            break



def main(args):
    test(args)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))