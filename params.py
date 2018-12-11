#!/usr/bin/env python
# -*- coding: utf-8 -*-

mean, std = 0.1307, 0.3081

mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

n_classes = 10
batch_size = 256
n_epochs = 20
log_interval = 50
lr = 1e-2

margin = 1.0
epsilon = 0.05