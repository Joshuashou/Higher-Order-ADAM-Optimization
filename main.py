import Adamax
import Vanilla_Adam
from torchvision.utils import draw_segmentation_masks
import math
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
from itertools import chain
import torch.optim as optim
import os
import argparse
import sys
from collections import defaultdict
import math
from typing import cast, List, Optional, Dict, Tuple
from typing import Callable, Dict
from torch.utils.hooks import RemovableHandle
import torch
from torch import Tensor
from collections import OrderedDict, defaultdict, abc as container_abcs
import functools
import matplotlib.pyplot as plt
import numpy as np

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    start = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        adam_optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        adam_optimizer.step()


        #print(loss.item())

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% | Train Loss: %.3f'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, train_loss))
    end = time.time()
    total_time = start - end
    print("Training Loss: " + str(train_loss))
    print("Accuracy Train: " + str(100.*correct/total))
     
    accuracy_train = 100.*correct/total


    return train_loss, total_time, accuracy_train


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    start = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    end = time.time()
    total_time = start - end

    accuracy_test = 100.*correct/total

    print("Accuracy Test: " + str(accuracy_test))

    return test_loss, total_time, accuracy_test

if __name__ == 'main':

    loss_epoch_train = []
    loss_epoch_test = []
    train_times = []
    test_times = []
    accuracy_train = []
    accuracy_test = []



    net = ResNet18()

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True


    criterion = nn.CrossEntropyLoss()
    #adam_optimizer = Higher_Moment_Adam(net.parameters(),num_moment = 4, lr=args.lr)


    adam_optimizer = Higher_Moment_Adam_Combination(net.parameters(), lr=args.lr)
    for epoch in range(start_epoch, start_epoch+50):

        train_loss, train_time, train_acc = train(epoch)
        test_loss, test_time, test_acc = test(epoch)
        train_times.append(train_time)
        test_times.append(test_time)
        loss_epoch_train.append(train_loss)
        loss_epoch_test.append(test_loss)
        accuracy_train.append(train_acc)
        accuracy_test.append(test_acc)

    Beta_pairs = [[0.9, 0.9], [0.9, 0.99], [0.9, 0.9999]] #Various experimentations with linear combinations
