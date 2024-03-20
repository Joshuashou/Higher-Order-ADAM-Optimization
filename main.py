from Adamax_Optimizer import Adamax
import Optimizer
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
import matplotlib.pyplot as plt
import numpy as np
from Resnet import ResNet18
from Higher_Order_Adam import Higher_Moment_Adam
from Higher_Order_Adam import Higher_Moment_Adam_Combination

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')

    parser.add_argument('-f')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')


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
