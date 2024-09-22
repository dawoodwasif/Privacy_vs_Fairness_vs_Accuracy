#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    acc_test = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # Initialize lists to store accuracies and variances
    all_rounds_variances = []
    acc_test_rounds = []

    for iter in range(args.epochs):
        w_locals, loss_locals, acc_locals = [], [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local_net = copy.deepcopy(net_glob).to(args.device)
            w, loss = local.train(net=local_net)
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

            # Test using the local model post-training
            local_net.eval()
            acc_test_local, loss_test_local = test_img(local_net, dataset_test, args)

            #print(idx, acc_test_local, loss_test_local)
            acc_locals.append(acc_test_local)

        # Update global weights
        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)
        net_glob.eval()

        # Calculate and store the variance of this round
        round_variance = np.var(acc_locals)
        all_rounds_variances.append(round_variance)

        # Compute and store average testing accuracy for the round
        round_avg_test_accuracy = np.mean(acc_locals)
        acc_test_rounds.append(round_avg_test_accuracy)

        print("\nRound {:3d}, Local models: \nTesting accuracy average: {:.2f}, Testing accuracy variance: {:.4f}".format(iter, round_avg_test_accuracy, round_variance))

        # Test using the updated global model
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        acc_test_rounds.append(acc_test)
        print("Round {:3d}, Global model testing accuracy: {:.2f}".format(iter, acc_test))

    # Calculate the final average variance across all rounds
    final_avg_variance = np.mean(all_rounds_variances)
    print("\nFinal Average Variance (AV): {:.4f}".format(final_avg_variance))

    # Save the average variance and accuracies data to files
    with open('./log/variance_file_{}_{}_{}.dat'.format(args.dataset, args.model, args.epochs), "w") as var_file, \
         open('./log/accfile_{}_{}_{}.dat'.format(args.dataset, args.model, args.epochs), "w") as acc_file:
        for variance, accuracy in zip(all_rounds_variances, acc_test_rounds):
            var_file.write("{:.4f}\n".format(variance))
            acc_file.write("{:.2f}\n".format(accuracy))

    # Plotting the variance and accuracy across rounds
    plt.figure()
    plt.subplot(211)
    plt.plot(range(len(all_rounds_variances)), all_rounds_variances)
    plt.title('Variance and Accuracy of Testing Accuracies Across Rounds')
    plt.ylabel('Testing Accuracy Variance')

    plt.subplot(212)
    plt.plot(range(len(acc_test_rounds)), acc_test_rounds)
    plt.ylabel('Testing Accuracy')
    plt.xlabel('Round')

    plt.savefig('./log/fed_{}_{}_{}_analysis.png'.format(args.dataset, args.model, args.epochs))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))