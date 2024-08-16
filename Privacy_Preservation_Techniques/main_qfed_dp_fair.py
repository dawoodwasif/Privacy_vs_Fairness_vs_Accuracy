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
import argparse

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--q', type=float, default=10, help="q-FedAvg q value")

    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    return parser.parse_args()

def euclidean_distance(local_accuracies, global_accuracy):
    return np.sqrt(np.sum((np.array(local_accuracies) - global_accuracy) ** 2))

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
    acc_test_rounds = []
    all_rounds_variances = []
    all_rounds_distances = []

    for iter in range(args.epochs):
        w_locals, h_locals, loss_locals, acc_locals = [], [], [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local_net = copy.deepcopy(net_glob).to(args.device)
            w, loss = local.train(net=local_net)

            # Calculate Δw_k^t = L(w^t - \tilde{w}_k^{t+1})
            delta_w_k = copy.deepcopy(w)
            for lk in delta_w_k.keys():
                delta_w_k[lk] = delta_w_k[lk] - w_glob[lk]

            # Calculate L(w^t - \tilde{w}_k^{t+1})
            L_w = 0
            for lk in delta_w_k.keys():
                L_w += torch.norm(delta_w_k[lk]) ** 2
            
            # Calculate h_k^t
            h_k = args.q * (L_w ** (args.q - 1)) + 1e-10  # Add small constant to avoid division by zero

            # Store the local weight and h_k
            w_locals.append(copy.deepcopy(w))
            h_locals.append(h_k)

            # Test using the local model post-training
            local_net.eval()
            acc_test_local, loss_test_local = test_img(local_net, dataset_test, args)
            acc_locals.append(acc_test_local)

        # Update global weights
        w_glob = copy.deepcopy(w_locals[0])
        for lk in w_glob.keys():
            w_glob[lk] = sum(h_k * w[lk] for w, h_k in zip(w_locals, h_locals)) / sum(h_locals)

        net_glob.load_state_dict(w_glob)
        net_glob.eval()

        # Calculate and store the variance of this round
        round_variance = np.var(acc_locals)
        all_rounds_variances.append(round_variance)

        # Compute and store average testing accuracy for the round
        round_avg_test_accuracy = np.mean(acc_locals)
        acc_test_rounds.append(round_avg_test_accuracy)

        # Calculate Euclidean Distance
        round_distance = euclidean_distance(acc_locals, round_avg_test_accuracy)
        all_rounds_distances.append(round_distance)

        print("\nRound {:3d}, Local models: \nTesting accuracy average: {:.2f}, Testing accuracy variance: {:.4f}, Euclidean Distance: {:.4f}".format(iter, round_avg_test_accuracy, round_variance, round_distance))

        # Test using the updated global model
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        acc_test_rounds.append(acc_test)
        print("Round {:3d}, Global model testing accuracy: {:.2f}".format(iter, acc_test))

    # Calculate the final average variance and distance across all rounds
    final_avg_variance = np.mean(all_rounds_variances)
    final_avg_distance = np.mean(all_rounds_distances)
    print("\nFinal Average Variance (AV): {:.4f}".format(final_avg_variance))
    print("Final Average Euclidean Distance (ED): {:.4f}".format(final_avg_distance))

    # Save the average variance, distances, and accuracies data to files
    with open('./log/variance_file_{}_{}_{}.dat'.format(args.dataset, args.model, args.epochs), "w") as var_file, \
         open('./log/accfile_{}_{}_{}.dat'.format(args.dataset, args.model, args.epochs), "w") as acc_file, \
         open('./log/distance_file_{}_{}_{}.dat'.format(args.dataset, args.model, args.epochs), "w") as dist_file:
        for variance, accuracy, distance in zip(all_rounds_variances, acc_test_rounds, all_rounds_distances):
            var_file.write("{:.4f}\n".format(variance))
            acc_file.write("{:.2f}\n".format(accuracy))
            dist_file.write("{:.4f}\n".format(distance))

    # Plotting the variance, distance, and accuracy across rounds
    plt.figure()
    plt.subplot(311)
    plt.plot(range(len(all_rounds_variances)), all_rounds_variances)
    plt.title('Variance, Euclidean Distance, and Accuracy of Testing Accuracies Across Rounds')
    plt.ylabel('Testing Accuracy Variance')

    plt.subplot(312)
    plt.plot(range(len(all_rounds_distances)), all_rounds_distances)
    plt.ylabel('Euclidean Distance')

    plt.subplot(313)
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


############################### SAVE MODEL WEIGHTS ###########################
import os

# Custom name option
custom_name = 'sample_dp'

# Create a directory to save the weights if it doesn't exist
weights_dir = os.path.join('./weights', '{}_{}_{}_{}_weights'.format(args.dataset, args.model, args.epochs, custom_name))
os.makedirs(weights_dir, exist_ok=True)

# Save the global model's weights
weights_filename = os.path.join(weights_dir, 'model_weights_final.pth')
torch.save(net_glob.state_dict(), weights_filename)

print(f"Weights saved successfully in {weights_filename}")
