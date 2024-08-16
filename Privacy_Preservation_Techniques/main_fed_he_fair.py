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

import tenseal as ts
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

def split_tensor(tensor, max_size):
    """Splits tensor into chunks of size max_size."""
    flat_tensor = tensor.flatten()
    return [flat_tensor[i:i+max_size] for i in range(0, len(flat_tensor), max_size)]

def encrypt_model_weights(weights, context, max_size):
    encrypted_weights = {}
    for key, value in weights.items():
        chunks = split_tensor(value, max_size)
        encrypted_weights[key] = [ts.ckks_vector(context, chunk.tolist()) for chunk in chunks]
    return encrypted_weights

def decrypt_model_weights(encrypted_weights, original_shape, max_size):
    decrypted_weights = {}
    for key, chunks in encrypted_weights.items():
        decrypted_chunks = [torch.tensor(chunk.decrypt()) for chunk in chunks]
        flat_tensor = torch.cat(decrypted_chunks)
        decrypted_weights[key] = flat_tensor.reshape(original_shape[key])
    return decrypted_weights

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

    # Initialize TenSEAL context with increased poly_modulus_degree
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=16384, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2**40
    context.generate_galois_keys()

    # Determine max size for CKKS vectors
    max_size = 8192  # Adjust based on poly_modulus_degree and input size

    # copy weights
    w_glob = net_glob.state_dict()

    # Store the shape of original weights
    original_shape = {k: v.shape for k, v in w_glob.items()}

    # training
    loss_train = []
    acc_test_rounds = []
    all_rounds_variances = []
    all_rounds_distances = []

    for iter in range(args.epochs):
        w_locals, loss_locals, acc_locals = [], [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local_net = copy.deepcopy(net_glob).to(args.device)
            w, loss = local.train(net=local_net)

            # Encrypt local weights
            encrypted_w = encrypt_model_weights(w, context, max_size)

            w_locals.append(copy.deepcopy(encrypted_w))
            loss_locals.append(copy.deepcopy(loss))

            # Test using the local model post-training
            local_net.eval()
            acc_test_local, loss_test_local = test_img(local_net, dataset_test, args)
            acc_locals.append(acc_test_local)

        # Decrypt and aggregate global weights
        decrypted_w_locals = [decrypt_model_weights(w, original_shape, max_size) for w in w_locals]
        w_glob = FedAvg(decrypted_w_locals)

        # copy weight to net_glob
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
