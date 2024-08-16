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


def encrypt_model_weights(weights, context):
    encrypted_weights = {}
    for key, value in weights.items():
        encrypted_weights[key] = ts.ckks_vector(context, value.flatten().tolist())
    return encrypted_weights


def decrypt_model_weights(encrypted_weights, context, original_shape):
    decrypted_weights = {}
    for key, value in encrypted_weights.items():
        decrypted_weights[key] = torch.tensor(value.decrypt()).reshape(original_shape[key])
    return decrypted_weights


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

    # copy weights
    w_glob = net_glob.state_dict()

    # Store the shape of original weights
    original_shape = {k: v.shape for k, v in w_glob.items()}

    # training
    loss_train = []
    acc_test = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    for iter in range(args.epochs):
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            # Encrypt local weights
            encrypted_w = encrypt_model_weights(w, context)

            w_locals.append(copy.deepcopy(encrypted_w))
            loss_locals.append(copy.deepcopy(loss))

        # Decrypt and aggregate global weights
        decrypted_w_locals = [decrypt_model_weights(w, context, original_shape) for w in w_locals]
        w_glob = FedAvg(decrypted_w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        # loss_avg = sum(loss_locals) / len(loss_locals)
        # print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        # loss_train.append(loss_avg)

        # print accuracy
        net_glob.eval()
        # acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_t, loss_t = test_img(net_glob, dataset_test, args)
        #print("Training accuracy: {:.2f}".format(acc_train))
        print("Round {:3d},Testing accuracy: {:.2f}".format(iter, acc_t))

        loss_train.append(loss_t)
        acc_test.append(acc_t)

    # save data to file loss.dat

    # lossfile = open("./lostfile_HE_NIID_600_le1.dat", "w")

    # for lo in loss_train:
    #     slo = str(lo)
    #     lossfile.write(slo)
    #     lossfile.write('\n')
    # lossfile.close()

    accfile = open('./log/accfile_{}_{}_{}_iid{}_HE.dat'.format(args.dataset, args.model, args.epochs, args.iid), "w")

    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()

    # plot loss curve
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel('test accuracy')
    plt.savefig('./log/fed_{}_{}_{}_C{}_iid{}_HE_acc.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
