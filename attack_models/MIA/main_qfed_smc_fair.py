import os
import torch
import numpy as np
from torchvision import datasets, transforms
from models.Nets import MLP, CNNMnist, CNNCifar
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from models.Update import LocalUpdate
from models.Fed import FedAvg
from models.test import test_img
import argparse
import copy 

def args_parser():
    parser = argparse.ArgumentParser()
    # Federated arguments
    parser.add_argument('--q', type=float, default=10, help="q-FedSGD q value")
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")  # Lowered learning rate
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True', help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    # SMC parameters
    parser.add_argument('--num_shares', type=int, default=5, help="number of SMC shares")
    parser.add_argument('--threshold', type=int, default=3, help="threshold for SMC reconstruction")

    return parser.parse_args()


def euclidean_distance(local_accuracies, global_accuracy):
    """Calculate Euclidean distance between local accuracies and the global accuracy."""
    return np.sqrt(np.sum((np.array(local_accuracies) - global_accuracy) ** 2))


############################


def apply_smc(gradient, num_shares, threshold):
    """Splits gradient into random shares for SMC."""
    shares = [torch.rand_like(gradient) for _ in range(num_shares - 1)]
    final_share = gradient - sum(shares)
    shares.append(final_share)
    return shares

def reconstruct_smc(shares, threshold, original_gradient):
    """Reconstruct the gradient from the minimum number of shares (threshold), normalize to match the original gradient's norm."""
    reconstructed_gradient = sum(shares[:threshold])
    
    # Avoid corrupt values by checking for invalid norms
    reconstructed_norm = torch.norm(reconstructed_gradient)
    original_norm = torch.norm(original_gradient)
    
    # Log a warning if the gradient norms differ drastically
    if abs(reconstructed_norm - original_norm) > 1e3:
        print(f"Warning: Large gradient discrepancy detected. Reconstructed norm: {reconstructed_norm:.4f}, Original norm: {original_norm:.4f}")

    # Normalize only if the norm has changed significantly
    if reconstructed_norm > 0 and abs(reconstructed_norm - original_norm) > 1e-5:
        normalization_factor = original_norm / reconstructed_norm
        reconstructed_gradient *= normalization_factor
    
    # Clip the gradient to prevent it from exploding
    reconstructed_gradient = torch.clamp(reconstructed_gradient, min=-1e5, max=1e5)
    
    return reconstructed_gradient

def predict_and_evaluate(args, net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()
    w_glob = net_glob.state_dict()

    acc_test_rounds = []
    all_rounds_variances = []
    all_rounds_distances = []

    for iter in range(args.epochs):
        w_locals, h_locals, acc_locals = [], [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local_net = copy.deepcopy(net_glob).to(args.device)
            w, _ = local.train(net=local_net)

            # Log the gradient norm before applying SMC
            gradient_norm_before = torch.norm(list(w.values())[0]).item()  # Example for the first layer
            print(f"Gradient norm before SMC: {gradient_norm_before:.4f}")

            # Apply SMC to the model gradients
            for key in w.keys():
                smc_shares = apply_smc(w[key], args.num_shares, args.threshold)
                w[key] = reconstruct_smc(smc_shares, args.threshold, w[key])  # Pass original gradient for normalization

            # Log the gradient norm after reconstructing with SMC
            gradient_norm_after = torch.norm(list(w.values())[0]).item()  # Example for the first layer
            print(f"Gradient norm after SMC: {gradient_norm_after:.4f}")

            delta_w_k = copy.deepcopy(w)
            for lk in delta_w_k.keys():
                delta_w_k[lk] = delta_w_k[lk] - w_glob[lk]

            L_w = sum(torch.norm(delta_w_k[lk]) ** 2 for lk in delta_w_k.keys())
            h_k = args.q * (L_w ** (args.q - 1)) + 1e-10

            w_locals.append(copy.deepcopy(w))
            h_locals.append(h_k)

            local_net.eval()
            acc_test_local, _ = test_img(local_net, dataset_test, args)
            acc_locals.append(acc_test_local)

        w_glob = copy.deepcopy(w_locals[0])
        for lk in w_glob.keys():
            w_glob[lk] = sum(h_k * w[lk] for w, h_k in zip(w_locals, h_locals)) / sum(h_locals)

        net_glob.load_state_dict(w_glob)
        net_glob.eval()

        round_variance = np.var(acc_locals)
        all_rounds_variances.append(round_variance)
        round_avg_test_accuracy = np.mean(acc_locals)
        acc_test_rounds.append(round_avg_test_accuracy)
        round_distance = euclidean_distance(acc_locals, round_avg_test_accuracy)
        all_rounds_distances.append(round_distance)

        print(f"\nRound {iter}, Avg Test Accuracy: {round_avg_test_accuracy:.2f}, Variance: {round_variance:.4f}, Euclidean Distance: {round_distance:.4f}")

        acc_test, _ = test_img(net_glob, dataset_test, args)
        acc_test_rounds.append(acc_test)
        print(f"Round {iter}, Global Test Accuracy: {acc_test:.2f}")

    final_avg_variance = np.mean(all_rounds_variances)
    final_avg_distance = np.mean(all_rounds_distances)
    print(f"\nFinal Avg Variance: {final_avg_variance:.4f}")
    print(f"Final Avg Euclidean Distance: {final_avg_distance:.4f}")

    return net_glob, acc_test_rounds[-1], final_avg_variance, final_avg_distance




###############################



if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cpu')

    # Load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
        args.num_channels = 1
    else:
        raise ValueError("Dataset not recognized!")

    # Build the model
    if args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 28 * 28
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        raise ValueError("Model not recognized!")

    net_glob, final_accuracy, final_variance, final_distance = predict_and_evaluate(args, net_glob, dataset_train, dataset_test, dict_users)

    # Save model weights
    weights_dir = './weights/q-fedavg_SMC_{}'.format(args.q)
    os.makedirs(weights_dir, exist_ok=True)
    weights_filename = os.path.join(weights_dir, 'model_weights_final.pth')
    torch.save(net_glob.state_dict(), weights_filename)
    print(f"Weights saved successfully in {weights_filename}")

    # Final evaluation on training and testing datasets
    acc_train, _ = test_img(net_glob, dataset_train, args)
    acc_test, _ = test_img(net_glob, dataset_test, args)
    print(f"Training Accuracy: {acc_train:.2f}")
    print(f"Testing Accuracy: {acc_test:.2f}")

