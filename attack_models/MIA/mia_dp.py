import os
import torch
import numpy as np
from torchvision import datasets, transforms
from models.Nets import MLP, CNNMnist, CNNCifar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings

# Load Predictor Functions
from main_qfed_dp_fair import predict_and_evaluate as predict_dp
from main_qfed_smc_fair_with_parties import predict_and_evaluate as predict_smc
from main_qfed_he_fair import predict_and_evaluate as predict_he

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

# Function to load dataset
def load_dataset(dataset_name, iid=True):
    if dataset_name == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        dict_users = mnist_iid(dataset_train, 100) if iid else mnist_noniid(dataset_train, 100)
    elif dataset_name == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        dict_users = cifar_iid(dataset_train, 100) if iid else cifar_noniid(dataset_train, 100)
    else:
        raise ValueError("Unrecognized dataset")
    return dataset_train, dataset_test, dict_users

# Function to initialize model
def initialize_model(args):
    if args.model == 'cnn' and args.dataset == 'cifar':
        model = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        model = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        if args.dataset == 'mnist':
            len_in = 28 * 28
        elif args.dataset == 'cifar':
            len_in = 32 * 32 * 3
        model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        raise ValueError("Unrecognized model")
    return model

# Function to train the shadow model
def train_shadow_model(dataset_train, dataset_test, dict_users, args, predictor_func):
    model = initialize_model(args)
    model, acc, variance, distance = predictor_func(args, model, dataset_train, dataset_test, dict_users)
    return model

# Function to get confidence scores
def get_confidence_scores(model, data_loader, args):
    model.eval()
    all_scores = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(args.device)
            output = model(data)
            probs = torch.nn.functional.softmax(output, dim=1)
            all_scores.extend(probs.cpu().numpy())
    return np.array(all_scores)

# Function to validate and clean confidence scores
def clean_confidence_scores(confidence_scores):
    # Replace NaN and infinity with 0, and clip excessively large values
    confidence_scores = np.nan_to_num(confidence_scores, nan=0.0, posinf=1.0, neginf=0.0)
    confidence_scores = np.clip(confidence_scores, 0, 1)
    return confidence_scores

# Function to train the attack model
def train_attack_model(shadow_scores, labels):
    shadow_scores = clean_confidence_scores(shadow_scores)  # Clean the confidence scores
    X_train, X_test, y_train, y_test = train_test_split(shadow_scores, labels, test_size=0.3, random_state=42)
    attack_model = LogisticRegression(max_iter=1000)
    attack_model.fit(X_train, y_train)
    predictions = attack_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Attack Model Accuracy: {accuracy:.2f}')
    return attack_model

# Function to perform MIA attack
def perform_mia_attack(attack_model, target_scores):
    target_scores = clean_confidence_scores(target_scores)  # Clean the target scores
    predictions = attack_model.predict(target_scores)
    success_rate = np.mean(predictions)
    print(f'MIA Attack Success Rate: {success_rate:.2f}')
    return success_rate

# Function to evaluate MIA success rate for different epsilon values
def evaluate_mia_at_epsilons(epsilons, dataset_train, dataset_test, dict_users, args):
    mia_results = {}
    for epsilon in epsilons:
        print(f"\nEvaluating MIA at epsilon: {epsilon}")
        args.epsilon = epsilon  # Set the epsilon value

        # Train and get predictions from the shadow model for DP
        shadow_model_dp = train_shadow_model(dataset_train, dataset_test, dict_users, args, predict_dp)

        # Create a DataLoader for the test dataset
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.bs, shuffle=False)

        # Get confidence scores from shadow model
        shadow_scores_dp = get_confidence_scores(shadow_model_dp, test_loader, args)

        # Labels: 1 for member (in training set), 0 for non-member
        labels = np.concatenate([np.ones(len(shadow_scores_dp) // 2), np.zeros(len(shadow_scores_dp) // 2)])

        # Train the attack model
        attack_model_dp = train_attack_model(shadow_scores_dp, labels)

        # Get confidence scores from target models (assuming the same predictor functions can be used)
        target_scores_dp = get_confidence_scores(shadow_model_dp, test_loader, args)

        # Perform MIA Attack
        success_rate = perform_mia_attack(attack_model_dp, target_scores_dp)

        # Store the success rate for the current epsilon
        mia_results[epsilon] = success_rate
    
    return mia_results

if __name__ == '__main__':
    # Define default arguments (suitable for running in a notebook)
    class Args:
        def __init__(self):
            self.q = 0
            self.epochs = 10
            self.num_users = 100
            self.frac = 0.1
            self.local_ep = 5
            self.local_bs = 10
            self.bs = 128
            self.lr = 0.01
            self.momentum = 0.5
            self.split = 'user'
            self.model = 'cnn'
            self.dataset = 'mnist'
            self.iid = True
            self.num_classes = 10
            self.num_channels = 1
            self.device = torch.device('cpu')
            self.seed = 1
            self.verbose = False
            self.gpu = -1

            # DP parameters
            self.epsilon = 16.0
            self.delta = 1e-5
            self.sensitivity = 1
            self.dp_mechanism = 'laplace'

    args = Args()

    # Load dataset
    dataset_train, dataset_test, dict_users = load_dataset(args.dataset, args.iid)

    # List of epsilon values to evaluate
    epsilons = [16, 32, 64]
    #epsilons = [32]


    # Evaluate MIA success rate at different epsilon values
    mia_results = evaluate_mia_at_epsilons(epsilons, dataset_train, dataset_test, dict_users, args)

    # Print out the MIA results for each epsilon
    print("\nMIA Success Rates at Different Epsilon Values:")
    for epsilon, success_rate in mia_results.items():
        print(f"Epsilon {epsilon}: MIA Success Rate = {success_rate:.2f}")
