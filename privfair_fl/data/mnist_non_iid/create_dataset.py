import os, json, sys
import gzip
import numpy as np
import random
import math

from tqdm import trange
from PIL import Image

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels

def generate_dataset():
    X_train, y_train = load_mnist('raw_data/mnist', kind='train')
    X_test, y_test = load_mnist('raw_data/mnist', kind='t10k')

    # Simple normalization
    mu = np.mean(X_train.astype(np.float32), axis=0)
    sigma = np.std(X_train.astype(np.float32), axis=0)

    X_train_whole = (X_train.astype(np.float32) - mu) / (sigma + 0.001)
    X_test_whole = (X_test.astype(np.float32) - mu) / (sigma + 0.001)

    X_train = []
    X_test = []

    for i in range(10):
        idx = np.where(y_train == i)[0]
        X_train.append(X_train_whole[idx].tolist())
        idx = np.where(y_test == i)[0]
        X_test.append(X_test_whole[idx].tolist())

    return X_train, y_train.tolist(), X_test, y_test.tolist()

def main():
    NUM_USER = 50
    CLASSES_PER_USER = 2  # Each user gets data from only 2 classes

    train_output = "./data/train/mytrain_non_iid.json"
    test_output = "./data/test/mytest_non_iid.json"

    # Create the necessary directories
    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    os.makedirs(os.path.dirname(test_output), exist_ok=True)

    X_train, _, X_test, _ = generate_dataset()

    X_user_train = [[] for _ in range(NUM_USER)]
    y_user_train = [[] for _ in range(NUM_USER)]

    X_user_test = [[] for _ in range(NUM_USER)]
    y_user_test = [[] for _ in range(NUM_USER)]

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    np.random.seed(233)
    num_samples = np.random.lognormal(3, 1, NUM_USER) + 10
    num_samples = 60000 * num_samples / sum(num_samples)

    # Assign random classes to each user
    user_classes = [random.sample(range(10), CLASSES_PER_USER) for _ in range(NUM_USER)]
    
    idx_train = np.zeros(10, dtype=np.int64)
    idx_test = np.zeros(10, dtype=np.int64)
    
    for user in range(NUM_USER):
        props = np.random.lognormal(1, 1, CLASSES_PER_USER)
        props = props / sum(props)
        for j in range(CLASSES_PER_USER):
            class_id = user_classes[user][j]
            train_sample_this_class = int(props[j] * num_samples[user]) + 1
            test_sample_this_class = int(props[j] * num_samples[user] / 6) + 1

            if idx_train[class_id] + train_sample_this_class > len(X_train[class_id]):
                idx_train[class_id] = 0
            if idx_test[class_id] + test_sample_this_class > len(X_test[class_id]):
                idx_test[class_id] = 0

            X_user_train[user] += X_train[class_id][idx_train[class_id]: (idx_train[class_id] + train_sample_this_class)]
            X_user_test[user] += X_test[class_id][idx_test[class_id]: (idx_test[class_id] + test_sample_this_class)]

            y_user_train[user] += (class_id * np.ones(train_sample_this_class)).tolist()
            y_user_test[user] += (class_id * np.ones(test_sample_this_class)).tolist()

            idx_train[class_id] += train_sample_this_class
            idx_test[class_id] += test_sample_this_class

        print('User:', user, 'Classes:', user_classes[user])
        print('num train:', len(X_user_train[user]), 'num test:', len(X_user_test[user]))
        print('train labels:', np.unique(np.asarray(y_user_train[user])), 'test labels:', np.unique(np.asarray(y_user_test[user])))

    for i in range(NUM_USER):
        uname = 'u_{0:05d}'.format(i)

        combined = list(zip(X_user_train[i], y_user_train[i]))
        random.shuffle(combined)
        X_user_train[i][:], y_user_train[i][:] = zip(*combined)

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X_user_train[i], 'y': y_user_train[i]}
        train_data['num_samples'].append(len(y_user_train[i]))

        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X_user_test[i], 'y': y_user_test[i]}
        test_data['num_samples'].append(len(y_user_test[i]))

    with open(train_output, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_output, 'w') as outfile:
        json.dump(test_data, outfile)

if __name__ == "__main__":
    main()
