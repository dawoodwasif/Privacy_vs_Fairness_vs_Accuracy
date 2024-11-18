import os
import json
import gzip
import numpy as np
import random

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
    X_train, y_train = load_mnist('raw_data/fashion', kind='train')
    X_test, y_test = load_mnist('raw_data/fashion', kind='t10k')

    # Normalize the data
    mu = np.mean(X_train.astype(np.float32), axis=0)
    sigma = np.std(X_train.astype(np.float32), axis=0)

    X_train = (X_train.astype(np.float32) - mu) / (sigma + 0.001)
    X_test = (X_test.astype(np.float32) - mu) / (sigma + 0.001)

    return X_train.tolist(), y_train.tolist(), X_test.tolist(), y_test.tolist()

def filter_and_relabel_classes(X, y, classes_to_include):
    """Filter the dataset to include only specific classes and relabel them to a 0-based index."""
    filtered_X, filtered_y = [], []
    class_map = {old_label: new_label for new_label, old_label in enumerate(classes_to_include)}

    for i in range(len(y)):
        if y[i] in classes_to_include:
            filtered_X.append(X[i])
            filtered_y.append(class_map[y[i]])

    return filtered_X, filtered_y

def main():
    NUM_USER = 50
    CLASSES_PER_USER = 1  # Each user will get data from only 1 class in non-IID setup

    train_output = "./data/train/mytrain_non_iid.json"
    test_output = "./data/test/mytest_non_iid.json"

    # Create necessary directories
    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    os.makedirs(os.path.dirname(test_output), exist_ok=True)

    # Generate and filter the dataset for only the desired classes
    X_train, y_train, X_test, y_test = generate_dataset()
    classes_to_include = [0, 2, 6]  # Classes for T-shirt/top, pullover, and shirt

    X_train, y_train = filter_and_relabel_classes(X_train, y_train, classes_to_include)
    X_test, y_test = filter_and_relabel_classes(X_test, y_test, classes_to_include)

    # Split data by class for non-IID distribution
    class_data_train = {i: [] for i in range(len(classes_to_include))}
    class_data_test = {i: [] for i in range(len(classes_to_include))}

    for idx, label in enumerate(y_train):
        class_data_train[label].append(X_train[idx])
    for idx, label in enumerate(y_test):
        class_data_test[label].append(X_test[idx])

    # Prepare the data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    np.random.seed(233)
    num_samples = np.random.lognormal(3, 1, NUM_USER) + 10
    num_samples = len(y_train) * num_samples / sum(num_samples)

    # Assign random classes to each user in a non-IID way
    user_classes = [random.choice(range(len(classes_to_include))) for _ in range(NUM_USER)]
    idx_train = {i: 0 for i in range(len(classes_to_include))}
    idx_test = {i: 0 for i in range(len(classes_to_include))}

    for user in range(NUM_USER):
        uname = f'u_{user:05d}'
        class_id = user_classes[user]  # Assign only one class to each user
        train_samples_for_user = int(num_samples[user]) + 1
        test_samples_for_user = int(num_samples[user] / 6) + 1

        # Ensure there is enough data for each user
        if idx_train[class_id] + train_samples_for_user > len(class_data_train[class_id]):
            idx_train[class_id] = 0
        if idx_test[class_id] + test_samples_for_user > len(class_data_test[class_id]):
            idx_test[class_id] = 0

        X_user_train = class_data_train[class_id][idx_train[class_id]:idx_train[class_id] + train_samples_for_user]
        y_user_train = [class_id] * train_samples_for_user
        X_user_test = class_data_test[class_id][idx_test[class_id]:idx_test[class_id] + test_samples_for_user]
        y_user_test = [class_id] * test_samples_for_user

        idx_train[class_id] += train_samples_for_user
        idx_test[class_id] += test_samples_for_user

        # Add to the user data structure
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X_user_train, 'y': y_user_train}
        train_data['num_samples'].append(len(y_user_train))

        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X_user_test, 'y': y_user_test}
        test_data['num_samples'].append(len(y_user_test))

        print(f"User {user}: Class {classes_to_include[class_id]}, num train {len(X_user_train)}, num test {len(X_user_test)}")

    # Write the data to JSON files
    with open(train_output, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_output, 'w') as outfile:
        json.dump(test_data, outfile)

if __name__ == "__main__":
    main()
