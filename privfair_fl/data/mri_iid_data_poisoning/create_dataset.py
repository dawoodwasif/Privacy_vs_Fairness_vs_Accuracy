import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

# Constants
NUM_CLIENTS = 5
BATCH_SIZE = 100
TARGET_IMAGE_SIZE = (64, 64)
RUN_NUMBER_START = 1
RUN_NUMBER_END = 10  # Run multiple times for different seeds

def check_for_black_images(images, stage=""):
    black_image_count = np.sum([np.all(img == 0) for img in images])
    print(f"Number of black images {stage}: {black_image_count} out of {len(images)}")

def check_image_statistics(images, stage=""):
    all_pixels = np.concatenate([img.flatten() for img in images])
    print(f"{stage} - Min pixel value: {np.min(all_pixels)}, Max: {np.max(all_pixels)}, Mean: {np.mean(all_pixels)}")

def load_mri_dataset():
    print("Loading MRI dataset...")
    with open('raw_data/X_sample.pkl', 'rb') as f:
        X_sample = pickle.load(f)
    with open('raw_data/y_sample.pkl', 'rb') as f:
        y_sample = pickle.load(f)
    
    y_sample[y_sample == 2] = 1
    X_sample_scaled = X_sample * 255.0

    X_sample_resized = []
    for img in tqdm(X_sample_scaled, desc="Resizing images"):
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8)
        img_resized = img_pil.resize(TARGET_IMAGE_SIZE)
        X_sample_resized.append(np.array(img_resized))

    return np.array(X_sample_resized), y_sample

def apply_data_poisoning(y_train, poison_rate=0.1):
    num_samples = int(len(y_train) * poison_rate)
    poisoned_indices = np.random.choice(len(y_train), num_samples, replace=False)
    
    for idx in poisoned_indices:
        y_train[idx] = 1 - y_train[idx]  # Flip between classes 0 and 1

    return y_train

def generate_iid_dataset(num_users, train_output_dir, test_output_dir, batch_size, run_number):
    X_sample, y_sample = load_mri_dataset()
    
    mu = np.mean(X_sample.astype(np.float32), axis=0)
    sigma = np.std(X_sample.astype(np.float32), axis=0)
    X_sample = (X_sample.astype(np.float32) - mu) / (sigma + 0.001)

    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=run_number)
    y_train = apply_data_poisoning(y_train, poison_rate=0.1)

    num_items_per_user_train = len(X_train) // num_users
    num_items_per_user_test = len(X_test) // num_users

    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    train_users_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_users_data = {'users': [], 'user_data': {}, 'num_samples': []}

    for i in tqdm(range(num_users), desc="Generating IID Dataset"):
        train_user_idxs = np.arange(i * num_items_per_user_train, (i + 1) * num_items_per_user_train)
        X_user_train = X_train[train_user_idxs].tolist()
        y_user_train = y_train[train_user_idxs].tolist()

        test_user_idxs = np.arange(i * num_items_per_user_test, (i + 1) * num_items_per_user_test)
        X_user_test = X_test[test_user_idxs].tolist()
        y_user_test = y_test[test_user_idxs].tolist()

        uname = f'u_{i:05d}'
        train_users_data['users'].append(uname)
        train_users_data['user_data'][uname] = {'x': X_user_train, 'y': y_user_train}
        train_users_data['num_samples'].append(len(y_user_train))

        test_users_data['users'].append(uname)
        test_users_data['user_data'][uname] = {'x': X_user_test, 'y': y_user_test}
        test_users_data['num_samples'].append(len(y_user_test))

    with open(os.path.join(train_output_dir, 'mytrain.json'), 'w') as outfile:
        json.dump(train_users_data, outfile)
    with open(os.path.join(test_output_dir, 'mytest.json'), 'w') as outfile:
        json.dump(test_users_data, outfile)

def main():
    for run_number in range(RUN_NUMBER_START, RUN_NUMBER_END + 1):
        train_output_dir = f'./run{run_number}/data/train'
        test_output_dir = f'./run{run_number}/data/test'
        print(f"Generating IID poisoned dataset for Run {run_number}...")
        generate_iid_dataset(NUM_CLIENTS, train_output_dir, test_output_dir, BATCH_SIZE, run_number)

if __name__ == "__main__":
    main()
