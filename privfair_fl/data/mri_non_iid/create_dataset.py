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

# This determines the run and random state
RUN_NUMBER = 10

def check_for_black_images(images, stage=""):
    """Check if any image is black (all pixels are zero)."""
    black_image_count = np.sum([np.all(img == 0) for img in images])
    print(f"Number of black images {stage}: {black_image_count} out of {len(images)}")

def check_image_statistics(images, stage=""):
    """Check the min, max, and mean pixel values of the images."""
    all_pixels = np.concatenate([img.flatten() for img in images])
    print(f"{stage} - Min pixel value: {np.min(all_pixels)}, Max pixel value: {np.max(all_pixels)}, Mean: {np.mean(all_pixels)}")

def load_mri_dataset():
    """Load MRI dataset from `X_sample.pkl` and `y_sample.pkl`"""
    print("Loading MRI dataset...")
    with open('raw_data/X_sample.pkl', 'rb') as f:
        X_sample = pickle.load(f)
    with open('raw_data/y_sample.pkl', 'rb') as f:
        y_sample = pickle.load(f)

    print(f"Dataset loaded. Shape of X_sample: {X_sample.shape}, Shape of y_sample: {y_sample.shape}")
    
    check_for_black_images(X_sample, stage="before resizing")
    check_image_statistics(X_sample, stage="before resizing")

    y_sample[y_sample == 2] = 1

    X_sample_scaled = X_sample * 255.0

    X_sample_resized = []
    print(f"Resizing images to {TARGET_IMAGE_SIZE}...")
    for img in tqdm(X_sample_scaled, desc="Resizing images"):
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8)
        img_resized = img_pil.resize(TARGET_IMAGE_SIZE)
        X_sample_resized.append(np.array(img_resized))

    X_sample_resized = np.array(X_sample_resized)
    print(f"Resizing complete. New shape of X_sample: {X_sample_resized.shape}")

    check_for_black_images(X_sample_resized, stage="after resizing")
    check_image_statistics(X_sample_resized, stage="after resizing")

    return X_sample_resized, y_sample

def save_user_data(users_data, output_file):
    """Save user data to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(users_data, f)
    print(f"Data saved to {output_file}.")

import random

def generate_non_iid_dataset(num_users, train_output_dir, test_output_dir, batch_size, run_number):
    X_sample, y_sample = load_mri_dataset()

    print("Normalizing dataset...")
    mu = np.mean(X_sample.astype(np.float32), axis=0)
    sigma = np.std(X_sample.astype(np.float32), axis=0)
    X_sample = (X_sample.astype(np.float32) - mu) / (sigma + 0.001)
    print("Normalization complete.")
    
    check_for_black_images(X_sample, stage="after normalization")
    check_image_statistics(X_sample, stage="after normalization")

    # Split by class
    class_0_indices = np.where(y_sample == 0)[0]
    class_1_indices = np.where(y_sample == 1)[0]

    # Shuffle the class indices for randomness
    np.random.shuffle(class_0_indices)
    np.random.shuffle(class_1_indices)

    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    train_users_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_users_data = {'users': [], 'user_data': {}, 'num_samples': []}

    total_samples = len(X_sample)

    for i in tqdm(range(num_users), desc="Generating Non-IID Dataset"):
        # Assign a random number of samples to each client
        client_sample_size = random.randint(100, total_samples // 2)  # Vary drastically between clients

        # Skew the class distribution for each client
        class_0_proportion = random.uniform(0.1, 0.9)  # Randomly choose proportion of class 0
        num_class_0 = int(client_sample_size * class_0_proportion)
        num_class_1 = client_sample_size - num_class_0

        # Ensure we do not assign 0 samples
        if num_class_0 == 0 or num_class_1 == 0:
            print(f"Skipping client {i} due to insufficient samples.")
            continue

        # Select class 0 and class 1 samples for the client
        class_0_user_indices = class_0_indices[:num_class_0]
        class_1_user_indices = class_1_indices[:num_class_1]

        # Remove the selected indices from the pool to avoid overlap
        class_0_indices = class_0_indices[num_class_0:]
        class_1_indices = class_1_indices[num_class_1:]

        # Combine the data for the client
        user_indices = np.concatenate([class_0_user_indices, class_1_user_indices])
        np.random.shuffle(user_indices)  # Shuffle user data for randomness

        X_user = X_sample[user_indices]
        y_user = y_sample[user_indices]

        # Check if there are enough samples to split into train and test
        if len(X_user) < 2:
            print(f"Skipping client {i} due to insufficient data to split.")
            continue

        # Split into train and test for this client
        X_user_train, X_user_test, y_user_train, y_user_test = train_test_split(
            X_user, y_user, test_size=0.2, random_state=run_number
        )

        uname = f'u_{i:05d}'
        
        # Save training data
        train_users_data['users'].append(uname)
        train_users_data['user_data'][uname] = {'x': X_user_train.tolist(), 'y': y_user_train.tolist()}
        train_users_data['num_samples'].append(len(y_user_train))

        # Save testing data
        test_users_data['users'].append(uname)
        test_users_data['user_data'][uname] = {'x': X_user_test.tolist(), 'y': y_user_test.tolist()}
        test_users_data['num_samples'].append(len(y_user_test))

        # Log the number of samples for each client
        print(f"Client {uname} has {len(y_user_train)} samples for training.")

    save_user_data(train_users_data, os.path.join(train_output_dir, 'mytrain.json'))
    save_user_data(test_users_data, os.path.join(test_output_dir, 'mytest.json'))

    print(f"Non-IID Data generation complete for {num_users} users.")


def main():
    # Create dynamic directories based on run number
    train_output_dir = f'./run{RUN_NUMBER}/data/train'
    test_output_dir = f'./run{RUN_NUMBER}/data/test'
    
    print(f"Generating Non-IID dataset for Run {RUN_NUMBER}...")
    generate_non_iid_dataset(NUM_CLIENTS, train_output_dir, test_output_dir, BATCH_SIZE, RUN_NUMBER)

if __name__ == "__main__":
    main()
