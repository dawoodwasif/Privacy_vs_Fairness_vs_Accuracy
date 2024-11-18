import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

# Constants
NUM_CLIENTS = 5
BATCH_SIZE = 100
TARGET_IMAGE_SIZE = (64, 64)
RUN_NUMBER_START = 1
RUN_NUMBER_END = 10
BACKDOOR_CLIENTS = [0]  # Clients with backdoor data
TRIGGER_SHAPE = (5, 5)
TRIGGER_LABEL = 1

def add_trigger(image):
    """Adds a trigger (small square) to the bottom-right corner of the image."""
    # Convert image to uint8 type to be compatible with PIL
    image = (image * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    trigger_start = (img.width - TRIGGER_SHAPE[0], img.height - TRIGGER_SHAPE[1])
    trigger_end = (img.width, img.height)
    draw.rectangle([trigger_start, trigger_end], fill="white")
    return np.array(img)


def load_mri_dataset():
    # Load and preprocess the MRI dataset
    with open('raw_data/X_sample.pkl', 'rb') as f:
        X_sample = pickle.load(f)
    with open('raw_data/y_sample.pkl', 'rb') as f:
        y_sample = pickle.load(f)
    return X_sample, y_sample

def save_user_data(users_data, output_file):
    """Save user data to JSON file."""
    # Convert ndarray to list
    for user in users_data['user_data']:
        users_data['user_data'][user]['x'] = [x.tolist() if isinstance(x, np.ndarray) else x for x in users_data['user_data'][user]['x']]
        users_data['user_data'][user]['y'] = [y.tolist() if isinstance(y, np.ndarray) else y for y in users_data['user_data'][user]['y']]
    
    with open(output_file, 'w') as f:
        json.dump(users_data, f)
    print(f"Data saved to {output_file}.")


def generate_iid_dataset(num_users, train_output_dir, test_output_dir, batch_size, run_number):
    X_sample, y_sample = load_mri_dataset()
    X_sample = (X_sample - np.mean(X_sample, axis=0)) / (np.std(X_sample, axis=0) + 0.001)
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=run_number)
    
    train_users_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_users_data = {'users': [], 'user_data': {}, 'num_samples': []}

    num_items_per_user_train = len(X_train) // num_users
    num_items_per_user_test = len(X_test) // num_users

    for i in tqdm(range(num_users), desc="Generating IID Dataset"):
        train_user_idxs = np.arange(i * num_items_per_user_train, (i + 1) * num_items_per_user_train)
        X_user_train = X_train[train_user_idxs].tolist()
        y_user_train = y_train[train_user_idxs].tolist()

        # Apply backdoor attack for specified clients
        if i in BACKDOOR_CLIENTS:
            for j in range(len(X_user_train) // 10):  # Add trigger to 10% of images
                X_user_train[j] = add_trigger(np.array(X_user_train[j]))
                y_user_train[j] = TRIGGER_LABEL  # Set target label for backdoored data

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

    save_user_data(train_users_data, os.path.join(train_output_dir, 'mytrain.json'))
    save_user_data(test_users_data, os.path.join(test_output_dir, 'mytest.json'))

def main():
    for run in range(RUN_NUMBER_START, RUN_NUMBER_END + 1):
        train_output_dir = f'./run{run}/data/train'
        test_output_dir = f'./run{run}/data/test'
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)
        generate_iid_dataset(NUM_CLIENTS, train_output_dir, test_output_dir, BATCH_SIZE, run)

if __name__ == "__main__":
    main()
