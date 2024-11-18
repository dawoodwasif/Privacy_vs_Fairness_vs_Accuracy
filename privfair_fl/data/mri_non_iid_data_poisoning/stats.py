# import os
# import json
# from collections import Counter

# def load_json(file_path):
#     """Load JSON file."""
#     if os.path.exists(file_path):
#         with open(file_path, 'r') as f:
#             data = json.load(f)
#         return data
#     else:
#         raise FileNotFoundError(f"{file_path} not found!")

# def print_stats(data, dataset_type):
#     """Print statistics for the dataset (train or test)."""
#     print(f"\n--- {dataset_type.upper()} DATASET STATISTICS ---")
#     num_clients = len(data['users'])
#     print(f"Number of clients: {num_clients}")

#     total_samples = 0
#     all_labels = []

#     for user in data['users']:
#         client_labels = data['user_data'][user]['y']
#         num_samples = len(client_labels)
#         total_samples += num_samples
#         all_labels.extend(client_labels)
        
#         # Label distribution for the individual client
#         client_label_distribution = Counter(client_labels)
#         print(f"\nClient {user} has {num_samples} samples.")
#         print(f"Client {user} label distribution: {dict(client_label_distribution)} (0 = Non-Alzheimer, 1 = Alzheimer)")

#     # Print overall statistics
#     label_distribution = Counter(all_labels)
#     print(f"\nTotal number of samples: {total_samples}")
#     print(f"Overall label distribution: {dict(label_distribution)} (0 = Non-Alzheimer, 1 = Alzheimer)")

# def main():
#     # Paths to the train and test JSON files
#     train_file = './data/train/mytrain.json'
#     test_file = './data/test/mytest.json'
    
#     # Load and print stats for training data
#     try:
#         train_data = load_json(train_file)
#         print_stats(train_data, dataset_type="train")
#     except FileNotFoundError as e:
#         print(e)
    
#     # Load and print stats for testing data
#     try:
#         test_data = load_json(test_file)
#         print_stats(test_data, dataset_type="test")
#     except FileNotFoundError as e:
#         print(e)

# if __name__ == "__main__":
#     main()
import os
import json
from collections import Counter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_json(file_path):
    """Load JSON file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    else:
        raise FileNotFoundError(f"{file_path} not found!")

def save_first_n_images(images, folder, n=5):
    """Save first N images to disk and check if all are black."""
    os.makedirs(folder, exist_ok=True)
    
    all_black = True
    for i, img_data in enumerate(images[:n]):
        img = Image.fromarray(np.array(img_data).astype(np.uint8))

        # Save the image
        img.save(f"{folder}/image_{i}.jpg")

        # Check if the image is all black (i.e., all pixel values are 0)
        if np.any(np.array(img) != 0):
            all_black = False

    if all_black:
        print("All images are black.")
    else:
        print("Not all images are black.")

def print_stats_and_check_images(data, dataset_type):
    """Print statistics for the dataset (train or test) and check if images are all black."""
    print(f"\n--- {dataset_type.upper()} DATASET STATISTICS ---")
    num_clients = len(data['users'])
    print(f"Number of clients: {num_clients}")

    total_samples = 0
    all_labels = []

    for user in data['users']:
        client_labels = data['user_data'][user]['y']
        client_images = data['user_data'][user]['x']  # Assuming the image data is in 'x'
        num_samples = len(client_labels)
        total_samples += num_samples
        all_labels.extend(client_labels)

        # Label distribution for the individual client
        client_label_distribution = Counter(client_labels)
        print(f"\nClient {user} has {num_samples} samples.")
        print(f"Client {user} label distribution: {dict(client_label_distribution)} (0 = Non-Alzheimer, 1 = Alzheimer)")

        # Save and check the first N images for each client
        save_first_n_images(client_images, folder=f"./{dataset_type}_images/{user}", n=5)

    # Print overall statistics
    label_distribution = Counter(all_labels)
    print(f"\nTotal number of samples: {total_samples}")
    print(f"Overall label distribution: {dict(label_distribution)} (0 = Non-Alzheimer, 1 = Alzheimer)")

def main():
    # Paths to the train and test JSON files
    train_file = './data/train/mytrain.json'
    test_file = './data/test/mytest.json'
    
    # Load and print stats for training data
    try:
        train_data = load_json(train_file)
        print_stats_and_check_images(train_data, dataset_type="train")
    except FileNotFoundError as e:
        print(e)
    
    # Load and print stats for testing data
    try:
        test_data = load_json(test_file)
        print_stats_and_check_images(test_data, dataset_type="test")
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
