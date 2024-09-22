import os
import json
from collections import Counter

def load_json(file_path):
    """Load JSON file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    else:
        raise FileNotFoundError(f"{file_path} not found!")

def print_stats(data, dataset_type):
    """Print statistics for the dataset (train or test)."""
    print(f"\n--- {dataset_type.upper()} DATASET STATISTICS ---")
    num_clients = len(data['users'])
    print(f"Number of clients: {num_clients}")

    total_samples = 0
    all_labels = []

    for user in data['users']:
        num_samples = len(data['user_data'][user]['y'])
        total_samples += num_samples
        all_labels.extend(data['user_data'][user]['y'])
        print(f"Client {user} has {num_samples} samples.")

    # Print overall statistics
    label_distribution = Counter(all_labels)
    print(f"\nTotal number of samples: {total_samples}")
    print(f"Label distribution: {dict(label_distribution)} (0 = Non-Alzheimer, 1 = Alzheimer)")

def main():
    # Paths to the train and test JSON files
    train_file = './run2/data/train/mytrain.json'
    test_file = './run2/data/test/mytest.json'
    
    # Load and print stats for training data
    try:
        train_data = load_json(train_file)
        print_stats(train_data, dataset_type="train")
    except FileNotFoundError as e:
        print(e)
    
    # Load and print stats for testing data
    try:
        test_data = load_json(test_file)
        print_stats(test_data, dataset_type="test")
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
