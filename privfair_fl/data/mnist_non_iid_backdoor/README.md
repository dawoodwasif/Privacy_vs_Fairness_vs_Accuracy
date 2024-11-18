# MNIST Dataset Setup and Partitioning for Federated Learning

This guide provides instructions for downloading, setting up, and partitioning the MNIST dataset for federated learning experiments.

## Dataset Overview

MNIST is a large dataset of handwritten digits commonly used for training various image processing systems. It includes 60,000 training images and 10,000 testing images, each of 28x28 pixels in grayscale.

## Steps to Prepare the Dataset

### 1. **Download the MNIST Dataset**

Download the MNIST dataset files from the following link:

```bash
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
```

### 2. Place the Files in the Correct Directory
After downloading, place the .gz files in the following directory:

```bash
raw_data/mnist/
```

The directory structure should look like this:

```bash
raw_data/mnist/train-images-idx3-ubyte.gz
raw_data/mnist/train-labels-idx1-ubyte.gz
raw_data/mnist/t10k-images-idx3-ubyte.gz
raw_data/mnist/t10k-labels-idx1-ubyte.gz
```

### 3. Partition the Dataset Across Devices
Next, run the script create_mnist_dataset.py to partition the MNIST dataset across 500 devices (users):

```bash
python create_mnist_dataset.py
```

This will create the following output files:

```
data/train/mytrain.json: Training data for all users.
data/test/mytest.json: Testing data for all users.
```

#### Dataset Partitioning Details
The create_mnist_dataset.py script partitions the dataset as follows:

- Normalization: The training data is normalized by subtracting the mean and dividing by the standard deviation.

- User Partitioning: The dataset is partitioned across 500 users. Each user gets a varying number of samples based on a log-normal distribution.

- JSON Output: The data for each user is stored in JSON files, with separate entries for training and testing data.