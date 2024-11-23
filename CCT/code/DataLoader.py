import os
import pickle
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from Configure import configs
from ImageUtils import CIFAR10Policy

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset and apply augmentations.
    """
    normalize = [transforms.Normalize(mean=configs["mean"], std=configs["std"])]
    augmentations = []
    
    # create augmentations
    augmentations += [
        CIFAR10Policy()
    ]
    augmentations += [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        *normalize,
    ]
    augmentations = transforms.Compose(augmentations)
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=augmentations)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            *normalize,
        ]))

    train_loader = DataLoader(train_dataset, batch_size=configs["batch_size"],
                              shuffle=True, num_workers=configs["workers"], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=configs["batch_size"],
                             shuffle=False, num_workers=configs["workers"],
                             pin_memory=True)

    return train_loader, test_loader


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 3072].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    x_test = np.load(data_dir + '/private_test_images_2024.npy')
    ### END CODE HERE

    return x_test


def train_valid_split(data_loader, train_ratio=0.8):

    train_size = int(train_ratio * len(data_loader.dataset))
    valid_size = len(data_loader.dataset) - train_size

    # Shuffle the indices
    indices = torch.randperm(len(data_loader.dataset)).tolist()

    # Split the indices
    train_indices, valid_indices = indices[:train_size], indices[train_size:]

    train_dataset = Subset(data_loader.dataset, train_indices)
    valid_dataset = Subset(data_loader.dataset, valid_indices)

    # Create new DataLoaders for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=data_loader.batch_size,
                              shuffle=True, num_workers=data_loader.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=data_loader.batch_size,
                              shuffle=False, num_workers=data_loader.num_workers)

    return train_loader, valid_loader

