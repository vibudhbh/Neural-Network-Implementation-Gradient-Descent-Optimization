# Import required libraries
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

import torch.multiprocessing as mp

# Ensure proper multiprocessing method is used
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

"""
Define the network architecture with batch normalization after each convolution and ReLU.
"""


class CIFAR_CNN_BatchNorm(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Part 1 (32x32) ---
        self.conv1_1 = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(9)
        self.conv1_2 = nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(9)
        # MaxPool will reduce to 16x16

        # --- Part 2 (16x16) ---
        self.conv2_1 = nn.Conv2d(9, 18, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(18)
        self.conv2_2 = nn.Conv2d(18, 18, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(18)
        # MaxPool will reduce to 8x8

        # --- Part 3 (8x8) ---
        self.conv3_1 = nn.Conv2d(18, 36, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(36)
        self.conv3_2 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(36)
        # MaxPool will reduce to 4x4

        # --- Part 4 (Dense layers) ---
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(36 * 4 * 4, 100)  # 36 feature maps of size 4x4 = 576 inputs
        self.bn_fc1 = nn.BatchNorm1d(100)  # BatchNorm for fully connected layer
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 10)  # 10 classes

    def forward(self, x):
        # --- Part 1 ---
        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.bn1_1(x)

        x = self.conv1_2(x)
        x = F.relu(x)
        x = self.bn1_2(x)

        x = F.max_pool2d(x, 2)  # 32x32 -> 16x16

        # --- Part 2 ---
        x = self.conv2_1(x)
        x = F.relu(x)
        x = self.bn2_1(x)

        x = self.conv2_2(x)
        x = F.relu(x)
        x = self.bn2_2(x)

        x = F.max_pool2d(x, 2)  # 16x16 -> 8x8

        # --- Part 3 ---
        x = self.conv3_1(x)
        x = F.relu(x)
        x = self.bn3_1(x)

        x = self.conv3_2(x)
        x = F.relu(x)
        x = self.bn3_2(x)

        x = F.max_pool2d(x, 2)  # 8x8 -> 4x4

        # --- Part 4 ---
        x = self.flatten(x)  # Flatten the 36x4x4 feature maps
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn_fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


"""
This function calculates the number of trainable parameters in the network.
"""


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


"""
This function calculates the accuracy.
"""


def accuracy(predictions, labels):
    predicted_labels = torch.argmax(predictions, dim=1)
    return (predicted_labels == labels).float().mean(), len(predictions)


"""
Function to compute loss and accuracy on a batch.
"""


def loss_batch(model, loss_func, xb, yb, optimizer=None):
    # Forward pass and compute the loss
    predictions = model(xb)
    loss = loss_func(predictions, yb)

    # Calculate accuracy
    acc, nums = accuracy(predictions, yb)

    # Backward pass and optimization step if optimizer is provided
    if optimizer is not None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item(), acc.item(), nums


"""
Function to evaluate the model on validation or test sets.
"""


def evaluate(model, loss_func, data_loader):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_samples = 0

    with torch.no_grad():
        for xb, yb in data_loader:
            loss, acc, nums = loss_batch(model, loss_func, xb, yb)
            total_loss += loss * nums
            total_acc += acc * nums
            total_samples += nums

    return total_loss / total_samples, total_acc / total_samples


"""
Function to train the network.
"""


def fit(epochs, model, loss_func, optimizer, train_dl, valid_dl):
    # Lists to store metrics for plotting
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0

        for xb, yb in train_dl:
            loss, acc, nums = loss_batch(model, loss_func, xb, yb, optimizer)
            train_loss += loss * nums
            train_acc += acc * nums
            train_samples += nums

        train_loss = train_loss / train_samples
        train_acc = train_acc / train_samples
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Evaluation phase
        val_loss, val_acc = evaluate(model, loss_func, valid_dl)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Plot training curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')

    plt.tight_layout()
    plt.savefig('training_curves_batchnorm.png')

    return train_losses, train_accs, val_losses, val_accs


"""
This is the main function, responsible for data loading and network execution.
"""


def main():
    # Load the dataset from torchvision and separate the test data
    dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
    test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())

    # Split the training data into training and validation sets
    val_size = 5000
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Training set size: {len(train_ds)}, Validation set size: {len(val_ds)}")

    # Define hyperparameters
    epochs = 50
    batch_size = 128
    learning_rate = 0.01
    momentum = 0.9
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}, Momentum: {momentum}")

    # Create data loaders for efficient data streaming
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, num_workers=1)

    # Instantiate the model with batch normalization
    model = CIFAR_CNN_BatchNorm()

    # Check the number of parameters
    param_count = count_parameters(model)
    print(f"Number of parameters: {param_count}")

    # Instantiate the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Declare the loss function
    loss_function = F.cross_entropy

    # Train the model
    train_losses, train_accs, val_losses, val_accs = fit(epochs, model, loss_function, optimizer, train_loader,
                                                         val_loader)

    # Calculate loss and accuracy on the test set and print
    test_loss, test_acc = evaluate(model, loss_function, test_loader)
    print(f"Final results on the test set - loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")

    # Save results to bonus_output.txt
    with open('bonus_output.txt', 'w') as f:
        f.write(f"Model with Batch Normalization\n")
        f.write(f"Model Parameters: {param_count}\n")
        f.write(f"Final Test Loss: {test_loss:.4f}\n")
        f.write(f"Final Test Accuracy: {test_acc:.4f}\n\n")
        f.write("Training History:\n")
        for epoch in range(epochs):
            f.write(
                f"Epoch {epoch}: Train Loss: {train_losses[epoch]:.4f}, Train Acc: {train_accs[epoch]:.4f}, Val Loss: {val_losses[epoch]:.4f}, Val Acc: {val_accs[epoch]:.4f}\n")


if __name__ == '__main__':
    main()