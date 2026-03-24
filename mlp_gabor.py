"""
mlp_gabor.py - Training Script for Gabor-Transformed MLP Classifiers

This script trains Multi-Layer Perceptron models on Gabor-transformed EEG-Audio data
using multi-class cross-entropy loss. Unlike the encoder approach (mlp_gabor_encoder.py),
this uses direct classification with CosMLP architecture and cosine similarity matching.

Approach:
    - Data: Gabor-transformed EEG (64 channels) and audio (1 channel)
    - Model: CosMLP - MLP with cosine similarity for matching
    - Loss: Multi-class cross-entropy (5-way classification)
    - Task: Identify which of 5 audio signals matches the EEG

Key Differences from Encoder Approach:
    - Direct classification vs. contrastive learning
    - Uses CosMLP (cosine similarity) instead of Siamese encoder
    - Multi-class CE loss instead of InfoNCE
    - Simpler training but may not learn as rich representations

Hyperparameters:
    - lr: 1e-1 (0.1) - High learning rate for SGD
    - batch_size: 145 - Relatively small batches
    - n_hiddens: 256 - Hidden layer size for feature extraction
    - tau: Temperature for softmax (controls confidence)

Author: Original research implementation
Date: 2 years ago (2024)
"""

import torch
from gabor import GaborDataset, SLP, CosMLP
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random

from nn_helpers import bin_cross_entropy, multi_cross_entropy, accuracy, test_match, init_weights

# Set seed manually for reproducibility across all RNGs
torch.manual_seed(1729)
# torch.mps.manual_seed(1729)
torch.cuda.manual_seed(1729)
torch.cuda.manual_seed_all(1729)
np.random.seed(1729)
random.seed(1729)

# Select device (CUDA GPU or CPU)
# MPS (Apple Silicon) commented out - may have compatibility issues
device = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"  # Apple Silicon GPU support
    # if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# ==================== TRAINING HYPERPARAMETERS ====================

# Training duration
epochs = 100  # Number of complete passes through dataset

# Optimizer parameters
lr = 1e-1  # Learning rate = 0.1 (high for SGD, aggressive learning)
batch_size = 145  # Number of EEG samples per batch (relatively small)

# Model architecture
n_hiddens = 256  # Hidden layer dimension for CosMLP feature extraction
                 # Smaller than encoder approach (which used 512)

# Regularization
cos_regularize = False  # Cosine similarity regularization (not used)
weight_decay = 0.0  # L2 regularization strength (no weight decay)

# Loss function parameter
tau = 1  # Temperature parameter for softmax in multi_cross_entropy
         # tau=1 is standard softmax (no temperature scaling)

# For large dataset
train_mat_file = '../train_gabor_results.mat'
test_mat_file = '../test_gabor_results.mat'
mat_file = '../gabor_results.mat'
n_channels = 1

# Load the training/testing data (gabor_results.mat)
training_data = GaborDataset(train_mat_file, train=True, data_prop=0.05)
testing_data = GaborDataset(test_mat_file, train=False, data_prop=0.05)

train_dataloader = DataLoader(training_data, batch_size, shuffle=False)
test_dataloader = DataLoader(testing_data, batch_size=40, shuffle=False)

# Our MLP model
net = CosMLP(n_hiddens=n_hiddens, M=16, N=64, n_channels=n_channels).to(device)
#net.apply(init_weights)
# net = DLP(n_hiddens0=64, n_hiddens1=64, M=16, N=64).to(device)

# net.load_state_dict(torch.load('weightsTenCycle2e-15e-1.pth'))

# Select loss function and optimizer
criterion = multi_cross_entropy
optimizer = optim.SGD(net.parameters(), lr, weight_decay=weight_decay)
#scheduler = optim.lr_scheduler.CyclicLR(optimizer, 5e-3, 5e-2)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# Keep track of train/test loss/accuracy for model evaluation
train_losses, test_losses = [], []
train_accs, test_accs = [], []

# Keep track for matching out of 5 EEG/Audio pairs
train_match_accs = []
test_match_accs = []

# Keep trach of best matching accuracies
max_match_acc = 0.20

# Training loop
for epoch in tqdm(range(epochs)): 

    train_loss, test_loss = 0.0, 0.0
    train_acc, test_acc = 0.0, 0.0
    train_match_acc, test_match_acc = 0.0, 0.0
    trials = 0

    for inputs, labels in train_dataloader:

        # get the inputs; data is a list of [inputs, labels]
        inputs = [input.to(device) for input in inputs]
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs.reshape(1, -1), labels, regularize=cos_regularize).mean()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy(outputs, labels).item()

        train_match_acc += test_match(outputs, labels).item()

        trials += 1
        # scheduler.step()

    train_losses.append(train_loss / trials)
    train_accs.append(train_acc / trials)
    train_match_accs.append(train_match_acc / trials)
    
    trials = 0

    # Compute testing loss/accuracy
    for inputs, labels in test_dataloader:
        with torch.no_grad():
            inputs = [input.to(device) for input in inputs]
            labels = labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs.reshape(1, -1), labels, regularize=cos_regularize).mean()

            test_loss += loss.item()
            test_acc += accuracy(outputs, labels).item()

            test_match_acc += test_match(outputs, labels).item()

            trials +=1

    test_losses.append(test_loss / trials)
    test_accs.append(test_acc / trials)
    test_match_accs.append(test_match_acc / trials)

    """
    if test_match_acc / trials > 0.5:
        print(f"Outputs: {outputs}")
        print(f"Loss: {loss}")
        print(f"Trials: {trials}")
        print(f"Test_match_acc: {test_match_acc}")
    """

    #scheduler.step(test_loss / trials)

    # Save good weights
    if test_match_acc / trials > max_match_acc:
        torch.save(net.state_dict(), 'max_weights.pth')
        max_match_acc = test_match_acc / trials

print(f"Finished Training. Maximum accuracy: {max_match_acc}")

# Save the model to 'weights.pth'
torch.save(net.state_dict(), 'weights.pth')

# Save loss/accuracy plot
plt.plot(train_losses, c='crimson', label='train loss')
plt.plot(train_accs, '--', c='crimson', label='train acc')
plt.plot(test_losses, c='navy', label='test loss')
plt.plot(test_accs, '--', c='navy', label='test acc')
plt.plot(train_match_accs, '-.', c='orange', label='train match acc')
plt.plot(test_match_accs, '-.', c='green', label='test match acc')
plt.xlabel('epoch')
plt.grid()
plt.xlim(0, epochs-1)
#plt.ylim(0, 1)
plt.title(f'Loss/Accuracy Gabor - lr={lr}, n_hiddens={n_hiddens}, Cosine')
plt.legend()

# Save figure with name 'LossAcc_-_epochs_{epochs},lr_{lr},n_hiddens_{n_hiddens}.png'
plt.savefig(f'LossAccTenStdCos{n_hiddens}-{lr}-{epochs}-2.png')
