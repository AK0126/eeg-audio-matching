"""
GaborMLP.py - Manual Neural Network Implementation for Gabor-Transformed EEG-Audio Matching

This is an exploratory/educational script that implements neural network training from scratch
without using PyTorch's nn.Module abstraction. It manually defines forward pass, loss computation,
and gradient descent for learning on Gabor-transformed data.

Purpose:
    - Educational: Demonstrates how neural networks work under the hood
    - Exploratory: Quick experimentation with different architectures
    - Research: Understanding gradient flow and parameter updates manually

Implementation:
    - Manual parameter initialization (W1, b1, W2, b2)
    - Manual forward pass: X → tanh(XW1+b1) → sigmoid(HW2+b2)
    - Manual loss: Binary cross-entropy implemented from scratch
    - Manual optimizer: SGD update rule (param -= lr * grad)
    - No nn.Module, no optim.SGD - all manual

Architecture:
    Input: Concatenated EEG + audio, shape (batch, 66560)
           - EEG: 64 channels × 16 time × 64 freq = 65,536
           - Audio: 1 channel × 16 time × 64 freq = 1,024
           - Total: 66,560 features
    Hidden: 512 units with tanh activation
    Output: 1 unit with sigmoid (binary classification)

Comparison to Other Scripts:
    - mlp_gabor.py: Uses PyTorch nn.Module (CosMLP)
    - mlp_gabor_encoder.py: Uses nn.Module (GaborEncoder)
    - This file: Everything manual for educational purposes

Author: Original research implementation
Date: 2 years ago (2024)
"""

import torch
from torch.utils.data import DataLoader
from torch import relu, sigmoid, tanh
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io as sio
import numpy as np
from gabor import GaborDataset

# Load training and testing data using PyTorch DataLoader
training_data = GaborDataset(mat_file='../gabor_results.mat', train=True)
testing_data = GaborDataset(mat_file='../gabor_results.mat', train=False)

train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=4, shuffle=True)

for X, y in train_dataloader:
    print(f"Shape of batch X:\t{len(X)}")

    eeg = X[0]
    audio = X[1]
    print(f"Shape of EEG:\t{eeg.shape}")
    print(f"Shape of Audio:\t{audio.shape}")

    # torch.t - Transpose
    # Plot EEG channel 9 from the first set of features of the batch
    # plt.plot(torch.t(eeg[0])[:,9])
    break

"""
We are given signals of dimension 320 by 69 (64 EEG plus 5 Audio), and we want to classify them as one of 5 classes.
At each step, we will be applying a linear transformation (W's) followed by an additive bias term (b's).
We then apply a non-linearity (relu or softmax).
"""

n_inputs = 66560 #D
n_hiddens = 512 #J
n_outputs = 1 #K

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize parameters and move to the correct device
W1 = torch.nn.Parameter(torch.randn(size=(n_inputs, n_hiddens), device=device, dtype=torch.float32))
b1 = torch.nn.Parameter(torch.randn(n_hiddens, device=device, dtype=torch.float32))
W2 = torch.nn.Parameter(torch.randn(size=(n_hiddens, n_outputs), device=device, dtype=torch.float32))
b2 = torch.nn.Parameter(torch.randn(n_outputs, device=device, dtype=torch.float32))

params = [W1, b1, W2, b2]

def net(X):
    """
    Implements the forward pass of our neural network. 
    
    :param X: a batch of signals of shape (batch_size, 320, 69).
    :return: a 2D numpy array of shape (batch_size, 5) where each row is a probability vector of length 5.
    """

    # Transpose and flatten EEG signals
    eeg = torch.transpose(X[0], dim0=1, dim1=2)
    eeg = torch.flatten(eeg, start_dim=1)

    # Transpose and flatten audio signals
    # audio = torch.transpose(X[1], dim0=1, dim1=2)
    audio = torch.flatten(X[1], start_dim=1)

    # Concatenate signals
    X = torch.cat((eeg, audio), dim=1)
    # Parameters are float32
    X = X.to(torch.float32)
    H = tanh(X @ W1 + b1)
    O = sigmoid(H @ W2 + b2)
    return O

## Testing neural net
# for X, _ in train_dataloader:
#     print(X[1].shape)
#     print(f"{X[0].shape}")
#     print(f"{net(X).shape}")
#     print(f"{net(X)}")
#     break

def bin_cross_entropy(y_hat, y):
    """
    Implement cross entropy loss for two inputs
    :param y_hat: a 2D numpy array of shape (batch_size,) where each entry is a probability of match.
    :param y: a 1D numpy array of shape (batch_size) where each element is a label (0 or 1).
  
    :return: a 1D numpy array of shape (batch_size) representing the cross entropy losses.
    """
    # Avoid nan values
    epsilon = 1e-7
    y_clamped = torch.clamp(y_hat, epsilon, 1 - epsilon)
    loss = -1 * y * torch.log(y_clamped) - (1 - y) * torch.log(1 - y_clamped)
    return loss


def sgd(params, lr=0.1):
    """
    Implements stochastic gradient descent. For each param, subtract the gradient times the learning rate.
    The gradient for each param is stored in param.grad. After updating each param, set its gradient to zero.
    
    :param params: a list of parameters to update.
    :param lr: the learning rate.
    
    :return: None
    """
    with torch.no_grad():
      for param in params:
        #print(param.grad)
        param -= lr*param.grad
        param.grad.zero_()

torch.log(torch.tensor(1 - 1e-9))

epochs = 10
batch_size = 120
lr = 10
train_iter = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(testing_data, batch_size=batch_size, shuffle=True)


def accuracy(y_hat, y):
  with torch.no_grad():
    y_labels = torch.round(y_hat)
    correct = y_labels == y
    return correct.sum() / correct.numel()


def train(net, params, train_iter, test_iter, loss, updater):

  # Ensure parameters are on the correct device
  for param in params:
    param.to(device)

  train_losses, train_accs = [], []
  test_losses, test_accs = [], []
  
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = 0.0, 0.0
    trials = 0

    for X, y in train_iter:
      X = [x.to(device) for x in X]  # Move batch to GPU
      y = y.to(device)  # Move labels to GPU
      
      trials += 1
      y_hat = net(X)
      l = loss(y_hat, y).mean()
      acc = accuracy(y_hat, y)

      l.backward()
      updater(params, lr)

      train_loss += l.item()
      train_acc += acc.item()

    train_losses.append(train_loss / trials)
    train_accs.append(train_acc / trials)

    test_loss, test_acc = 0.0, 0.0
    trials = 0

    y_pred = []
    y_true = []

    with torch.no_grad():
      for X, y in test_iter:
        X = [x.to(device) for x in X]  # Move batch to GPU
        y = y.to(device)

        trials += 1

        y_hat = net(X)

        # Recording for confusion matrix
        if epoch == epochs-1:
          y_pred += torch.round(y_hat).cpu().numpy().tolist()
          y_true += y.cpu().numpy().tolist()
      
        l = loss(y_hat, y).mean()
        acc = accuracy(y_hat, y)

        test_loss += l.item()
        test_acc += acc.item()
    
    test_losses.append(test_loss / trials)
    test_accs.append(test_acc / trials)
  
  return train_losses, train_accs, test_losses, test_accs, y_pred, y_true

"""
We are given signals of dimension 320 by 69 (64 EEG plus 5 Audio), and we want to classify them as one of 5 classes.
At each step, we will be applying a linear transformation (W's) followed by an additive bias term (b's).
We then apply a non-linearity (relu or softmax).
"""

# n_inputs = 66560 #D
# n_hiddens = 512 #J
# n_outputs = 1 #K

# W1 = torch.nn.Parameter(torch.randn(size = (n_inputs, n_hiddens))) # matrix applying linear transformation
# b1 = torch.nn.Parameter(torch.zeros(size = (1, n_hiddens))) # additive bias term
# W2 = torch.nn.Parameter(torch.randn(size = (n_hiddens, n_outputs))) # matrix applying linear transformation
# b2 = torch.nn.Parameter(torch.zeros(size = (1, n_outputs))) # additive bias term

params = [W1, b1, W2, b2]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_losses, train_accs, test_losses, test_accs, y_pred, y_true = train(net, params, train_iter, test_iter, loss=bin_cross_entropy, updater=sgd)

#plt.plot(train_losses, c='crimson', label='train loss')
plt.plot(train_accs, '--', c='crimson', label='train acc')
#plt.plot(test_losses, c='navy', label='test loss')
plt.plot(test_accs, '--', c='navy', label='test acc')

plt.xlabel('epoch')
plt.grid()
plt.xlim(0, epochs-1)
# plt.ylim(0, 1)
plt.legend()
plt.savefig('figs/accuracy')
# plt.show()

plt.plot(train_losses, c='crimson', label='train loss')
plt.plot(test_losses, c='navy', label='test loss')
plt.xlabel('epoch')
plt.grid()
plt.xlim(0, epochs-1)
plt.legend()
plt.savefig('figs/loss')
# plt.show()

print(y_pred)

all_clips = sio.loadmat("../gabor_results.mat")['results']
results = all_clips['results'][0, 0]
ans = all_clips['ans'][0, 0][0]

test_clips = [[torch.tensor(np.array(results[i,0:64])).unsqueeze(0),
                    torch.tensor(np.array(results[i,64+j])).unsqueeze(0)] for i in range(results.shape[0]) for j in range(5)]
for i in range(len(test_clips)):
    test_clips[i] = [clip.to(device) for clip in test_clips[i]]
# test_clips = [clip.to(device) for clip in test_clips]

correct = []
for i in range(len(test_clips) // 5):
    output = []
    for j in range(5):
        output.append(net(test_clips[i*5+j]))
    # print(output)
    correct.append(output.index(max(output)) == ans[i])

# print(correct)
print(sum(correct) / len(correct)) # 0.5