"""
mlp_raw.py - Training Script for Raw Signal MLP Classifiers

This script trains Multi-Layer Perceptron models on raw (non-transformed) EEG-Audio data.
Unlike the Gabor approach, this uses time-domain signals directly without any frequency
transformation, serving as a baseline for comparison.

Approach:
    - Data: Raw time-domain EEG (320 timesteps × 64 channels) and audio (320 timesteps)
    - Model: RawSLP - Single-layer perceptron with massive hidden layer
    - Loss: Binary cross-entropy (match/no-match classification)
    - Task: Binary classification for each EEG-audio pair

Key Characteristics:
    - No feature transformation: Uses raw signals as-is
    - Very large hidden layer: n_hiddens=8192 (to compensate for lack of features)
    - Baseline model: Helps evaluate whether Gabor transform adds value
    - Simple architecture: Flatten → Linear(8192) → ReLU → Linear(1) → Sigmoid

Comparison to Gabor Approach:
    - Raw signals: 320×64 = 20,480 input features (larger than Gabor's 16×64 = 1,024)
    - Needs larger hidden layer to capture patterns without frequency features
    - Direct time-domain patterns vs. time-frequency patterns
    - Serves as ablation study: How much does Gabor transformation help?

Hyperparameters:
    - n_hiddens: 8192 - Very large! Compensates for lack of feature engineering
    - lr: 3e-3 (0.003) - Lower than Gabor approach (0.1)
    - batch_size: 128 - Standard size
    - epochs: 2 - Short training (may be for debugging/quick experiments)

Author: Original research implementation
Date: 2 years ago (2024)
"""

import torch
from raw import ClipsUniformDataset, RawSLP
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from nn_helpers import bin_cross_entropy, accuracy, test_match

# Set seed for reproducibility
torch.manual_seed(1729)

# Select device (CUDA > MPS > CPU)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# ==================== TRAINING HYPERPARAMETERS ====================

# Training duration
epochs = 2  # Very short! May be for debugging or quick experiments
            # Production training would need 50-100+ epochs

# Optimizer parameters
lr = 3e-3  # Learning rate = 0.003 (lower than Gabor's 0.1)
           # Raw signals may need gentler learning

batch_size = 128  # Standard batch size

# Model architecture
n_hiddens = 8192  # VERY LARGE hidden layer!
                  # Raw signals have 20,480 input features (320×64)
                  # Need large capacity to learn patterns without Gabor features
                  # Compare to Gabor approach: only 256 hiddens for 1,024 inputs

# Regularization
cos_regularize = False  # Cosine regularization not used
weight_decay = 0.0  # No L2 regularization (may lead to overfitting with large model)

# MAT file with Gabor transformed data
mat_file = '../clips_uniform.mat'
n_channels = 1

# Load the training/testing data (gabor_results.mat)
training_data = ClipsUniformDataset(mat_file, train=True)
testing_data = ClipsUniformDataset(mat_file, train=False)

train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=5, shuffle=False)

# Our MLP model
net = RawSLP(n_hiddens=n_hiddens).to(device)
# net = DLP(n_hiddens0=64, n_hiddens1=64, M=16, N=64).to(device)

# Select loss function and optimizer
criterion = bin_cross_entropy
optimizer = optim.SGD(net.parameters(), lr, weight_decay=weight_decay)

# Keep track of train/test loss/accuracy for model evaluation
train_losses, test_losses = [], []
train_accs, test_accs = [], []

# Keep track for matching out of 5 EEG/Audio pairs
match_accs = []

# Training loop
for epoch in tqdm(range(epochs)): 

    train_loss, test_loss = 0.0, 0.0
    train_acc, test_acc = 0.0, 0.0
    match_acc = 0.0
    trials = 0

    for inputs, labels in train_dataloader:

        # get the inputs; data is a list of [inputs, labels]
        inputs = [input.to(device) for input in inputs]
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs.reshape(1, -1), labels, cos_regularize).mean()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy(outputs, labels).item()

        trials += 1

    train_losses.append(train_loss / trials)
    train_accs.append(train_acc / trials)
    trials = 0

    # Compute testing loss/accuracy
    for inputs, labels in test_dataloader:
        with torch.no_grad():
            inputs = [input.to(device) for input in inputs]
            labels = labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs.reshape(1, -1), labels).mean()

            test_loss += loss.item()
            test_acc += accuracy(outputs, labels).item()

            match_acc += test_match(outputs, labels).item()

            trials +=1

    test_losses.append(test_loss / trials)
    test_accs.append(test_acc / trials)
    match_accs.append(match_acc / trials)


print('Finished Training')

# Save the model to 'weights.pth'
torch.save(net.state_dict(), 'weights.pth')

# Save loss/accuracy plot
plt.plot(train_losses, c='crimson', label='train loss')
plt.plot(train_accs, '--', c='crimson', label='train acc')
plt.plot(test_losses, c='navy', label='test loss')
plt.plot(test_accs, '--', c='navy', label='test acc')
plt.plot(match_accs, '-.', c='green', label='match acc')
plt.xlabel('epoch')
plt.grid()
plt.xlim(0, epochs-1)
plt.title(f'Raw Loss/Accuracy - lr={lr}, n_hiddens={n_hiddens}, epochs={epochs}')
# plt.title(f'Raw Loss/Accuracy - lr={lr}, n_hiddens={n_hiddens}, weight_decay={weight_decay}')
plt.legend()

# Save figure with name 'LossAcc_-_epochs_{epochs},lr_{lr},n_hiddens_{n_hiddens}.png'
plt.savefig(f'RawLossAcc_-_lr_{lr},n_hiddens_{n_hiddens},epochs_{epochs}.png')
# plt.savefig(f'RawLossAcc_-_weight_decay_{weight_decay},lr_{lr},n_hiddens_{n_hiddens}.png')
