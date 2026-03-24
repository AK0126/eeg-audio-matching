"""
mlp_gabor_encoder.py - Contrastive Learning Training Script for EEG-Audio Matching

This script trains a Siamese encoder using contrastive learning with InfoNCE (Noise Contrastive
Estimation) loss. Unlike direct classification approaches, this method learns embeddings where
matching EEG-audio pairs are close together and mismatches are far apart in the embedding space.

Contrastive Learning Approach:
    Traditional Classification:
        - Learns: Is this EEG-audio pair a match? (binary classifier)
        - Output: Single probability score

    Contrastive Learning (This Script):
        - Learns: Embeddings where similar pairs are nearby, dissimilar pairs are distant
        - Output: Embedding vectors; matching computed via cosine similarity
        - Advantage: Generalizes better, learns richer representations

InfoNCE Loss:
    A contrastive loss function that maximizes agreement between positive pairs and
    minimizes agreement with negative pairs. For each EEG sample:
        - 1 positive audio (correct match)
        - 4 negative audios (incorrect matches)

    Loss encourages:
        - High similarity: EEG embedding ↔ positive audio embedding
        - Low similarity: EEG embedding ↔ negative audio embeddings

    Temperature parameter (τ): Controls sharpness of similarity distribution
        - Low τ: Sharp distinctions, hard negatives
        - High τ: Softer distinctions, easier training

Training Strategy:
    1. Dataset provides 5x expanded samples (each EEG with 5 audio candidates)
    2. Remove duplicates with [::5] stride to get unique EEG samples
    3. Reshape audio to (batch, 5, embedding_size) for grouped comparison
    4. Separate positive (label=1) and negative (label=0) audio embeddings
    5. Compute InfoNCE loss encouraging EEG to be close to positive, far from negatives

Evaluation Metrics:
    - InfoNCE Loss: Contrastive loss value
    - Match Accuracy: Given 5 audio candidates, did the model rank the correct one highest?
    - Top-K Accuracy: Is the correct audio in the top K predictions?

Key Hyperparameters:
    - embedding_size: Dimension of learned embeddings (128)
    - lr: Learning rate for SGD (1e-2)
    - temperature: InfoNCE temperature parameter (0.5)
    - batch_size: Number of EEG clips per batch (500)

Author: Original research implementation
Date: 2 years ago (2024)
"""

import torch
from gabor import GaborDataset, GaborEncoder
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from info_nce import InfoNCE

from nn_helpers import test_match_encoder, test_match_encoder_topk

# Set seed manually for reproducibility across all random number generators
# Ensures deterministic training for debugging and comparison
torch.manual_seed(1729)  # Set PyTorch CPU random seed
# torch.mps.manual_seed(1729)  # Uncomment for Apple Silicon MPS reproducibility
torch.cuda.manual_seed(1729)  # Set CUDA random seed for single GPU
torch.cuda.manual_seed_all(1729)  # Set CUDA random seed for all GPUs
np.random.seed(1729)  # Set NumPy random seed
random.seed(1729)  # Set Python random seed
# Seed value 1729 (Ramanujan number) chosen for reproducibility

# Select appropriate device for machine
# Priority order: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
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
epochs = 10  # Number of complete passes through the dataset

# Optimizer parameters
lr = 1e-2  # Learning rate for SGD (0.01)
            # Relatively high for SGD, suitable for contrastive learning

# Batch processing
batch_size = 500  # Number of EEG clips per batch
                  # Large batch helps contrastive learning see more negatives

# Architecture parameters
n_hiddens = 512  # Legacy parameter, not used in GaborEncoder
embedding_size = 128  # Dimension of learned embedding space
                      # Balance between expressiveness and computational cost

# Regularization parameters
cos_regularize = False  # Cosine similarity regularization (not used in this script)
weight_decay = 0.0  # L2 regularization strength (0.0 = no regularization)
                    # Can help prevent overfitting if increased

# Data loading
data_prop = 1  # Proportion of dataset to use (1 = 100%, all data)
               # Can reduce for faster debugging/experimentation

# MAT file with Gabor transformed data
train_mat_file = '../train_gabor_results.mat'
test_mat_file = '../test_gabor_results.mat'
mat_file = '../gabor_results.mat'
n_channels = 64

# Load the training/testing data (gabor_results.mat)
training_data = GaborDataset(mat_file, train=True, data_prop=data_prop)
testing_data = GaborDataset(mat_file, train=False, data_prop=data_prop)

train_dataloader = DataLoader(training_data, batch_size, shuffle=False)
test_dataloader = DataLoader(testing_data, batch_size=500, shuffle=False)

# EEG / Audio Encoders
encoder = GaborEncoder(n_hiddens, embedding_size, M=16, N=64).to(device)

# Select loss function and optimizer
criterion = InfoNCE(temperature = 0.5, negative_mode='paired', reduction = 'sum')
optimizer = optim.SGD(encoder.parameters(), lr, weight_decay=weight_decay)
#scheduler = optim.lr_scheduler.CyclicLR(optimizer, 5e-3, 5e-2)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# Keep track of train/test loss/accuracy for model evaluation
train_losses, test_losses = [], []
train_accs, test_accs = [], []

# Keep track for matching out of 5 EEG/Audio pairs
train_match_accs = []
test_match_accs_top1 = []
test_match_accs_top2 = []
test_match_accs_top3 = []
test_match_accs_top4 = []
test_match_accs_top5 = []

# Keep trach of best matching accuracies
max_match_acc = 0.20

# Training loop
for epoch in tqdm(range(epochs)): 

    train_loss, test_loss = 0.0, 0.0
    train_acc, test_acc = 0.0, 0.0
    train_match_acc = 0.0
    test_match_acc = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    trials = 0

    for inputs, labels in train_dataloader:

        # Move data to GPU/device
        # inputs is a tuple: (eeg_batch, audio_batch)
        inputs = [input.to(device) for input in inputs]
        labels = labels.to(device)

        # ========== CRITICAL DATA RESHAPING FOR CONTRASTIVE LEARNING ==========
        # Problem: Dataset provides 5x expanded samples (each EEG repeated 5 times)
        # Original batch structure:
        #   EEG: [eeg0, eeg0, eeg0, eeg0, eeg0, eeg1, eeg1, ...] (batch_size*5 samples)
        #   Audio: [audio0, audio1, audio2, audio3, audio4, audio0, audio1, ...] (batch_size*5 samples)
        #   Labels: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, ...] (one 1 per group of 5)

        # Solution 1: Remove duplicate EEG samples with stride indexing
        # Extract every 5th sample to get unique EEG clips
        # Shape transformation: (batch_size*5, 1024) → (batch_size, 1024)
        # [::5] means start at 0, take every 5th element: indices 0, 5, 10, 15, ...
        # Result: [eeg0, eeg1, eeg2, ...] (no duplicates)
        eeg_input = inputs[0][::5]

        # Solution 2: Reshape audio to group 5 candidates per EEG
        # Shape transformation: (batch_size*5, 1024) → (batch_size, 5, 1024)
        # -1 infers batch_size automatically: batch_size*5 // 5 = batch_size
        # Result: Each row contains 5 audio candidates for one EEG sample
        # audio_input[0] = [audio0, audio1, audio2, audio3, audio4] for eeg0
        # audio_input[1] = [audio0, audio1, audio2, audio3, audio4] for eeg1
        audio_input = inputs[1].view(-1, 5, 1024)

        # Update inputs with deduplicated and reshaped data
        inputs = [eeg_input, audio_input]
        # ======================================================================

        # Zero gradients from previous iteration
        optimizer.zero_grad()

        # Forward pass through encoder
        # eeg_output: (batch_size, embedding_size) - embeddings for unique EEGs
        # audio_output: (batch_size*5, embedding_size) - embeddings for all audio
        eeg_output, audio_output = encoder(inputs)

        # Flatten audio output for easier indexing
        # Ensure shape is (batch_size*5, embedding_size)
        audio_output = audio_output.view(-1, embedding_size)

        # ========== POSITIVE/NEGATIVE SEPARATION FOR InfoNCE LOSS ==========
        # InfoNCE requires: 1 positive + N negatives per anchor (EEG)
        # Labels structure: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, ...] (one 1 per 5 samples)

        # Extract positive audio embeddings (where label == 1)
        # These are the correct matches for each EEG
        # Shape: (batch_size, embedding_size)
        positive_audio = audio_output[labels == 1]

        # Extract negative audio embeddings (where label == 0)
        # These are the incorrect matches for each EEG
        # Reshape to group 4 negatives per EEG: (batch_size*4, emb) → (batch_size, 4, emb)
        # Each EEG has 4 negative audio candidates
        negative_audio = audio_output[labels == 0].reshape(-1, 4, embedding_size)
        # ===================================================================

        # Compute InfoNCE contrastive loss
        # Arguments:
        #   - eeg_output: anchor embeddings (batch_size, embedding_size)
        #   - positive_audio: positive embeddings (batch_size, embedding_size)
        #   - negative_audio: negative embeddings (batch_size, 4, embedding_size)
        # Loss encourages high similarity between anchor-positive pairs
        # and low similarity between anchor-negative pairs
        loss = criterion(eeg_output, positive_audio, negative_audio)

        # Backpropagation and optimization
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model weights

        # Accumulate training loss
        train_loss += loss.item()

        # Compute matching accuracy: how often is the correct audio ranked highest?
        # Reshape audio_output back to (batch_size, 5, embedding_size) for comparison
        train_match_acc += test_match_encoder(eeg_output, audio_output.view(-1, 5, embedding_size), labels).item()

        # Track total number of samples processed
        trials += len(labels)
        # scheduler.step()  # Uncomment if using learning rate scheduler

    # Compute average losses/accuracies for this epoch
    # Multiply by 5 to account for the 5x expansion in dataset
    # (dividing by trials gives per-expanded-sample, *5 gives per-EEG-clip)
    train_losses.append(train_loss / trials * 5)
    train_accs.append(train_acc / trials * 5)  # Not used in encoder training
    train_match_accs.append(train_match_acc / trials * 5)

    trials = 0  # Reset for test loop

    # ==================== TESTING LOOP ====================
    # Evaluate model on held-out test set
    for inputs, labels in test_dataloader:
        with torch.no_grad():  # Disable gradient computation for efficiency
            # Move data to device
            inputs = [input.to(device) for input in inputs]
            labels = labels.to(device)

            # Same data reshaping as training loop
            # Remove duplicate EEGs: (batch*5, 1024) → (batch, 1024)
            eeg_input = inputs[0][::5]

            # Reshape audio into groups of 5: (batch*5, 1024) → (batch, 5, 1024)
            audio_input = inputs[1].view(-1, 5, 1024)

            inputs = [eeg_input, audio_input]

            # Forward pass through encoder
            eeg_output, audio_output = encoder(inputs)

            # Flatten audio output for loss computation
            audio_output = audio_output.view(-1, embedding_size)

            # Separate positive and negative audio embeddings
            positive_audio = audio_output[labels == 1]
            negative_audio = audio_output[labels == 0].reshape(-1, 4, embedding_size)

            # Compute InfoNCE loss on test set
            loss = criterion(eeg_output, positive_audio, negative_audio)

            test_loss += loss.item()

            # Compute top-k accuracy: is correct audio in top K predictions?
            # Returns tensor of shape (5,) with cumulative top-1, top-2, ..., top-5 accuracies
            # Top-1: Correct audio ranked #1
            # Top-2: Correct audio in top 2 predictions
            # ... and so on
            test_match_acc += test_match_encoder_topk(eeg_output, audio_output.view(-1, 5, embedding_size), labels)

            trials += len(labels)

    # Store test metrics for this epoch
    # Multiply by 5 to account for 5x expansion (same reasoning as training)
    test_losses.append(test_loss / trials * 5)
    test_accs.append(test_acc / trials * 5)

    # Store top-k accuracies separately
    # test_match_acc is a tensor: [top1_acc, top2_acc, top3_acc, top4_acc, top5_acc]
    test_match_accs_top1.append(test_match_acc[0] / trials * 5)  # Correct audio ranked #1
    test_match_accs_top2.append(test_match_acc[1] / trials * 5)  # Correct in top 2
    test_match_accs_top3.append(test_match_acc[2] / trials * 5)  # Correct in top 3
    test_match_accs_top4.append(test_match_acc[3] / trials * 5)  # Correct in top 4
    test_match_accs_top5.append(test_match_acc[4] / trials * 5)  # Correct in top 5 (always 1.0)

    # Uncomment to use learning rate scheduler
    # scheduler.step(test_loss / trials)

    # Save model weights if this epoch achieved best top-1 accuracy
    # Keep track of best performing model for later use
    if test_match_acc[0] / trials * 5 > max_match_acc:
        torch.save(encoder.state_dict(), 'max_weights.pth')
        max_match_acc = test_match_acc[0] / trials * 5

print(f"Finished Training. Maximum top-1 accuracy: {max_match_acc:.4f}")

# Save final model weights (last epoch, may not be the best)
torch.save(encoder.state_dict(), 'weights.pth')

# ==================== VISUALIZATION ====================
# Create loss/accuracy plot to visualize training progress
# Plot training and test losses
plt.plot(train_losses, c='crimson', label='train loss')  # InfoNCE loss on training set
plt.plot(test_losses, c='navy', label='test loss')  # InfoNCE loss on test set

# Plot matching accuracies (commented lines are for binary accuracy, not used with encoder)
# plt.plot(train_accs, '--', c='crimson', label='train acc')
# plt.plot(test_accs, '--', c='navy', label='test acc')

# Plot training matching accuracy (correct audio ranked highest among 5 candidates)
plt.plot(train_match_accs, '-.', c='orange', label='train match acc')

# Plot test top-k accuracies
# Top-1 is most important: correct audio must be ranked #1
# Top-5 should always be 1.0 (correct audio is always in all 5)
plt.plot(test_match_accs_top1, '-.', c='green', label='test acc top 1')  # Strictest metric
plt.plot(test_match_accs_top2, '-.', c='purple', label='test acc top 2')
plt.plot(test_match_accs_top3, '-.', c='cyan', label='test acc top 3')
plt.plot(test_match_accs_top4, '-.', c='black', label='test acc top 4')

# Format plot
plt.xlabel('epoch')
plt.grid()
plt.xlim(0, epochs-1)
# plt.ylim(0, 1)  # Uncomment to fix y-axis to [0, 1] for percentages
plt.title(f'Loss/Accuracy Encoder - lr={lr}, n_hiddens={n_hiddens}, data_prop={data_prop}')
plt.legend()

# Save figure with hyperparameters in filename for easy comparison
plt.savefig(f'LossAccEncoder_-_lr_{lr},n_hiddens_{n_hiddens},data_prop_{data_prop}.png')
