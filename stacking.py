"""
stacking.py - Ensemble Meta-Learning for EEG-Audio Matching

This script implements stacking (also called stacked generalization), an ensemble
method that combines predictions from multiple base models using a meta-model.
The idea is that different models capture different patterns, and a meta-learner
can learn how to best combine their predictions.

Ensemble Stacking Approach:
    Traditional Single Model:
        Input → Model → Prediction

    Stacking Ensemble:
        Input → Model 1 → Predictions 1 ┐
        Input → Model 2 → Predictions 2 ┤ → Concatenate → Meta-Model → Final Prediction
        Input → Model 3 → Predictions 3 ┘

Why Stacking Works:
    - Different models make different types of errors
    - Ensemble reduces variance and improves generalization
    - Meta-model learns which base model to trust in different situations
    - Often outperforms individual models

Base Models in This Implementation:
    1. Raw Signal Model (GaborBaseline1D):
       - Processes raw time-domain EEG/audio signals
       - 1D dilated convolutions
       - Output dimension: 4928

    2. Gabor-Transformed Model (GaborBaseline):
       - Processes Gabor-transformed time-frequency data
       - 2D dilated convolutions
       - Output dimension: 3328

Meta-Model:
    - LogisticRegression (scikit-learn)
    - Input: Concatenated predictions from both base models (5+5 = 10 features)
    - Output: Final classification (which of 5 audio signals matches the EEG)
    - Why Logistic Regression? Simple, interpretable, less prone to overfitting than
      neural networks when meta-features are already informative

Pipeline:
    1. Load pre-trained base models (raw and Gabor)
    2. Generate predictions on validation set from both models
    3. Concatenate predictions to create meta-features
    4. Train meta-model (LogisticRegression) on meta-features
    5. Evaluate combined ensemble performance

Meta-Features Structure:
    Each sample gets 10 meta-features:
    [raw_prob_audio0, raw_prob_audio1, ..., raw_prob_audio4,
     gabor_prob_audio0, gabor_prob_audio1, ..., gabor_prob_audio4]

    The meta-model learns patterns like:
    - When raw model is confident and correct, trust it
    - When models disagree, which one is usually right?
    - How to combine complementary information

Author: Original research implementation
Date: 2 years ago (2024)
"""

from gabor import GaborDataset, GaborBaseline, RawDataset, GaborBaseline1D
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.nn.functional import softmax
from torch.nn import CrossEntropyLoss
from torch import optim
import numpy as np
import random

from stacking_model import StackingModel

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
torch.manual_seed(1729)
torch.cuda.manual_seed(1729)
torch.cuda.manual_seed_all(1729)
np.random.seed(1729)
random.seed(1729)

# Select device for computation
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Proportion of validation data to use
data_prop = 1  # Use 100% of validation set

# ==================== LOAD VALIDATION DATA ====================
# Load validation data for both raw and Gabor-transformed representations
# Note: Using separate validation set (not train/test) to avoid overfitting meta-model
val_raw_data = RawDataset(mat_file='../val_clips.mat', train=False, train_prop=1, data_prop=data_prop, val=True)
val_gabor_data = GaborDataset(mat_file='../val_gabor_results.mat', train=False, train_prop=1, data_prop=data_prop)
# val_wavelet_data =  # Placeholder for potential third model (wavelet transform)

# Create data loaders for batch processing
raw_dataloader = DataLoader(val_raw_data, batch_size=500, shuffle=False)
gabor_dataloader = DataLoader(val_gabor_data, batch_size=500, shuffle=False)
# wavelet_dataloader = DataLoader(val_wavelet_data, batch_size=500, shuffle=False)

# ==================== LOAD PRE-TRAINED BASE MODELS ====================
# Base Model 1: Raw signal model (1D convolutions)
raw_baseline = GaborBaseline1D().to(device)
# Load best weights from previous training run
raw_baseline.load_state_dict(torch.load('max_weights_raw_-_lr_0.001,weight_decay_0.0,data_prop_1.pth'))

# Base Model 2: Gabor-transformed model (2D convolutions)
gabor_baseline = GaborBaseline().to(device)
# Load best weights from previous training run
gabor_baseline.load_state_dict(torch.load('max_weights_-_lr_0.001,weight_decay_0.0,data_prop_1.pth'))


def get_model_predictions(model, val_loader, device, raw=True):
    """
    Generate predictions from a base model for ensemble stacking.

    This function evaluates a trained encoder model on validation data and returns
    probability distributions over the 5 audio candidates for each EEG sample.
    These probabilities become meta-features for the ensemble.

    Process:
        1. Pass EEG and audio through encoder to get embeddings
        2. Compute cosine similarity between EEG and each of 5 audio embeddings
        3. Apply softmax to convert similarities to probabilities
        4. Return probability distribution for each sample

    Args:
        model: Trained encoder model (GaborBaseline or GaborBaseline1D)
        val_loader: DataLoader for validation set
        device: Device to run computation on (cuda/mps/cpu)
        raw (bool): If True, model is raw signal encoder (output dim 4928)
                   If False, model is Gabor encoder (output dim 3328)

    Returns:
        all_outputs: numpy array of shape (n_samples, 5) with probability distributions
        all_labels: numpy array of shape (n_samples,) with ground truth labels (0-4)
    """
    # Set model to evaluation mode (disables dropout, batch norm in eval mode)
    model.eval()

    # Lists to accumulate predictions and labels across all batches
    all_outputs = []  # Probability distributions over 5 audio candidates
    all_labels = []  # Ground truth labels (which audio is correct)

    with torch.no_grad():  # Disable gradient computation for efficiency
        for inputs, labels in val_loader:
            # Move inputs to device
            inputs = [input.to(device) for input in inputs]

            # Remove duplicate EEG samples (dataset has 5x expansion)
            # [::5] extracts every 5th sample to get unique EEGs
            # Same pattern as in training scripts
            inputs[0] = inputs[0][::5]

            # Forward pass through encoder
            # eeg_output: (batch_size, output_dim) - EEG embeddings
            # audio_output: (batch_size*5, output_dim) - Audio embeddings for all 5 candidates
            eeg_output, audio_output = model(inputs)

            # ========== COSINE SIMILARITY COMPUTATION ==========
            # Goal: Compute similarity between each EEG and its 5 audio candidates
            #
            # Strategy: Use broadcasting with unsqueeze(1) for vectorized computation

            if raw:
                # Raw model output dimension: 4928
                # Reshape audio: (batch*5, 4928) → (batch, 5, 4928)
                # Groups 5 audio embeddings per EEG
                audio_reshaped = audio_output.view(-1, 5, 4928)

                # Compute cosine similarity with broadcasting
                # eeg_output shape: (batch, 4928)
                # eeg_output.unsqueeze(1): (batch, 1, 4928) - adds singleton dimension for broadcasting
                # audio_reshaped: (batch, 5, 4928)
                # Broadcasting: (batch, 1, 4928) vs (batch, 5, 4928) → compares each EEG with 5 audios
                # Result: (batch, 5) - similarity between each EEG and its 5 audio candidates
                # dim=2: compute similarity along the embedding dimension (4928)
                cosine_similarities = torch.cosine_similarity(
                    eeg_output.unsqueeze(1),  # Shape: (batch, 1, 4928)
                    audio_reshaped,           # Shape: (batch, 5, 4928)
                    dim=2                     # Compute along embedding dimension
                )
            else:
                # Gabor model output dimension: 3328
                # Same logic as raw model, different embedding size
                audio_reshaped = audio_output.view(-1, 5, 3328)
                cosine_similarities = torch.cosine_similarity(
                    eeg_output.unsqueeze(1),  # Shape: (batch, 1, 3328)
                    audio_reshaped,           # Shape: (batch, 5, 3328)
                    dim=2
                )
            # ===================================================

            # Convert cosine similarities to probabilities via softmax
            # Cosine similarity range: [-1, 1]
            # Softmax normalizes to probability distribution summing to 1
            # Shape: (batch, 5) - probability for each of 5 audio candidates
            smx_cosine_similarities = softmax(cosine_similarities, dim=1)

            # Convert binary labels to class indices (0-4)
            # labels structure: [0,1,0,0,0, 0,0,1,0,0, ...] (one 1 per group of 5)
            # Reshape to (batch, 5) and find which position has the 1
            labels_indices = labels.reshape(-1, 5)
            labels_indices = torch.argmax(labels_indices, dim=1)  # Indices in range [0, 4]

            # Accumulate outputs and labels
            # Move to CPU and convert to numpy for sklearn compatibility
            all_outputs.extend(smx_cosine_similarities.cpu().numpy())
            all_labels.extend(labels_indices.numpy())

    # Convert lists to numpy arrays for sklearn
    return np.array(all_outputs), np.array(all_labels)

# ==================== GENERATE BASE MODEL PREDICTIONS ====================
# Get predictions from both base models on validation set
# predictions1: (n_samples, 5) - raw model probabilities for 5 audio candidates
# true_labels: (n_samples,) - ground truth labels (0-4)
predictions1, true_labels = get_model_predictions(raw_baseline, raw_dataloader, device, raw=True)
print(f"Raw model predictions shape: {predictions1.shape}")  # Should be (n_samples, 5)

# predictions2: (n_samples, 5) - Gabor model probabilities for 5 audio candidates
predictions2, _ = get_model_predictions(gabor_baseline, gabor_dataloader, device, raw=False)
print(f"Gabor model predictions shape: {predictions2.shape}")  # Should be (n_samples, 5)

# meta_features = np.concatenate([predictions1, predictions2], axis=1)
# print(meta_features.shape)

# predictions1_indices = np.argmax(predictions1, axis=1)
# predictions2_indices = np.argmax(predictions2, axis=1)

# model1_correct = np.sum(predictions1_indices == true_labels)
# model2_correct = np.sum(predictions2_indices == true_labels)

# print(model1_correct, model2_correct)

# epochs = 100
# lr = 1e-4
# weight_decay = 0.0

# meta_features = TensorDataset(torch.tensor(meta_features), torch.tensor(true_labels))
# meta_features_dataloader = DataLoader(meta_features, batch_size=500, shuffle=True)

# model = StackingModel(2).to(device)

# criterion = CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr, weight_decay=weight_decay)

# for epoch in range(epochs):
    
#     model.train()
#     train_acc, train_loss = 0.0, 0.0
    
#     for inputs, labels in meta_features_dataloader:
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()

#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         predictions = torch.argmax(outputs, dim=1)

#         train_loss += loss.item()
#         train_acc += torch.sum(predictions == labels).item()
#     print(f'Epoch {epoch} - Loss: {train_loss:.4f}, Accuracy: {train_acc / len(true_labels):.4f}')    


# ==================== META-MODEL TRAINING ====================
## Logistic Regression Meta-Learner

# ========== META-FEATURE CONSTRUCTION ==========
# Concatenate predictions from both base models to create meta-features
# Each sample gets 10 features:
#   - 5 from raw model: [prob_audio0, prob_audio1, ..., prob_audio4]
#   - 5 from Gabor model: [prob_audio0, prob_audio1, ..., prob_audio4]
# Shape: (n_samples, 5) + (n_samples, 5) → (n_samples, 10)
meta_features = np.concatenate([predictions1, predictions2], axis=1)
print(f"Meta-features shape: {meta_features.shape}")

# ========== TRAIN-VALIDATION SPLIT ==========
# Split validation data further for meta-model training and evaluation
# This prevents overfitting the meta-model to the base model predictions
seed = random.randint(0, 100000)  # Random seed for this specific run
print(f'Random seed for train-test split: {seed}')

# 80-20 split of validation data
# X_train_meta: (80% samples, 10) - meta-features for training meta-model
# y_train_meta: (80% samples,) - labels for training
# X_val_meta: (20% samples, 10) - meta-features for evaluating ensemble
# y_val_meta: (20% samples,) - labels for evaluation
X_train_meta, X_val_meta, y_train_meta, y_val_meta = train_test_split(
    meta_features,
    true_labels,
    test_size=0.2,
    random_state=85687  # Fixed seed for reproducibility
)

# ========== TRAIN META-MODEL ==========
# Why Logistic Regression?
#   - Simple and interpretable
#   - Works well when meta-features are already informative
#   - Less prone to overfitting than neural networks
#   - Fast to train
#   - Learns linear combination of base model predictions
#
# The meta-model learns patterns like:
#   - When raw model is confident → weight it higher
#   - When models disagree → which one to trust based on feature patterns
#   - Complementary strengths of different representations (raw vs Gabor)
meta_model = LogisticRegression(max_iter=1000)  # Max iterations for convergence
meta_model.fit(X_train_meta, y_train_meta)

# ========== EVALUATE ENSEMBLE ==========
# Make predictions on held-out validation set
meta_predictions = meta_model.predict(X_val_meta)
print(f"Number of meta-model predictions: {len(meta_predictions)}")

# Compute accuracy of the ensemble
# This should ideally be better than either base model alone
meta_accuracy = accuracy_score(y_val_meta, meta_predictions)

print(f'Meta-Model (Ensemble) Accuracy: {meta_accuracy:.4f}')
print("\nComparison to base models:")
print(f"  - Raw model best-case accuracy: ~{np.max(predictions1, axis=1).mean():.4f} (if always chose max prob)")
print(f"  - Gabor model best-case accuracy: ~{np.max(predictions2, axis=1).mean():.4f}")
print(f"  - Ensemble accuracy: {meta_accuracy:.4f}")
