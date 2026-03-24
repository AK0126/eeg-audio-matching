"""
nn_helpers.py - Loss Functions and Evaluation Metrics for EEG-Audio Matching

This module provides custom loss functions and evaluation metrics for the EEG-Audio
matching task. It includes both binary and multi-class loss functions, as well as
specialized metrics for ranking evaluation.

Loss Functions:
    - bin_cross_entropy: Binary cross-entropy for match/no-match classification
    - multi_cross_entropy: Multi-class cross-entropy for 5-way classification
      (with efficient matrix multiplication implementation)

Evaluation Metrics:
    - accuracy: Point-wise binary classification accuracy
    - test_match: Ranking accuracy - is highest probability the correct match?
    - test_match_encoder: Ranking accuracy using cosine similarity of embeddings
    - test_match_encoder_topk: Top-K ranking accuracy (cumulative)

Key Implementation Details:
    - multi_cross_entropy uses a clever matrix multiplication trick for efficiency
    - test_match_encoder_topk computes cumulative top-k accuracy in one pass
    - All functions handle the 5x expanded dataset structure (5 audio per EEG)

Author: Original research implementation
Date: 2 years ago (2024)
"""

import torch
from torch.nn.functional import softmax, sigmoid
from info_nce import InfoNCE
import numpy as np


def bin_cross_entropy(y_hat, y, regularize=False):
    """
    Binary cross-entropy loss for EEG-Audio matching.

    Computes the standard binary cross-entropy loss with optional cosine similarity
    regularization. Used for binary classification (match vs. no-match).

    Mathematical Form:
        BCE = -[y * log(p) + (1-y) * log(1-p)]
        where p = predicted probability, y = true label (0 or 1)

    Args:
        y_hat (torch.Tensor): Predicted probabilities, shape (batch_size,)
                             Values should be in range [0, 1]
        y (torch.Tensor): True binary labels, shape (batch_size,)
                         Values are 0 (no match) or 1 (match)
        regularize (bool): If True, adds cosine similarity regularization term
                          (experimental feature, not commonly used)

    Returns:
        torch.Tensor: Per-sample losses, shape (batch_size,)

    Example:
        >>> y_hat = torch.tensor([0.8, 0.3, 0.6])  # Predictions
        >>> y = torch.tensor([1.0, 0.0, 1.0])  # Labels
        >>> loss = bin_cross_entropy(y_hat, y)
    """
    # Clamp probabilities to avoid log(0) which causes NaN gradients
    epsilon = 1e-7
    y_clamped = torch.clamp(y_hat.flatten(), epsilon, 1 - epsilon)

    # Compute binary cross-entropy loss
    if regularize:
        # Optional: Add cosine similarity between predictions and labels as regularization
        # Encourages predictions to align with labels in feature space
        cosine = torch.nn.CosineSimilarity(dim=0, eps=1e-7)
        loss = -1 * y * torch.log(y_clamped) - (1 - y) * torch.log(1 - y_clamped) - cosine(y_clamped, y)
    else:
        # Standard binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
        loss = -1 * y * torch.log(y_clamped) - (1 - y) * torch.log(1 - y_clamped)

    # Debug: Print if loss becomes negative (shouldn't happen with proper BCE)
    if loss.mean() < 0:
        print(f"WARNING: Negative loss detected!")
        print(f"y_hat: {y_hat.reshape(-1, 5)}")
        print(f"y: {y.reshape(-1, 5)}")
        print(f"loss: {loss.reshape(-1, 5)}")

    return loss


def multi_cross_entropy(y_hat, y, tau=1, regularize=False):
    """
    Multi-class cross-entropy loss for 5-way EEG-Audio matching.

    Computes cross-entropy between model predictions and one-hot encoded labels
    for the 5-candidate audio matching task. Uses temperature scaling for
    controlling prediction confidence and a clever matrix multiplication trick
    for efficient computation.

    Why This Loss?
        - The task is to identify which of 5 audio signals matches the EEG
        - Standard binary CE treats each pair independently
        - Multi-class CE considers all 5 candidates jointly
        - Encourages model to discriminate between the 5 options

    Mathematical Form:
        L = -log(softmax(y_hat / tau)[correct_class])
        where tau is temperature parameter

    Temperature Scaling (tau):
        - tau = 1: Standard softmax (default)
        - tau > 1: Softer probabilities, less confident predictions
        - tau < 1: Sharper probabilities, more confident predictions

    Args:
        y_hat (torch.Tensor): Raw model outputs (logits), shape (batch_size*5,)
                             Contains predictions for all 5 audio candidates
                             Grouped as: [pred0_audio0, pred0_audio1, ..., pred0_audio4,
                                         pred1_audio0, ...]
        y (torch.Tensor): Binary labels (0/1), shape (batch_size*5,)
                         One 1 per group of 5 indicates correct audio match
                         Example: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, ...]
        tau (float): Temperature parameter for softmax. Default: 1
                    Higher values make predictions more uniform (less confident)
        regularize (bool): Legacy parameter, not used in current implementation

    Returns:
        torch.Tensor: Per-sample losses, shape (batch_size,)

    Implementation Details - Matrix Multiplication Trick:
        Standard implementation (slow):
            for i in range(batch_size):
                loss[i] = -log(probs[i, correct_class[i]])

        Matrix multiplication trick (fast, vectorized):
            loss = diagonal(-log(probs) @ labels^T)

        Why this works:
            - probs: (batch, 5) - probabilities for each audio
            - labels^T: (5, batch) - one-hot encoded (transposed)
            - probs @ labels^T: (batch, batch) matrix where diagonal contains
              log-prob of correct class for each sample
            - diagonal() extracts the relevant values

    Example:
        >>> y_hat = torch.randn(500)  # Raw predictions for batch_size=100
        >>> y = torch.tensor([1,0,0,0,0, 0,1,0,0,0, ...])  # One-hot labels
        >>> loss = multi_cross_entropy(y_hat, y, tau=1.0)
        >>> print(loss.shape)  # (100,) - one loss per EEG sample
    """
    # Prevent log(0) which causes NaN gradients
    epsilon = 1e-7

    # Reshape flat predictions to 5-way classification format
    # Apply temperature scaling and softmax to get probabilities
    # Shape: (batch_size*5,) → (batch_size, 5)
    # Temperature scaling: dividing by tau before softmax
    #   - tau > 1: makes distribution more uniform (softer)
    #   - tau < 1: makes distribution more peaked (sharper)
    y_probs = softmax(y_hat.reshape(-1, 5) / tau, dim=1)

    # Clamp probabilities to prevent numerical instability
    # Ensures values stay in (epsilon, 1-epsilon) range for safe log operation
    y_clamped = torch.clamp(y_probs, epsilon, 1 - epsilon)

    # Convert binary labels to one-hot format for 5 candidates
    # Shape: (batch_size*5,) → (batch_size, 5)
    # Example row: [1, 0, 0, 0, 0] indicates first audio is correct
    y_labels = y.float().reshape(-1, 5)

    # ========== EFFICIENT CROSS-ENTROPY VIA MATRIX MULTIPLICATION ==========
    # Goal: Compute -log(prob of correct class) for each sample
    #
    # Approach: Use matrix multiplication instead of for loop
    #
    # Step 1: Compute -log(probabilities)
    #   -log(y_clamped): (batch, 5) matrix of negative log-probabilities
    #
    # Step 2: Matrix multiply with transposed labels
    #   @ torch.t(y_labels): multiply (batch, 5) × (5, batch) → (batch, batch)
    #
    # Step 3: Extract diagonal
    #   Result is (batch, batch) matrix where result[i,j] = sum of -log(probs[i]) * labels[j]
    #   The diagonal result[i,i] gives -log(prob[i, correct_class[i]])
    #
    # Why diagonal? Because labels are one-hot:
    #   -log(probs[i]) @ labels[i] = -log(probs[i, k]) where k is the 1 in labels[i]
    #   This is exactly the cross-entropy for sample i!
    #
    # Alternative (equivalent but slower):
    #   loss = -torch.sum(y_labels * torch.log(y_clamped), dim=1)
    # ======================================================================
    loss = torch.diagonal(-1 * torch.log(y_clamped) @ torch.t(y_labels))

    return loss


def accuracy(y_hat, y):
    """
    Binary classification accuracy (simple point-wise).

    Converts logits to binary predictions and computes accuracy.

    Args:
        y_hat: Logits (raw predictions before sigmoid)
        y: True binary labels (0 or 1)

    Returns:
        float: Accuracy as a fraction (0.0 to 1.0)
    """
    with torch.no_grad():
        # Convert logits to probabilities via sigmoid, then threshold at 0.5
        y_labels = torch.round(sigmoid(y_hat))
        # Count correct predictions
        correct = torch.t(y_labels) == y
        return correct.sum() / correct.numel()


def test_match(y_hat, y):
    """
    Matching accuracy for 5-way classification (non-encoder models).

    For each EEG sample, checks if the audio with highest predicted probability
    matches the audio with the ground truth label.

    Args:
        y_hat: Predicted probabilities, shape (batch*5,)
        y: True labels (binary), shape (batch*5,)

    Returns:
        float: Proportion of correctly matched samples
    """
    # Reshape to groups of 5 and find which audio has max probability
    y_label = y_hat.reshape(-1, 5).max(1).indices  # Predicted audio index (0-4)
    y_true = y.reshape(-1, 5).max(1).indices  # True audio index (0-4)
    # Return accuracy: fraction of samples where prediction matches ground truth
    return (y_label == y_true).sum().div(len(y_label))


def test_match_encoder(eeg_output, audio_output, labels):
    """
    Matching accuracy for encoder models using cosine similarity.

    Ranks audio candidates by cosine similarity to EEG embedding and checks
    if the highest-ranked audio is the correct match.

    Args:
        eeg_output: EEG embeddings, shape (batch, embedding_dim)
        audio_output: Audio embeddings, shape (batch, 5, embedding_dim)
        labels: Binary labels, shape (batch*5,)

    Returns:
        int: Number of correct matches (not proportion)
    """
    # Compute cosine similarity between EEG and all 5 audio embeddings
    # unsqueeze(1) broadcasts EEG to compare with all 5 audios
    cosine_similarities = torch.cosine_similarity(eeg_output.unsqueeze(1), audio_output, dim=2)

    # Convert cosine similarities to probabilities via softmax
    # Shape: (batch, 5) - probability distribution over 5 audio candidates
    smx_cosine_similarities = softmax(cosine_similarities, dim=1)

    # Find the audio with highest probability
    # Shape: (batch,) with values 0-4 indicating predicted audio index
    predicted_answer = torch.argmax(smx_cosine_similarities, dim=1)

    # Convert binary labels to class indices (0-4)
    labels_indices = labels.reshape(-1, 5)
    labels_indices = torch.argmax(labels_indices, dim=1)

    # Count how many predictions match the ground truth
    # Returns integer count (not proportion)
    return (predicted_answer == labels_indices).sum()


def test_match_encoder_topk(eeg_output, audio_output, labels):
    """
    Compute cumulative top-K accuracy for encoder-based matching.

    Evaluates how often the correct audio appears in the top K predictions when
    ranking by cosine similarity. This metric is more forgiving than strict top-1
    accuracy and shows how well the model's rankings perform.

    Why Top-K Accuracy?
        - Top-1: Strictest metric, correct audio must be ranked #1
        - Top-2: More forgiving, correct audio in top 2 predictions
        - Top-5: Should always be 100% (all 5 candidates included)
        - Useful for understanding model confidence and ranking quality

    Cumulative Counting:
        Unlike independent top-k, uses cumulative counting:
        - top2_correct includes all top1_correct samples PLUS new top-2 samples
        - top3_correct includes all top2_correct samples PLUS new top-3 samples
        - Result: top1 <= top2 <= top3 <= top4 <= top5

    Args:
        eeg_output (torch.Tensor): EEG embeddings, shape (batch_size, embedding_dim)
        audio_output (torch.Tensor): Audio embeddings, shape (batch_size, 5, embedding_dim)
                                     5 audio candidates per EEG
        labels (torch.Tensor): Binary labels, shape (batch_size*5,)
                              One 1 per group of 5 indicates correct match

    Returns:
        torch.Tensor: Cumulative top-k counts, shape (5,)
                     [top1_correct, top2_correct, top3_correct, top4_correct, top5_correct]
                     All values are integers counting number of correct samples

    Example:
        >>> eeg_emb = torch.randn(100, 128)  # 100 EEG samples, 128-dim embeddings
        >>> audio_emb = torch.randn(100, 5, 128)  # 100 × 5 audio, 128-dim
        >>> labels = torch.tensor([1,0,0,0,0, ...])  # Binary labels
        >>> topk_acc = test_match_encoder_topk(eeg_emb, audio_emb, labels)
        >>> print(topk_acc)  # e.g., [65, 82, 91, 97, 100] out of 100 samples
        >>> # Interpretation: 65% top-1 acc, 82% top-2 acc, ..., 100% top-5 acc
    """
    # Compute cosine similarity between EEG and all 5 audio embeddings
    # eeg_output.unsqueeze(1): (batch, 1, emb) for broadcasting
    # audio_output: (batch, 5, emb)
    # Result: (batch, 5) - similarity scores between EEG and each of 5 audios
    cosine_similarities = torch.cosine_similarity(eeg_output.unsqueeze(1), audio_output, dim=2)

    # Convert similarities to probabilities via softmax
    # Higher similarity → higher probability
    # Shape: (batch, 5) - probability distribution over 5 audio candidates
    smx_cosine_similarities = softmax(cosine_similarities, dim=1)

    # Sort audio indices by probability (descending)
    # Result: (batch, 5) where each row is [best_idx, 2nd_best, ..., worst]
    # Example row: [2, 0, 4, 1, 3] means audio #2 has highest probability
    predicted_answer = torch.argsort(smx_cosine_similarities, dim=1, descending=True)

    # Convert binary labels to class indices (0-4)
    # labels structure: [1,0,0,0,0, 0,1,0,0,0, ...] (one 1 per group of 5)
    # Reshape to (batch, 5) and find which position has the 1
    labels_indices = labels.reshape(-1, 5)
    labels_indices = torch.argmax(labels_indices, dim=1)  # Shape: (batch,) with values 0-4

    # ========== CUMULATIVE TOP-K ACCURACY COMPUTATION ==========
    # For each sample, check if the correct audio appears in top K predictions
    #
    # Important: Uses CUMULATIVE counting, not independent counting!

    # Top-1 Accuracy: Correct audio is the #1 prediction
    # predicted_answer[:, 0] extracts the top prediction for each sample
    # Compare with ground truth labels_indices
    top1_correct = (predicted_answer[:, 0] == labels_indices).sum()

    # Top-2 Accuracy: Correct audio in top 2 predictions
    # CUMULATIVE: Includes all top-1 correct + samples where #2 prediction is correct
    # If sample was already top-1 correct, it stays counted
    # If sample's #2 prediction is correct, add it to the count
    top2_correct = top1_correct + (predicted_answer[:, 1] == labels_indices).sum()

    # Top-3 Accuracy: Correct audio in top 3 predictions
    # CUMULATIVE: top2_correct + new samples where #3 prediction is correct
    top3_correct = top2_correct + (predicted_answer[:, 2] == labels_indices).sum()

    # Top-4 Accuracy: Correct audio in top 4 predictions
    # CUMULATIVE: top3_correct + new samples where #4 prediction is correct
    top4_correct = top3_correct + (predicted_answer[:, 3] == labels_indices).sum()

    # Top-5 Accuracy: Correct audio in all 5 predictions
    # Should ALWAYS equal batch_size (100% accuracy) since all 5 are included
    # Useful as sanity check - if this != batch_size, something is wrong
    top5_correct = top4_correct + (predicted_answer[:, 4] == labels_indices).sum()
    # ============================================================

    # Return cumulative counts as tensor
    # To convert to accuracies, divide by batch size later
    return torch.tensor([top1_correct, top2_correct, top3_correct, top4_correct, top5_correct])
    

# NOT IN USE

# Initialize weights of network
def init_weights(layer):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.normal_(layer.weight, std=1)
        layer.bias.data.fill_(0.0)

# Contrastive loss
def contrastive_loss(distance, label, m=0.1):
    """
    Implements contrastive loss.
    """
    return label * distance + (1 - label) * torch.max(0, m - distance)
