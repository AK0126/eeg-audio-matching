"""
stacking_model.py - Neural Network Meta-Model for Ensemble Stacking

This module provides a simple neural network meta-model for combining predictions
from multiple base models in an ensemble stacking approach. Alternative to using
scikit-learn's LogisticRegression for the meta-learner.

Classes:
    - StackingModel: Neural network that takes concatenated base model predictions
      and outputs final ensemble predictions

Architecture:
    Input: Concatenated probabilities from N base models (5*N features)
    Hidden: Small MLP (default 16 hidden units)
    Output: 5-way softmax classification

Author: Original research implementation
Date: 2 years ago (2024)
"""

import torch
from torch.utils.data import Dataset
from torch import nn
from math import ceil

import numpy as np
import scipy.io as sio
import h5py
from torch.nn import RNN, LSTM


class StackingModel(nn.Module):
    """
    Neural network meta-model for ensemble stacking.

    Takes concatenated predictions from multiple base models and learns to combine
    them for final classification. Alternative to LogisticRegression meta-learner.

    Architecture:
        Input(5*num_models) → Linear(n_hiddens) → ReLU → Linear(5) → Softmax

    Args:
        num_models (int): Number of base models being ensembled
        n_hiddens (int): Hidden layer size (default 16)

    Example:
        >>> meta_model = StackingModel(num_models=2, n_hiddens=16)
        >>> # Input: [model1_probs(5), model2_probs(5)] = 10 features
        >>> meta_features = torch.randn(batch_size, 10)
        >>> output = meta_model(meta_features)  # Shape: (batch_size, 5)
    """

    def __init__(self, num_models, n_hiddens=16):
        """
        Initialize stacking meta-model.

        :param num_models: Number of base models to ensemble
        :param n_hiddens: Number of hidden units (default 16)
        """
        super().__init__()
        self.num_models = num_models
        self.linear_stack = nn.Sequential(
            nn.Linear(5 * num_models, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, 5),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        output = self.linear_stack(x)

        return output