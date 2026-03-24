"""
raw.py - Dataset Loader and Model for Raw Signal EEG-Audio Matching

This module provides dataset loaders and neural network models for processing raw
(non-transformed) EEG and audio signals. Raw signals are used as-is without any
frequency transformation like Gabor or wavelet transforms.

Classes:
    - ClipsUniformDataset: Loads raw EEG-Audio pairs from MATLAB .mat files
    - RawSLP: Single-Layer Perceptron for direct classification on raw signals

Use Case:
    This serves as a baseline approach to compare against feature-engineered methods
    (Gabor transform, wavelet transform). By processing raw signals directly, we can
    evaluate how much value the transformations add.

Author: Original research implementation
Date: 2 years ago (2024)
"""

import torch
from torch.utils.data import Dataset
from torch import nn
from math import ceil

import scipy.io as sio


class ClipsUniformDataset(Dataset):
    """
    PyTorch Dataset for loading raw EEG and audio signals from MATLAB files.

    Loads time-domain EEG (320 timesteps × 64 channels) and audio (320 timesteps × 5 candidates)
    without any frequency transformation. Each EEG clip is paired with 5 audio candidates,
    creating a 5x expanded dataset.

    Args:
        mat_file: Path to .mat file containing raw signals
        train: If True, loads training split; if False, loads test split
        train_prop: Proportion of data used for training (default 0.9)
        data_prop: Proportion of total data to load (default 1.0)
    """
    def __init__(self, mat_file, train=True, train_prop=0.9, data_prop=1):
        """
        Initializes instance of ClipsUniformDataset, used to load 'clips_uniform.mat' into PyTorch

        :param mat_file: path to .mat file, e.g. 'clips_uniform.mat'
        :param train: True to load training data, False to load testing data
        """
        # Calculate size of training data
        all_clips = sio.loadmat(mat_file)['clips'][0]
        train_size = int(data_prop * train_prop * len(all_clips))
        test_size = ceil(data_prop * (1 - train_prop) * len(all_clips))

        # Load MAT file
        if train:
            self.clips = all_clips[:train_size]
        else:
            self.clips = all_clips[train_size:train_size + test_size]

    def __len__(self):
        return 5 * len(self.clips)

    def __getitem__(self, idx):
        """
        Obtains item at specified index from dataset.
        The range of idx is 0-5800 (5x1160). Index 0 corresponds to the pair (EEG 1, Audio 1). 
        Index 1 corresponds to the pair (EEG 1, Audio 2), and similarly for indices 2-4.
        Index 5 corresponds to the pair (EEG 2, Audio 1). 

        In general, idx corresponds to the pair (EEG idx // 5, Audio idx % 5)
        
        :param idx: index of datapoint
        :return:    'sample', tuple of matrices (EEG, Audio), where EEG is 320x64 and Audio is 320x1
                    'label', label of correct audio
        """
        clip_idx = idx // 5
        clip = self.clips[clip_idx]
        answer = clip[2][0, 0]
        audio_idx = idx % 5

        # Retrieve EEG and Audio (shapes are 320x64 and 320x5, respectively)
        eegs = torch.tensor(clip[0], dtype=torch.float32)
        audios = torch.tensor(clip[1], dtype=torch.float32)

        # Transpose EEG/Audio, select correct audio clip
        sample = (torch.t(eegs), 
                  torch.t(audios)[audio_idx])
        # 0 for mismatch, 1 for match
        label = int(answer == audio_idx + 1)
        
        return sample, label


# Single layer preceptron (one hidden layer), for raw EEG/Audio data
class RawSLP(nn.Module):
    def __init__(self, n_hiddens, n_channels=64):
        """
        Initialize MLP with one hidden layer
        
        :param n_hiddens: Number of nodes in hidden layer
        :param n_channels: Number of EEG channels used. Default is 64.
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(320*(n_channels + 1), n_hiddens),
            nn.Tanh(),
            nn.Linear(n_hiddens, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        eeg = self.flatten(torch.transpose(x[0], dim0=1, dim1=2))
        audio = self.flatten(x[1])
        X = torch.concat((eeg, audio), dim=1)
        logits = self.linear_relu_stack(X)

        return logits
