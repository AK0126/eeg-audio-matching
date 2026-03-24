"""
clips_uniform.py - Simple Dataset Loader for Raw EEG-Audio Clips

This module provides a basic PyTorch Dataset loader for raw EEG-Audio clips stored
in MATLAB format. Unlike other loaders, this doesn't expand the dataset 5x - it
returns one sample per EEG clip with all 5 audio candidates together.

Classes:
    - ClipsUniformDataset: Loads EEG clips with all 5 audio candidates as single samples

Use Case:
    Used for experiments where you want to process all 5 audio candidates together
    rather than treating them as separate samples. Different from the 5x expansion
    approach used in other loaders.

Author: Original research implementation
Date: 2 years ago (2024)
"""

from torch.utils.data import Dataset

import numpy as np
import scipy.io as sio


class ClipsUniformDataset(Dataset):
    """
    PyTorch Dataset for loading EEG clips with all 5 audio candidates.

    Unlike GaborDataset and RawDataset which use 5x expansion, this loader
    returns each EEG clip once with all 5 audio candidates together.

    Args:
        mat_file: Path to 'clips_uniform.mat' MATLAB file
        train: If True, loads training split (90%); if False, test split (10%)
        transform: Optional transform to apply to samples
        target_transform: Optional transform to apply to labels
    """
    def __init__(self, mat_file, train=True, transform=None, target_transform=None):
        """
        Initializes instance of ClipsUniformDataset, used to load 'clips_uniform.mat' into pytorch

        :param mat_file: path to .mat file, e.g. 'clips_uniform.mat'
        :param train: True to load training data, False to load testing data
        """
        # Calculate size of training data
        train_prop = 0.9
        all_clips = sio.loadmat(mat_file)['clips'][0]
        train_size = int(train_prop * len(all_clips))

        # Load MAT file
        if train:
            self.clips = all_clips[:train_size]
        else:
            self.clips = all_clips[train_size:]
        
        # Apply transforms (optional)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        """
        Obtains item at specified index from database
        
        :param idx: index of datapoint
        :return:    'sample', tuple of matrices (EEG, Audio), where EEG is 320x64 and Audio is 320x5
                    'label', label of correct audio
        """
        clip = self.clips[idx]
        sample = (clip[0], clip[1])
        # -1 to change to zero indexing
        label = clip[2][0][0] - 1

        # Optional transforms
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)
        
        return sample, label
