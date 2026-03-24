"""
gabor.py - Neural Network Architectures and Data Loaders for Gabor-Transformed EEG-Audio Matching

This module is part of an EEG-Audio Matching research project that uses Multi-Layer Perceptrons
and various neural network architectures to classify whether brain signals (EEG) correspond to
perceived audio stimuli.

Research Context:
    - Task: Binary classification to match EEG brain signals with corresponding audio
    - Data: 1160 EEG clips, each with 5 candidate audio signals
      * EEG: 64 channels × 16 time bins × 64 frequency bins (after Gabor transform)
      * Audio: 1 channel × 16 time bins × 64 frequency bins (after Gabor transform)
    - Gabor Transform: Discrete Gabor Transform (DGT) with M=16, N=64 parameters
      provides time-frequency representation of signals
    - Dataset Structure: Each EEG clip paired with 5 audio candidates creates 5800 total samples

Architecture Approaches:
    This file contains 14+ neural network variants exploring different strategies:
    - Encoder Models: Siamese networks with separate EEG/audio encoders
    - Dilated Convolutions: Multi-scale feature extraction (GaborBaseline)
    - Recurrent Models: LSTM/RNN for sequential processing
    - Direct Classifiers: MLP-based direct classification (SLP, DLP)
    - Hybrid Models: Combining convolution + fully connected layers

Classes:
    Dataset Loaders:
        - GaborDataset: Loads Gabor-transformed data from 'gabor_results.mat'
        - RawDataset: Loads raw EEG/audio data from HDF5 files
        - GaborChannelDataset: Loads specific EEG channels for single-channel experiments

    Encoder Architectures (Siamese Networks):
        - GaborEncoder: 2D convolution-based encoder with embedding output
        - GaborEncoderRaw: 1D convolution variant for raw signals
        - GaborEncoder2: Reduced-dimension encoder with stride-2 convolution
        - GaborBaseline/GaborBaselineFull: Dilated convolution encoders
        - GaborBaseline1D: 1D dilated convolution variant

    Recurrent Architectures:
        - GaborLSTM: LSTM-based encoder treating frequency bins as sequences
        - GaborConvLSTM: Combines convolution channel reduction with LSTM
        - GaborRecurrent: Simple RNN-based encoder

    Direct Classification Architectures:
        - SLP: Single-layer perceptron (1 hidden layer)
        - DLP: Double-layer perceptron (2 hidden layers)
        - EmbedMLP: Separate feature extraction layers for EEG and audio
        - CosMLP: Cosine similarity-based matching
        - ConvMLP/ConvMLP2: Convolutional feature extraction with MLP classifier

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


class GaborDataset(Dataset):
    """
    PyTorch Dataset for loading Gabor-transformed EEG-Audio matching data.

    Loads preprocessed data from 'gabor_results.mat' where both EEG and audio signals
    have been transformed using Discrete Gabor Transform (DGT) to obtain time-frequency
    representations. The Gabor transform converts time-domain signals into a 2D
    representation with time and frequency dimensions.

    Data Structure:
        - Input file contains 1160 EEG clips, each with 5 candidate audio signals
        - Each clip in the array contains:
          * First 64 elements: EEG channels (each is M×N Gabor coefficients)
          * Last 5 elements: 5 audio candidates (each is M×N Gabor coefficients)
        - Dataset size: 1160 clips × 5 audio = 5800 samples (5x expansion)

    Dataset Indexing:
        The dataset expands each EEG clip into 5 samples (one per audio candidate).
        Index mapping:
            idx 0-4:   EEG clip 0 with audio 0,1,2,3,4
            idx 5-9:   EEG clip 1 with audio 0,1,2,3,4
            idx 10-14: EEG clip 2 with audio 0,1,2,3,4
            ...
        Formula: clip_idx = idx // 5, audio_idx = idx % 5

    Args:
        mat_file (str): Path to .mat file containing Gabor-transformed data
        M (int): Gabor transform time dimension, typically 16
        N (int): Gabor transform frequency dimension, typically 64
        train (bool): If True, loads training data; if False, loads test data
        train_prop (float): Proportion of dataset used for training (default 0.9)
        data_prop (float): Proportion of total data to load (default 1.0, i.e., all data)

    Example:
        >>> dataset = GaborDataset('gabor_results.mat', M=16, N=64, train=True)
        >>> print(len(dataset))  # 5800 (1160 clips × 5 audio)
        >>> (eeg, audio), label = dataset[0]
        >>> print(eeg.shape)  # (64, 16, 64) - 64 channels × 16 time × 64 freq
        >>> print(audio.shape)  # (16, 64) - 16 time × 64 freq
    """
    def __init__(self, mat_file, M=16, N=64, train=True, train_prop=0.9, data_prop=1):
        """
        Initializes instance of GaborDataset, loads Gabor-transformed data from HDF5 file.

        :param mat_file:    path to .mat file, e.g. 'gabor_results.mat'
        :param M, N:        Parameters for Gabor transform (time bins, frequency bins)
        :param train:       True to load training data, False to load testing data
        :param train_prop:  Proportion of dataset used as training data
        :param data_prop:   Proportion of total data to load (for debugging/experiments)
        """
        # Load HDF5 file containing Gabor-transformed EEG and audio data
        # File structure: 'results' contains the Gabor coefficients
        #                 'ans' contains ground truth labels (1-5 indicating correct audio)
        all_clips = h5py.File(mat_file)['results']

        # Calculate dataset size based on data_prop (allows loading partial data)
        size = int(data_prop * len(all_clips['results'][0, 0]))
        print(f"Size of dataset: {size}")

        # Load clips and labels from HDF5 file
        # Original shape: (69, M*N, num_clips) where 69 = 64 EEG channels + 5 audio
        # Extract first 'size' clips based on data_prop
        self.clips = all_clips['results'][:,:,:size]

        # Labels are 1-indexed (MATLAB convention): 1, 2, 3, 4, or 5
        # Indicates which of the 5 audio signals is the correct match
        self.labels = all_clips['ans'][:size]

        # Transpose to put clips as first dimension for easier indexing
        # Shape transformation: (69, M*N, num_clips) -> (num_clips, M*N, 69)
        # This makes self.clips[i] retrieve the i-th clip with all its channels
        self.clips = np.transpose(self.clips, (2, 1, 0))
        
        ### FOR SMALL DATASET
        # Calculate size of training data
        # all_clips = sio.loadmat(mat_file)['results']

        # train_size = int(data_prop * train_prop * len(all_clips['results'][0, 0]))
        # test_size = ceil(data_prop * (1 - train_prop) * len(all_clips['results'][0, 0]))
  
        # # Load MAT file
        # if train:
        #     self.clips = all_clips['results'][0, 0][:train_size]
        #     self.labels = all_clips['ans'][0, 0][0][:train_size]
        # else:
        #     self.clips = all_clips['results'][0, 0][train_size:train_size + test_size]
        #     self.labels = all_clips['ans'][0, 0][0][train_size:train_size + test_size]

    def __len__(self):
        return 5 * len(self.clips)

    def __getitem__(self, idx):
        """
        Obtains item at specified index from dataset.

        Dataset Indexing Strategy:
            The dataset has 1160 EEG clips, each with 5 audio candidates.
            Total dataset size: 1160 × 5 = 5800 samples.

            Index mapping:
                idx 0-4:   EEG clip 0 with audio 0,1,2,3,4
                idx 5-9:   EEG clip 1 with audio 0,1,2,3,4
                idx 10-14: EEG clip 2 with audio 0,1,2,3,4
                ...

            Formula: clip_idx = idx // 5, audio_idx = idx % 5

        :param idx: Index in range [0, 5800)
        :return:    Tuple of (sample, label) where:
                    - sample: Tuple of (EEG tensor, Audio tensor)
                      * EEG shape: (64, M, N) - 64 channels × M time × N freq
                      * Audio shape: (M, N) - M time × N freq
                    - label: Binary int (0 or 1)
                      * 1 if this audio matches the EEG (correct pair)
                      * 0 if mismatch (incorrect pair)
        """
        # Determine which EEG clip this index corresponds to
        # Since each clip appears 5 times (once per audio), divide by 5
        clip_idx = idx // 5

        # Retrieve the clip data (contains EEG + all 5 audio signals)
        # Shape: (M*N, 69) where 69 = 64 EEG channels + 5 audio signals
        clip = self.clips[clip_idx]

        # Ground truth: which audio (1-5) is the correct match?
        # Note: MATLAB 1-indexing, so answer is in range [1, 5]
        answer = self.labels[clip_idx]

        # Determine which of the 5 audio candidates this index represents
        # Uses modulo to cycle through 0,1,2,3,4 for each EEG clip
        audio_idx = idx % 5

        # Data structure in clip array:
        # clip[0:64] - First 64 elements are EEG channels (each is M×N Gabor coefficients)
        # clip[64:69] - Last 5 elements are the 5 audio signals (each is M×N Gabor coefficients)

        # Extract EEG (all 64 channels) and specific audio candidate
        # clip[:-5] extracts first 64 channels (all EEG)
        # clip[-5 + audio_idx] extracts one of the 5 audio signals
        #   -5 + 0 = -5 (first audio, index 64)
        #   -5 + 1 = -4 (second audio, index 65)
        #   ...
        #   -5 + 4 = -1 (fifth audio, index 68)
        sample = (torch.tensor(clip[:-5], dtype=torch.float32),
                  torch.tensor(clip[-5 + audio_idx], dtype=torch.float32))

        # Create binary label: 1 if this audio matches, 0 otherwise
        # Convert from 1-indexed (answer) to 0-indexed (audio_idx) for comparison
        # answer is in [1,5], audio_idx is in [0,4], so add 1 to audio_idx
        label = int(answer == audio_idx + 1)

        return sample, label


class RawDataset(Dataset):
    """
    PyTorch Dataset for loading raw (non-Gabor-transformed) EEG-Audio matching data.

    Loads EEG and audio data stored in HDF5 format where the data is saved separately
    as EEG arrays, audio arrays, and answer labels. Unlike GaborDataset which loads
    Gabor-transformed data, this class loads the raw time-domain signals.

    Data Structure:
        - HDF5 file contains separate 'train_clips' and 'test_clips' groups
        - Each group contains:
          * 'eeg': References to EEG signal arrays
          * 'audio': References to audio signal arrays (5 candidates per EEG)
          * 'answer': Ground truth labels (1-5 indicating correct audio)

    HDF5 Reference Handling:
        MATLAB saves large arrays as references in HDF5. The 'eeg', 'audio', and
        'answer' fields contain references (pointers) to the actual data, not the
        data itself. This code dereferences these pointers to access the actual arrays.

    Dataset Indexing:
        Similar to GaborDataset, uses 5x expansion:
        - Each EEG clip paired with 5 audio candidates
        - Index formula: clip_idx = idx // 5, audio_idx = idx % 5

    Args:
        mat_file (str): Path to HDF5 file containing raw EEG and audio data
        M (int): Legacy parameter from Gabor version, not used here
        N (int): Legacy parameter from Gabor version, not used here
        train (bool): If True, loads training data; if False, loads test data
        train_prop (float): Legacy parameter, not used (train/test split pre-defined)
        data_prop (float): Proportion of data to load (default 1.0 for all data)

    Example:
        >>> dataset = RawDataset('raw_data.mat', train=True, data_prop=0.5)
        >>> (eeg, audio), label = dataset[0]
        >>> print(eeg.shape)  # EEG signal shape
        >>> print(audio.shape)  # Audio signal shape
    """

    def __init__(self, mat_file, M=16, N=64, train=True, train_prop=0.9, data_prop=1):
        """
        Initializes instance of RawDataset, loads raw EEG and audio from HDF5 file.

        :param mat_file:    Path to HDF5 .mat file containing raw data
        :param M, N:        Legacy parameters from Gabor version (not used here)
        :param train:       True to load training data, False to load testing data
        :param train_prop:  Legacy parameter (not used, train/test split pre-defined in file)
        :param data_prop:   Proportion of total data to load
        """

        if train:
            # Open HDF5 file and access the 'train_clips' group
            all_clips = h5py.File(mat_file)['train_clips']

            # Calculate how many clips to load based on data_prop
            size = int(data_prop * len(all_clips['eeg']))
            print(f"Size of dataset: {size}")

            # Load references to EEG, audio, and labels
            # Note: In MATLAB-generated HDF5 files, large arrays are stored as references
            # These variables contain HDF5 object references, not the actual data
            self.eeg = all_clips['eeg'][:size]
            self.audio = all_clips['audio'][:size]
            self.labels = all_clips['answer'][:size]

            # Dereference EEG data
            # ref[0] extracts the reference pointer from each element
            # all_clips[ref[0]] follows the pointer to get the actual EEG array
            # This pattern is necessary when MATLAB saves data with '-v7.3' flag
            self.eeg = [np.array(all_clips[ref[0]]) for ref in self.eeg]
            self.eeg = np.stack(self.eeg)  # Stack list of arrays into single numpy array

            # Dereference audio data (same pattern as EEG)
            # Each audio reference points to an array containing 5 audio candidates
            self.audio = [np.array(all_clips[ref[0]]) for ref in self.audio]
            self.audio = np.stack(self.audio)

            # Dereference labels
            # Labels are scalar values, but still stored as references
            # [0] at the end extracts the scalar from the 1-element array
            self.labels = [all_clips[ref[0]][0] for ref in self.labels]
            self.labels = np.stack(self.labels)
        else:
            # Open HDF5 file and access the 'test_clips' group
            # Test data is pre-separated in the HDF5 file
            all_clips = h5py.File(mat_file)['test_clips']

            # Calculate how many test clips to load based on data_prop
            size = int(data_prop * len(all_clips['eeg']))
            print(f"Size of dataset: {size}")

            # Load references to EEG, audio, and labels (same as training)
            self.eeg = all_clips['eeg'][:size]
            self.audio = all_clips['audio'][:size]
            self.labels = all_clips['answer'][:size]

            # Dereference EEG data (same HDF5 reference dereferencing as training)
            self.eeg = [np.array(all_clips[ref[0]]) for ref in self.eeg]
            self.eeg = np.stack(self.eeg)

            # Dereference audio data
            self.audio = [np.array(all_clips[ref[0]]) for ref in self.audio]
            self.audio = np.stack(self.audio)

            # Dereference labels
            self.labels = [all_clips[ref[0]][0] for ref in self.labels]
            self.labels = np.stack(self.labels)
        
        
        ### FOR SMALL DATASET
        # Calculate size of training data
        # all_clips = sio.loadmat(mat_file)['results']

        # train_size = int(data_prop * train_prop * len(all_clips['results'][0, 0]))
        # test_size = ceil(data_prop * (1 - train_prop) * len(all_clips['results'][0, 0]))
  
        # # Load MAT file
        # if train:
        #     self.clips = all_clips['results'][0, 0][:train_size]
        #     self.labels = all_clips['ans'][0, 0][0][:train_size]
        # else:
        #     self.clips = all_clips['results'][0, 0][train_size:train_size + test_size]
        #     self.labels = all_clips['ans'][0, 0][0][train_size:train_size + test_size]

    def __len__(self):
        return 5 * len(self.eeg)

    def __getitem__(self, idx):
        """
        Obtains item at specified index from dataset.

        Dataset Indexing Strategy (same as GaborDataset):
            Uses 5x expansion: each EEG clip paired with 5 audio candidates.
            Total dataset size: num_clips × 5 samples.

            Index mapping:
                idx 0-4:   EEG clip 0 with audio 0,1,2,3,4
                idx 5-9:   EEG clip 1 with audio 0,1,2,3,4
                ...

            Formula: clip_idx = idx // 5, audio_idx = idx % 5

        :param idx: Index in range [0, dataset_size)
        :return:    Tuple of (sample, label) where:
                    - sample: Tuple of (EEG tensor, Audio tensor)
                    - label: Binary int (0 or 1) indicating match/mismatch
        """
        # Determine which EEG clip this index corresponds to
        clip_idx = idx // 5

        # Retrieve the EEG and audio data for this clip
        # audio contains all 5 candidate audio signals for this EEG
        eeg = self.eeg[clip_idx]
        audio = self.audio[clip_idx]

        # Ground truth: which audio (1-5) is the correct match?
        answer = self.labels[clip_idx]

        # Determine which of the 5 audio candidates this index represents
        audio_idx = idx % 5

        # Extract EEG and specific audio candidate
        # audio[audio_idx] selects one of the 5 audio candidates
        sample = (torch.tensor(eeg, dtype=torch.float32),
                  torch.tensor(audio[audio_idx], dtype=torch.float32))

        # Create binary label: 1 if this audio matches, 0 otherwise
        # Convert from 1-indexed (answer) to 0-indexed (audio_idx) for comparison
        label = int(answer == audio_idx + 1)

        return sample, label


class GaborEncoder(nn.Module):
    """
    Siamese Neural Network for EEG-Audio matching using Gabor-transformed signals.

    This encoder uses separate but parallel pathways (Siamese architecture) to encode
    EEG and audio signals into a shared embedding space. The key idea is to learn
    representations where matching EEG-audio pairs have similar embeddings.

    Architecture Design:
        Two parallel encoders share the same architecture but have separate weights:
        - EEG Encoder: Processes 64-channel EEG Gabor coefficients
        - Audio Encoder: Processes single-channel audio Gabor coefficients

        Both encoders use:
        1. 2D Convolution for spatial feature extraction
        2. Tanh activation for non-linearity
        3. Flatten to convert 2D features to 1D
        4. Linear layer to project to embedding space

    Architecture Diagram:

        EEG Pathway:
            Input: (batch, 64, M, N) - 64 channels × M time × N freq
                ↓
            Conv2d(64→1, kernel=3×3, pad=1) - Channel reduction to single feature map
                ↓
            Tanh() - Non-linear activation
                ↓
            Flatten - Reshape to (batch, M*N) = (batch, 1024) for M=16, N=64
                ↓
            Linear(1024 → embedding_size) - Project to embedding space
                ↓
            Output: (batch, embedding_size)

        Audio Pathway:
            Input: (batch, 1, M, N) - 1 channel × M time × N freq
                ↓
            Conv2d(1→1, kernel=3×3, pad=1) - Feature extraction
                ↓
            Tanh() - Non-linear activation
                ↓
            Flatten - Reshape to (batch, M*N) = (batch, 1024)
                ↓
            Linear(1024 → embedding_size) - Project to embedding space
                ↓
            Output: (batch, embedding_size)

        Matching: Cosine similarity or other distance metrics computed between
                  EEG and audio embeddings

    Args:
        n_hiddens (int): Legacy parameter, not used in current architecture
        embedding_size (int): Dimension of output embedding space (e.g., 128, 256)
        M (int): Gabor time dimension, typically 16
        N (int): Gabor frequency dimension, typically 64
        n_channels (int): Number of EEG channels, default 64

    Attributes:
        M (int): Stored Gabor time dimension for tensor reshaping in forward pass
        N (int): Stored Gabor frequency dimension for tensor reshaping
        eeg_encoder (nn.Sequential): EEG encoding pathway
        audio_encoder (nn.Sequential): Audio encoding pathway

    Example:
        >>> model = GaborEncoder(n_hiddens=512, embedding_size=128, M=16, N=64)
        >>> eeg_emb, audio_emb = model([eeg_batch, audio_batch])
        >>> similarity = torch.cosine_similarity(eeg_emb, audio_emb)
        >>> print(similarity.shape)  # (batch_size,)
    """

    def __init__(self, n_hiddens, embedding_size, M, N, n_channels=64):
        """
        Initialize Gabor Encoder (Siamese Neural Network architecture)

        :param n_hiddens: Number of nodes in hidden layer (legacy, not used)
        :param embedding_size: Dimension of output embedding space
        :param M, N: Values of M and N in Gabor transform (time bins, frequency bins)
        :param n_channels: Number of EEG channels used. Default is 64.
        """
        super().__init__()
        # Store Gabor dimensions for tensor reshaping in forward pass
        self.M = M  # Time bins in Gabor transform (typically 16)
        self.N = N  # Frequency bins in Gabor transform (typically 64)

        # EEG Encoder: Processes 64-channel Gabor-transformed EEG
        self.eeg_encoder = nn.Sequential(
            # Channel reduction: 64 EEG channels → 1 feature map
            # Padding=1 maintains spatial dimensions (M×N stays the same)
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(3, 3), stride=1, padding=1),
            nn.Tanh(),  # Non-linear activation, range [-1, 1]
            nn.Flatten(),  # Reshape from (batch, 1, M, N) to (batch, M*N)
            # Project to embedding space: M*N features → embedding_size
            # For M=16, N=64: 1024 → embedding_size
            nn.Linear(1024, embedding_size),
        )

        # Audio Encoder: Processes single-channel Gabor-transformed audio
        # Architecture mirrors EEG encoder but processes 1 channel instead of 64
        self.audio_encoder = nn.Sequential(
            # Feature extraction from single audio channel
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1),
            nn.Tanh(),
            nn.Flatten(),  # Reshape from (batch, 1, M, N) to (batch, M*N)
            # Project to same embedding space as EEG
            nn.Linear(1024, embedding_size),
        )

    def forward(self, x):
        """
        Forward pass through Siamese network.

        Args:
            x: Tuple of (eeg_batch, audio_batch)
               - eeg_batch: shape (batch, 64, M*N) or (batch, 64*M*N)
               - audio_batch: shape (batch, M*N) or (batch, 1*M*N)

        Returns:
            Tuple of (encoded_eeg, encoded_audio)
            - encoded_eeg: shape (batch, embedding_size)
            - encoded_audio: shape (batch, embedding_size)
        """
        # Reshape EEG from flat or (64, M*N) to proper 2D convolution format
        # Shape transformation: (batch, 64*M*N) or (batch, 64, M*N) → (batch, 64, M, N)
        # -1 for batch size allows variable batch sizes
        eeg = torch.reshape(x[0], (-1, 64, self.M, self.N))

        # Reshape audio to 2D convolution format with single channel
        # Shape transformation: (batch, M*N) → (batch, 1, M, N)
        audio = torch.reshape(x[1], (-1, 1, self.M, self.N))

        # Encode EEG and audio through their respective pathways
        # Both outputs have shape (batch, embedding_size)
        encoded_eeg = self.eeg_encoder(eeg)
        encoded_audio = self.audio_encoder(audio)

        return encoded_eeg, encoded_audio


class GaborEncoderRaw(nn.Module):
    """
    1D Convolutional Siamese Network for raw signal encoding.

    Variant of GaborEncoder that uses 1D convolutions instead of 2D. This is designed
    for processing raw time-domain signals or flattened Gabor representations where
    the time-frequency structure is not explicitly preserved.

    Comparison to GaborEncoder:
        - GaborEncoder: Uses Conv2d to preserve 2D time-frequency structure
        - GaborEncoderRaw: Uses Conv1d for sequential processing

    Architecture:
        Similar to GaborEncoder but with 1D convolutions:
        - EEG: Conv1d(64→1) → Tanh → Flatten → Linear(1024→embedding)
        - Audio: Conv1d(1→1) → Tanh → Flatten → Linear(1024→embedding)

    Args:
        embedding_size (int): Dimension of output embedding space

    Example:
        >>> model = GaborEncoderRaw(embedding_size=128)
        >>> eeg_emb, audio_emb = model([eeg_batch, audio_batch])
    """

    def __init__(self, embedding_size):
        """
        Initialize 1D Siamese Encoder for raw signals.

        :param embedding_size: Dimension of output embedding space
        """
        super().__init__()
        # EEG Encoder: 1D convolution for sequential processing
        self.eeg_encoder = nn.Sequential(
            # Channel reduction: 64 EEG channels → 1 feature sequence
            nn.Conv1d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Flatten(),  # Flatten to (batch, 1024)
            nn.Linear(1024, embedding_size),  # Project to embedding space
        )

        # Audio Encoder: 1D convolution for audio sequence
        self.audio_encoder = nn.Sequential(
            # Single-channel audio feature extraction
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(1024, embedding_size),
        )

    def forward(self, x):
        """
        Forward pass through 1D Siamese network.

        Args:
            x: Tuple of (eeg_batch, audio_batch) with 1D sequential format

        Returns:
            Tuple of (encoded_eeg, encoded_audio), both shape (batch, embedding_size)
        """
        # Process EEG and audio through their respective 1D encoders
        encoded_eeg = self.eeg_encoder(x[0])
        encoded_audio = self.audio_encoder(x[1])

        return encoded_eeg, encoded_audio
    

class GaborBaseline(nn.Module):
    """
    Siamese Encoder with Dilated Convolutions for Multi-Scale Feature Extraction.

    This architecture uses dilated convolutions to capture features at multiple
    scales without increasing the number of parameters or reducing spatial resolution.
    Dilated convolutions expand the receptive field by inserting gaps between
    kernel elements, allowing the network to see broader context.

    Why Dilated Convolutions?
        - Dilation=1: Standard convolution, local 3×3 receptive field
        - Dilation=2: Receptive field expands to 5×5 without extra parameters
        - Dilation=3: Receptive field expands to 7×7
        This multi-scale approach helps capture both fine-grained and coarse patterns
        in the time-frequency representation.

    Architecture Diagram:

        EEG Pathway:
            Input: (batch, 64, M, N)
                ↓
            Conv2d(64→8, k=3×3, pad=1) - Channel reduction
            ReLU
                ↓
            Conv2d(8→16, k=3×3, dilation=1) - Standard receptive field
            ReLU
                ↓
            Conv2d(16→16, k=3×3, dilation=2) - Expanded receptive field (5×5 effective)
            ReLU
                ↓
            Conv2d(16→16, k=3×3, dilation=3) - Further expanded field (7×7 effective)
            ReLU
                ↓
            Flatten - Output: (batch, 16*M'*N') where M',N' depend on dilations
                ↓
            Output: Encoded features

        Audio Pathway:
            Similar architecture but starts with 1 input channel instead of 64

    Args:
        n_hiddens (int): Legacy parameter, not used
        embedding_size (int): Legacy parameter, not used in current implementation
        M (int): Gabor time dimension
        N (int): Gabor frequency dimension
        n_channels (int): Number of EEG channels, default 64

    Note:
        This model doesn't explicitly project to a fixed embedding_size like
        GaborEncoder. The output size depends on M, N, and the dilated conv operations.
    """

    def __init__(self, n_hiddens, embedding_size, M, N, n_channels=64):
        """
        Initialize Dilated Convolution Siamese Encoder.

        :param n_hiddens: Legacy parameter (not used in current implementation)
        :param embedding_size: Legacy parameter (not used in current implementation)
        :param M, N: Values of M and N in Gabor transform
        :param n_channels: Number of EEG channels used. Default is 64.
        """
        super().__init__()
        self.M = M  # Store for reshaping in forward pass
        self.N = N
        self.flatten = nn.Flatten()

        # EEG Encoder: Multi-scale feature extraction via dilated convolutions
        self.eeg_encoder = nn.Sequential(
            # Initial channel reduction: 64 → 8
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            # Scale 1: Standard convolution (dilation=1), receptive field = 3×3
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, dilation=1),
            nn.ReLU(),
            # Scale 2: Dilated convolution (dilation=2), effective receptive field = 5×5
            # Captures medium-range patterns without spatial downsampling
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, dilation=2),
            nn.ReLU(),
            # Scale 3: More dilated (dilation=3), effective receptive field = 7×7
            # Captures broader context and longer-range dependencies
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, dilation=3),
            nn.ReLU(),
            nn.Flatten()  # Flatten to 1D feature vector
        )

        # Audio Encoder: Same multi-scale strategy but for single-channel audio
        self.audio_encoder = nn.Sequential(
            # No initial channel reduction needed (already 1 channel)
            # Directly apply multi-scale dilated convolutions
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, dilation=3),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, x):
        """
        Forward pass through dilated convolution Siamese network.

        Args:
            x: Tuple of (eeg_batch, audio_batch)

        Returns:
            Tuple of (encoded_eeg, encoded_audio) with variable output sizes
        """
        # Reshape inputs to 2D convolution format
        # Shape: (batch, 64, M, N) for EEG, (batch, 1, M, N) for audio
        eeg = torch.reshape(x[0], (-1, 64, self.M, self.N))
        audio = torch.reshape(x[1], (-1, 1, self.M, self.N))

        # Encode through respective multi-scale pathways
        encoded_eeg = self.eeg_encoder(eeg)
        encoded_audio = self.audio_encoder(audio)

        return encoded_eeg, encoded_audio


# Gabor model based on challenge baseline model with 1D dilated convolutions
class GaborBaseline1D(nn.Module):
    def __init__(self, n_hiddens, embedding_size, M, N, n_channels=64):
        """
        Initialize Gabor Encoder (Siamese Neural Network architecture)
        
        :param n_hiddens: Number of nodes in hidden layer
        :param M, N: Values of M and N in Gabor transform
        :param n_channels: Number of EEG channels used. Default is 64.
        """
        super().__init__()
        self.M = M
        self.N = N
        self.flatten = nn.Flatten()
        self.eeg_encoder = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=3),
            nn.ReLU(),
            nn.Flatten()
        )

        self.audio_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=3),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, x):
        eeg = torch.reshape(x[0], (-1, 64, 320))
        audio = torch.reshape(x[1], (-1, 1, 320))
        encoded_eeg = self.eeg_encoder(eeg)
        encoded_audio = self.audio_encoder(audio)

        print(encoded_eeg.shape, encoded_audio.shape)

        return encoded_eeg, encoded_audio


class GaborEncoder2(nn.Module):
    def __init__(self, n_hiddens, embedding_size, M, N, n_channels=64):
        """
        Initialize Gabor Encoder (Siamese Neural Network architecture)
        
        :param n_hiddens: Number of nodes in hidden layer
        :param M, N: Values of M and N in Gabor transform
        :param n_channels: Number of EEG channels used. Default is 64.
        """
        super().__init__()
        self.M = M
        self.N = N
        self.flatten = nn.Flatten()
        self.eeg_encoder = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(3, 3), stride=2, padding=1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(256, 64),
        )

        self.audio_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=2, padding=1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(256, 64),
        )

    def forward(self, x):
        eeg = torch.reshape(x[0], (-1, 64, self.M, self.N))
        audio = torch.reshape(x[1], (-1, 1, self.M, self.N))
        encoded_eeg = self.eeg_encoder(eeg)
        encoded_audio = self.audio_encoder(audio)

        return encoded_eeg, encoded_audio

class GaborLSTM(nn.Module):
    """
    LSTM-based Siamese Encoder for Sequential Processing of Gabor-transformed Signals.

    This architecture treats the Gabor-transformed data as a sequence, using LSTM
    (Long Short-Term Memory) to capture temporal dependencies across frequency bins.
    The key insight is that the frequency bins (N=64) can be treated as a sequence,
    with each time step having M=16 features.

    Why LSTM for Gabor Data?
        - Gabor transform produces time-frequency representations
        - LSTM can model dependencies between adjacent frequency bins
        - Sequential processing captures spectral patterns
        - Bidirectional option allows looking at both low→high and high→low freq

    Architecture Design:

        Data Reshaping Strategy:
            Input: Flattened Gabor coefficients (M*N = 1024 values)
            Reshape to: (N, M) = (64, 16)
            Interpretation: 64 sequence steps, each with 16 features

        EEG Pathway:
            Input: (batch, M*N) = (batch, 1024)
                ↓
            Reshape: (batch, N, M) = (batch, 64, 16)
            Treat as sequence of 64 time steps with 16 features each
                ↓
            LSTM(input_size=M, hidden_size=embedding_size, num_layers)
            Processes sequence and maintains hidden state across steps
                ↓
            Extract final hidden state: encoded_eeg[-1]
            Output: (batch, embedding_size)

        Audio Pathway:
            Identical architecture to EEG pathway

    Args:
        embedding_size (int): Dimension of LSTM hidden state and output embedding
        num_layers (int): Number of stacked LSTM layers (typically 1 or 2)
        M (int): Gabor time dimension (becomes sequence feature dimension)
        N (int): Gabor frequency dimension (becomes sequence length)
        bidirectional (bool): If True, processes sequence in both directions

    Example:
        >>> model = GaborLSTM(embedding_size=128, num_layers=2, M=16, N=64)
        >>> eeg_emb, audio_emb = model([eeg_batch, audio_batch])
        >>> print(eeg_emb.shape)  # (batch, 128)
    """

    def __init__(self, embedding_size, num_layers, M, N, bidirectional=False):
        super().__init__()
        # Store architecture parameters
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.M = M  # Becomes LSTM input_size (features per time step)
        self.N = N  # Becomes sequence length

        # EEG LSTM: Processes frequency bins as a sequence
        # input_size=M: Each sequence step has M features
        # hidden_size=embedding_size: Output embedding dimension
        # batch_first=True: Input shape is (batch, seq_len, features)
        self.eeg_lstm = LSTM(M, embedding_size, num_layers, batch_first=True, bidirectional=bidirectional)

        # Audio LSTM: Identical architecture to EEG LSTM
        self.audio_lstm = LSTM(M, embedding_size, num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        """
        Forward pass through LSTM-based Siamese network.

        Args:
            x: Tuple of (eeg_batch, audio_batch)
               Each has shape (batch, M*N) or (batch, M, N)

        Returns:
            Tuple of (encoded_eeg, encoded_audio)
            Both have shape (batch, embedding_size)
        """
        # Reshape EEG from flat to sequence format
        # Shape: (batch, M*N) → (batch, N, M)
        # Interpretation: batch of sequences with N=64 steps, M=16 features per step
        eeg = x[0].view(-1, self.N, self.M)

        # Reshape audio similarly
        audio = x[1].view(-1, self.N, self.M)

        # Process sequences through LSTM
        # LSTM returns: (output, (hidden_state, cell_state))
        # We only need the final hidden state for the embedding
        # hidden_state shape: (num_layers, batch, embedding_size)
        _, (encoded_eeg, _) = self.eeg_lstm(eeg)
        _, (encoded_audio, _) = self.audio_lstm(audio)

        # Extract final layer's hidden state
        # encoded_eeg[-1] gets the last layer's hidden state
        # Shape: (batch, embedding_size)
        # view(-1, embedding_size) ensures correct output shape
        return encoded_eeg[-1].view(-1, self.embedding_size), encoded_audio[-1].view(-1, self.embedding_size)


class GaborConvLSTM(nn.Module):
    def __init__(self, embedding_size, num_layers, M, N, bidirectional=False):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.M = M
        self.N = N
        self.conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(3, 3), padding=1)
        self.eeg_lstm   = LSTM(M, embedding_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.audio_lstm = LSTM(M, embedding_size, num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        eeg     = self.conv(x[0].view(-1, 64, self.M, self.N))
        eeg     = torch.transpose(eeg.view(-1, self.M, self.N), dim0=1, dim1=2)
        audio   = torch.transpose(x[1].view(-1, self.M, self.N), dim0=1, dim1=2)
        _, (encoded_eeg,    _)  = self.eeg_lstm(eeg)
        _, (encoded_audio,  _)  = self.audio_lstm(audio)

        return encoded_eeg[-1].view(-1, self.embedding_size), encoded_audio[-1].view(-1, self.embedding_size)



class GaborRecurrent(nn.Module):
    def __init__(self, embedding_size, M, N):
        """
        Initialize Gabor Encoder (Siamese Neural Network architecture)
        
        :param n_hiddens: Number of nodes in hidden layer
        :param M, N: Values of M and N in Gabor transform
        :param n_channels: Number of EEG channels used. Default is 64.
        """
        super().__init__()
        self.M = M
        self.N = N
        self.embedding_size = embedding_size
        self.eeg_rnn = RNN(M, embedding_size, 1, batch_first=True)
        self.audio_rnn = RNN(M, embedding_size, 1, batch_first=True)

    def forward(self, x):
        eeg = torch.transpose(x[0].view(-1, self.M, self.N), dim0=1, dim1=2)
        audio = torch.transpose(x[1].view(-1, self.M, self.N), dim0=1, dim1=2)
        _, encoded_eeg = self.eeg_rnn(eeg)
        _, encoded_audio = self.audio_rnn(audio)

        return encoded_eeg.view(-1, self.embedding_size), encoded_audio.view(-1, self.embedding_size)


class GaborChannelDataset(Dataset):
    def __init__(self, mat_file, channels, M=16, N=64, train=True, train_prop=0.9, data_prop=1):
        """
        Initializes instance of GaborDataset, used to load particular channels 
        of 'clips_uniform.mat' into pytorch

        :param mat_file:    path to .mat file, e.g. 'gabor_results.mat'
        :channels:          EEG channel to load (1-indexing)
        :param M, N:        Parameters for Gabor transform
        :param train:       True to load training data, False to load testing data
        :param train_prop:  Proportion of dataset used as training data.
        """
        # Calculate size of training data
        all_clips = sio.loadmat(mat_file)['results']
        train_size = int(data_prop * train_prop * len(all_clips['results'][0, 0]))
        test_size = ceil(data_prop * (1 - train_prop) * len(all_clips['results'][0, 0]))

        self.channels = channels

        # Load MAT file
        if train:
            self.clips = all_clips['results'][0, 0][:train_size]
            self.labels = all_clips['ans'][0, 0][0][:train_size]
        else:
            self.clips = all_clips['results'][0, 0][train_size:train_size + test_size]
            self.labels = all_clips['ans'][0, 0][0][train_size:train_size + test_size]

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
        answer = self.labels[clip_idx]
        audio_idx = idx % 5

        # Assuming last 5 rows correspond to 5 audios
        sample = (torch.tensor(clip[[self.channels - 1]], dtype=torch.float32), 
                  torch.tensor(clip[-5 + audio_idx], dtype=torch.float32))
        # 0 for mismatch, 1 for match
        label = int(answer == audio_idx + 1)
        
        return sample, label


class SLP(nn.Module):
    """
    Single Layer Perceptron for Direct EEG-Audio Classification.

    Unlike the Siamese encoders, this architecture concatenates EEG and audio signals
    and performs direct binary classification (match/no-match). It's the simplest
    baseline: flatten everything, concatenate, and pass through 1 hidden layer.

    Architecture:
        EEG (64×M×N) + Audio (M×N) → Flatten → Concat → Linear(n_hiddens) → ReLU → Linear(1) → Sigmoid

    Args:
        n_hiddens: Number of hidden units
        M, N: Gabor dimensions
        n_channels: Number of EEG channels (default 64)
    """

    def __init__(self, n_hiddens, M, N, n_channels=64):
        """Initialize single-layer perceptron."""
        super().__init__()
        self.flatten = nn.Flatten()
        # Direct classification: concatenated features → hidden → output
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(M*N*(n_channels + 1), n_hiddens),  # Input: EEG + audio concatenated
            nn.ReLU(),
            nn.Linear(n_hiddens, 1),  # Binary output
            nn.Sigmoid(),  # Output probability in [0, 1]
        )

    def forward(self, x):
        """
        Forward pass: concatenate EEG and audio, classify directly.

        Args:
            x: Tuple of (eeg, audio)

        Returns:
            Probability of match, shape (batch, 1)
        """
        # Flatten EEG and audio
        eeg = self.flatten(torch.transpose(x[0], dim0=1, dim1=2))
        audio = self.flatten(x[1])
        # Concatenate features and classify
        X = torch.concat((eeg, audio), dim=1)
        logits = self.linear_relu_stack(X)

        return logits


class DLP(nn.Module):
    """
    Double Layer Perceptron for Direct EEG-Audio Classification.

    Extends SLP with an additional hidden layer for more representational capacity.
    Uses Tanh activation (range [-1, 1]) instead of ReLU for the hidden layers.

    Architecture:
        Concat(EEG, Audio) → Linear(n_hiddens0) → Tanh → Linear(n_hiddens1) → Tanh → Linear(1) → Sigmoid

    Comparison to SLP:
        - SLP: 1 hidden layer with ReLU
        - DLP: 2 hidden layers with Tanh
        - More parameters allow learning more complex decision boundaries

    Args:
        n_hiddens0: Number of units in first hidden layer
        n_hiddens1: Number of units in second hidden layer
        M, N: Gabor dimensions
        n_channels: Number of EEG channels (default 64)
    """

    def __init__(self, n_hiddens0, n_hiddens1, M, N, n_channels=64):
        """Initialize two-layer perceptron."""
        super().__init__()
        self.flatten = nn.Flatten()
        # Two-layer classification network with Tanh activations
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(M*N*(n_channels + 1), n_hiddens0),  # First hidden layer
            nn.Tanh(),  # Tanh activation: range [-1, 1]
            nn.Linear(n_hiddens0, n_hiddens1),  # Second hidden layer
            nn.Tanh(),
            nn.Linear(n_hiddens1, 1),  # Binary output
            nn.Sigmoid(),  # Probability output
        )

    def forward(self, x):
        """
        Forward pass: concatenate EEG and audio, classify through 2 hidden layers.

        Args:
            x: Tuple of (eeg, audio)

        Returns:
            Probability of match, shape (batch, 1)
        """
        # Flatten and concatenate features
        eeg = self.flatten(torch.transpose(x[0], dim0=1, dim1=2))
        audio = self.flatten(x[1])
        X = torch.concat((eeg, audio), dim=1)
        # Pass through two-layer network
        logits = self.linear_tanh_stack(X)

        return logits


class ConvMLP(nn.Module):
    def __init__(self, n_hiddens, M, N):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=65, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        # Calculate the flattened size after convolutions and pooling
        conv_output_size = 128 * (M // 8) * (N // 8)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(conv_output_size, n_hiddens),  # Adding the size of flattened audio input
            nn.Tanh(),
            nn.Linear(n_hiddens, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        eeg = x[0]
        audio = x[1].unsqueeze(1)
        # print(f"Shape of EEG:\t{eeg.shape}")
        # print(f"Shape of Audio:\t{audio.shape}")
        X = torch.concat((eeg, audio), dim=1)
        # print(f"Shape of X:\t{X.shape}")
        X = X.view(X.size(0), X.size(1), 16, 64)
        # print(f"Shape of X:\t{X.shape}")
        logits = self.conv_layers(X)
        # print(f"Shape of logits:\t{logits.shape}")
        logits = self.flatten(logits)
        # print(f"Shape of logits:\t{logits.shape}")
        logits = self.linear_relu_stack(logits)
        return logits


class EmbedMLP(nn.Module):
    def __init__(self, n_eeg, n_audio, n_hiddens, M=16, N=64, n_channels=64):
        """
        Initialize the following neural network architecture:

        EEG Input (64xMxN) -> EEG Feature Extraction Layer (n_eeg)     
                                                                        => Hidden Layer (n_hiddens) -> Output (1)
        Audio Input (MxN)  -> Audio Feature Extraction Layer (n_audio) 
        
        :param n_eeg: Number of nodes in EEG FE layer
        :param n_audio: Number of nodes in audio FE layer
        :param n_hiddens: Number of nodes in hidden layer
        :param M, N: Values of M and N in Gabor transform
        :param n_channels: Number of EEG channels used. Default is 64.
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.eeg_stack = nn.Sequential(
            nn.Linear(M*N*n_channels, n_eeg),
            nn.Tanh(),
        )
        self.audio_stack = nn.Sequential(
            nn.Linear(M*N, n_audio),
            nn.Tanh(),
        )
        self.total_stack = nn.Sequential(
            nn.Linear(n_eeg + n_audio, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        eeg = self.flatten(torch.transpose(x[0], dim0=1, dim1=2))
        audio = self.flatten(x[1])

        eeg_features = self.eeg_stack(eeg)
        audio_features = self.audio_stack(audio)
        X = torch.concat((eeg_features, audio_features), dim=1)

        return self.total_stack(X)


class CosMLP(nn.Module):
    """
    MLP-based Encoder with Cosine Similarity Matching.

    This architecture learns feature representations for EEG and audio, then uses
    cosine similarity as the matching metric instead of a learned classifier.
    Cosine similarity measures the angle between vectors, making it insensitive to
    magnitude and focusing purely on directional similarity.

    Why Cosine Similarity?
        - Scale-invariant: Ignores magnitude, focuses on pattern similarity
        - Natural for embeddings: Matches if vectors point in same direction
        - Range [-1, 1]: -1 = opposite, 0 = orthogonal, 1 = identical direction
        - Common in metric learning and contrastive approaches

    Architecture Diagram:

        EEG Input (n_channels×M×N)
            ↓
        Flatten → (batch, n_channels*M*N)
            ↓
        Linear(n_channels*M*N → n_hiddens) - Feature extraction
        ReLU
            ↓
        EEG Features (batch, n_hiddens)
                                             \
                                              → Cosine Similarity → Output (batch,)
                                             /
        Audio Input (M×N)
            ↓
        Flatten → (batch, M*N)
            ↓
        Linear(M*N → n_hiddens) - Feature extraction
        ReLU
            ↓
        Audio Features (batch, n_hiddens)

        Cosine Similarity: cos(θ) = (eeg · audio) / (||eeg|| * ||audio||)

    Args:
        n_hiddens (int): Dimension of feature space for both EEG and audio
        M (int): Gabor time dimension, default 16
        N (int): Gabor frequency dimension, default 64
        n_channels (int): Number of EEG channels to process, default 1
                         (Note: Despite default=1, can be used with 64 channels)

    Attributes:
        flatten (nn.Flatten): Flattens input tensors
        eeg_stack (nn.Sequential): EEG feature extraction (linear + ReLU)
        audio_stack (nn.Sequential): Audio feature extraction (linear + ReLU)
        metric (nn.CosineSimilarity): Computes cosine similarity between features

    Example:
        >>> model = CosMLP(n_hiddens=512, M=16, N=64, n_channels=64)
        >>> similarity = model([eeg_batch, audio_batch])
        >>> print(similarity.shape)  # (batch,) with values in [-1, 1]
        >>> matches = (similarity > 0.5).float()  # Threshold for classification
    """

    def __init__(self, n_hiddens, M=16, N=64, n_channels=1):
        """
        Initialize CosMLP with feature extractors and cosine similarity metric.

        :param n_hiddens: Number of nodes in feature extraction layers
        :param M, N: Values of M and N in Gabor transform
        :param n_channels: Number of EEG channels used. Default is 1.
        """
        super().__init__()
        self.flatten = nn.Flatten()

        # EEG Feature Extraction: Linear projection + ReLU activation
        # Maps EEG input to n_hiddens-dimensional feature space
        self.eeg_stack = nn.Sequential(
            nn.Linear(M*N*n_channels, n_hiddens),
            nn.ReLU(),  # Non-linear activation
        )
        # Optional identity initialization (commented out)
        # Would initialize weights as identity matrix for specific experiments
        # self.eeg_stack[0].weight.data.copy_(torch.eye(n_hiddens))

        # Audio Feature Extraction: Same architecture as EEG
        # Projects audio to same n_hiddens-dimensional space
        self.audio_stack = nn.Sequential(
            nn.Linear(M*N, n_hiddens),
            nn.ReLU(),
        )
        # Optional identity initialization (commented out)
        # self.audio_stack[0].weight.data.copy_(torch.eye(n_hiddens))

        # Cosine Similarity: Computes cos(angle) between EEG and audio features
        # Output range: [-1, 1] where 1 = perfectly aligned, -1 = opposite
        self.metric = nn.CosineSimilarity()

    def forward(self, x):
        """
        Forward pass: Extract features and compute cosine similarity.

        Args:
            x: Tuple of (eeg_batch, audio_batch)
               - eeg_batch: shape (batch, n_channels, M, N)
               - audio_batch: shape (batch, M, N)

        Returns:
            torch.Tensor: Cosine similarity scores, shape (batch,)
                         Values in range [-1, 1]
        """
        # Flatten and transpose EEG to correct shape
        # Transpose swaps dimensions before flattening for proper layout
        eeg = self.flatten(torch.transpose(x[0], dim0=1, dim1=2))

        # Flatten audio
        audio = self.flatten(x[1])

        # Extract features through respective pathways
        # Both outputs have shape (batch, n_hiddens)
        eeg_features = self.eeg_stack(eeg)
        audio_features = self.audio_stack(audio)

        # Compute cosine similarity between corresponding EEG-audio pairs
        # Output shape: (batch,) with one similarity score per sample
        return self.metric(eeg_features, audio_features)


class ConvMLP2(nn.Module):
    def __init__(self, M, N):
        """
        Initialize MLP with convolution
        
        :param n_hiddens: Number of nodes in hidden layer
        :param M, N: Values of M and N in Gabor transform
        :param n_channels: Number of EEG channels used. Default is 64.
        """
        super().__init__()
        self.M = M
        self.N = N
        self.flatten = nn.Flatten()
        self.audio_conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.eeg_conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.linear_stack = nn.Linear(16640, 1)

    def forward(self, x):
        eeg = torch.reshape(x[0], (-1, 64, self.M, self.N))
        audio = torch.reshape(x[1], (-1, 1, self.M, self.N))

        conv_eeg = self.flatten(self.eeg_conv_stack(eeg))
        conv_audio = self.flatten(self.audio_conv_stack(audio))

        return self.linear_stack(torch.cat((conv_eeg, conv_audio), dim=1))


class GaborBaselineFull(nn.Module):
    def __init__(self, n_hiddens, embedding_size, M, N, n_channels=64):
        """
        Initialize Gabor Encoder (Siamese Neural Network architecture)
        
        :param n_hiddens: Number of nodes in hidden layer
        :param M, N: Values of M and N in Gabor transform
        :param n_channels: Number of EEG channels used. Default is 64.
        """
        super().__init__()
        self.M = M
        self.N = N
        self.flatten = nn.Flatten()
        self.eeg_encoder = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, dilation=3),
            nn.ReLU(),
        )

        self.audio_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, dilation=3),
            nn.ReLU(),
        )

        # self.dot
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.linear_stack = nn.Sequential(
            nn.Linear(80, 5),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        eeg = torch.reshape(x[0], (-1, 64, self.M, self.N))
        audio = torch.reshape(x[1], (-1, 1, self.M, self.N))

        encoded_eeg = self.eeg_encoder(eeg)
        encoded_audio = self.audio_encoder(audio)

        encoded_eeg = torch.reshape(encoded_eeg, (encoded_eeg.size(0), 16, -1))
        encoded_audio = torch.reshape(encoded_audio, (encoded_audio.size(0), 16, -1))
        encoded_audio = torch.reshape(encoded_audio, (-1, 5, 16, encoded_audio.size(2)))

        cos_sim = []
        for i in range(5):
            cos_sim.append(self.cos_sim(encoded_eeg, encoded_audio[:, i, :, :]))     

        # print(f"Shape of Cosine Similarity:\t{cos_sim[0].shape}")
        cos_sim = torch.cat(cos_sim, dim=1)

        # print(f"Shape of Cosine Similarity:\t{cos_sim.shape}")

        # cos_sim = torch.reshape(cos_sim, (-1, 16))
        logits = self.linear_stack(cos_sim)

        return logits

