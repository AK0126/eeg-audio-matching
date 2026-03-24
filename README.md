# EEG-Audio Matching with Neural Networks

Code used for 2024 Summer Applied Math REU project at the University of Maryland. Comments / descriptions written by Claude

This project tackles a neural decoding problem: given an EEG recording of a subject listening to audio, and 5 audio candidates, identify which audio the subject was actually hearing. This is a 5-way ranking/classification task evaluated by whether the correct audio receives the highest predicted similarity score.

Three complementary approaches are implemented and compared: a raw signal baseline, a Gabor-transformed feature pipeline, and contrastive learning with siamese encoders. A final ensemble model combines the base approaches via stacking.

---

## Highlighted Files

### `mlp_gabor_encoder.py`
The most advanced training script. Implements contrastive learning using a siamese `GaborEncoder` architecture trained with InfoNCE loss. Rather than direct classification, the model learns an embedding space where EEG recordings and their matching audio are geometrically close. Evaluation includes top-K ranking accuracy across all 5 candidates.

### `gabor.py`
The core library module. Contains all PyTorch dataset loaders (with Gabor-transformed and raw variants) and 14+ neural network architectures, including `CosMLP`, `GaborEncoder`, `GaborBaseline`, and `GaborBaseline1D`. The starting point for understanding the data pipeline and model zoo.

### `stacking.py`
Ensemble meta-learning: loads pre-trained base models (raw signal and Gabor-transformed), generates cosine-similarity predictions from each, concatenates them into meta-features, and trains a logistic regression meta-learner to combine them. Demonstrates how heterogeneous models can be composed for improved performance.

### `nn_helpers.py`
Custom loss functions and evaluation metrics. Includes a vectorized multi-class cross-entropy using a matrix multiply trick, an InfoNCE wrapper, and `test_match_encoder_topk()` for computing cumulative top-K ranking accuracy.

### `GaborMLP.ipynb` / `RawSignalMLP.ipynb`
Documented notebooks walking through each approach end-to-end, with markdown cells explaining the Gabor transform, dataset structure, architecture design choices, and training loop mechanics.

---

## File Reference

| File | Description |
|---|---|
| `gabor.py` | Core module: Gabor-transform dataset loaders and 14+ neural network architectures |
| `nn_helpers.py` | Loss functions (multi-class CE, InfoNCE) and evaluation metrics (top-K ranking accuracy) |
| `mlp_gabor_encoder.py` | Contrastive learning training script: siamese encoder with InfoNCE loss |
| `mlp_gabor.py` | Direct classification training script: CosMLP on Gabor-transformed features |
| `mlp_raw.py` | Baseline training script: raw time-domain signals with no feature engineering |
| `GaborMLP.py` | Manual from-scratch MLP implementation (explicit weight matrices, no `nn.Module`) |
| `stacking.py` | Ensemble stacking: combines raw and Gabor base models via a meta-learner |
| `stacking_model.py` | Neural network meta-model architecture for use in the stacking pipeline |
| `raw.py` | Raw signal dataset loader and `RawSLP` model definition |
| `clips_uniform.py` | Alternative dataset loader returning all 5 audio candidates per sample (no 5x expansion) |
| `GaborMLP.ipynb` | Notebook: documented walkthrough of Gabor MLP training |
| `RawSignalMLP.ipynb` | Notebook: documented walkthrough of the raw signal baseline |
| `verification_report.txt` | Code integrity and documentation verification report |
