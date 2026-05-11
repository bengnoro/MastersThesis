# SYNTHESIS OF SOUND FOR APPLICATION IN  AN INTERACTIVE GAME FOR VISUALLY IMPAIRED USERS


This repository contains PyTorch implementation of textually conditioned Generative Adversarial Network (GAN) designed for synthesizing environmental audio (e.g., footsteps, gunshots, other Foley sounds). 

Originally developed for generating interactive game audio for visually impaired users. This system evaluates the gap between synthetic Mel-spectrograms and phase reconstruction algorithms. It specifically compares the performance of neural vocoders (NVIDIA BigVGAN) against mathematical estimation (Griffin-Lim).

## Installation

Clone the repository and install required dependencies. CLAP text encoder relies natively on the HuggingFace `transformers` library.

```bash
git clone [https://github.com/bengnoro/MastersThesis.git](https://github.com/bengnoro/MastersThesis.git)
cd MastersThesis
pip install torch torchvision torchaudio librosa pandas numpy matplotlib transformers
```
Note: NVIDIA BigVGAN must be installed manually into your environment. Refer to their official documentation for CUDA compatibility.

## Pre-trained Checkpoints & Data Paths
Due to file size constraints fully trained Generator checkpoints and the DCASE dataset are not hosted in this repository. To request access to the trained weights, please contact: hasan.norbert99@gmail.com

Config Note: This codebase was originally run in a Google Colab environment. File paths for datasets, checkpoints, and output directories (e.g., /content/drive/MyDrive/diplomka/...) are defined as variables within the scripts. You must update these target directories in the code to match your local or cloud environment before execution.

## Execution & Usage
This pipeline was built for evaluation, execution does not rely on command-line arguments. Hyperparameters, evaluation prompts, and truncation values are configured directly within the script headers.

 1. Training the Architecture
To initialize the Exponential Moving Average environment and start training, run:

```bash
python train.py
```
Checkpoints, telemetry and debug Mel-spectrograms are saved at epoch save interval

2. BigVGAN Inference & DSP Mitigation
To generate text-conditioned Mel-spectrograms and reconstruct the phase using BigVGAN, run:
```bash
python inference.py
```

3. Griffin-Lim Baseline Evaluation
To synthesize the exact same generation pipeline utilizing mathematical phase estimation for comparative analysis (e.g., Fréchet Audio Distance baselines), run:
```bash
python grffinLimInference.py
```

Repository File Structure:
- models.py: Definitions for the Generator, Critic.

- data_pipeline.py: Handles audio normalization and LAION-CLAP text encoding.

- train.py: Main adversarial training loop.

- inference.py: Pipeline for text to audio generation utilizing BigVGAN.

- grffinLimInference.py: Pipeline for text to audio generation utilizing Griffin-Lim.
