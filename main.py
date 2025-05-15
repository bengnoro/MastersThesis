import torch
import librosa
import numpy
import matplotlib
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)
x = torch.rand(5, 3)
print(x)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")