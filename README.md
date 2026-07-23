Audio Classification Using Hugging Face Transformers

Introduction

Audio classification is an important application of Artificial Intelligence (AI) that enables computers to recognize and categorize sounds automatically. It is widely used in smart surveillance, wildlife monitoring, voice assistants, healthcare, and multimedia analysis. In this project, a pre-trained Audio Spectrogram Transformer (AST) model from Hugging Face is used to classify environmental sounds. The system downloads an audio file, preprocesses it, extracts features, and predicts the most probable sound classes along with confidence scores.




Project Overview

This project demonstrates how transformer-based deep learning models can be applied to audio classification without training a model from scratch. The MIT Audio Spectrogram Transformer (AST) model, fine-tuned on the AudioSet dataset, is used to analyze audio and identify sounds.

The workflow includes:

Downloading an audio file.

Loading a pre-trained model.

Preprocessing the audio.

Extracting audio features.

Predicting sound classes.

Displaying the top predictions.




Why This Project?

Demonstrates practical use of AI in audio processing.

Eliminates the need for custom model training.

Provides accurate predictions using transfer learning.

Easy to implement using pre-trained models.

Suitable for academic projects, research, and real-world applications.





Features

Automatic audio file download.

Uses a pre-trained Hugging Face AST model.

Supports audio preprocessing with Torchaudio.

Resamples audio to 16 kHz.

Predicts Top-5 sound classes.

Displays confidence scores.

Works in Google Colab, Jupyter Notebook, and local environments.





Tech Stack

Technology	Purpose

Python	Programming Language
PyTorch	Deep Learning Framework
Torchaudio	Audio Processing
Transformers	Pre-trained Models
Requests	Download Audio Files
Hugging Face Hub	Model Repository





Quick Start

Clone Repository

git clone https://github.com/your-username/audio-classification.git
cd audio-classification

Install Dependencies

pip install transformers torch torchaudio requests

Run the Project

python audio_classification.py




Installation

Step 1: Install Python

Install Python 3.10 or above.

Step 2: Install Required Libraries

pip install transformers
pip install torch
pip install torchaudio
pip install requests




Usage Example

model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)

Load audio:

audio, sampling_rate = torchaudio.load("sample.wav")

Predict audio classes:

with torch.no_grad():
    outputs = model(**inputs)




Code Structure

audio-classification/
│
├── audio_classification.py
├── sample.wav
├── requirements.txt
├── README.md
└── output/
    └── predictions.txt

File Description

audio_classification.py → Main source code.

sample.wav → Input audio file.

requirements.txt → Required libraries.

README.md → Project documentation.

output/ → Stores prediction results.





API Used

Hugging Face Model API

Model Name:

MIT/ast-finetuned-audioset-10-10-0.4593

Libraries Used

from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification

Dataset Source

ESC-50 Audio Dataset

AudioSet Dataset




Source Code

import requests
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Download sample audio
url = "https://github.com/karolpiczak/ESC-50/raw/master/audio/1-100032-A-0.wav"
response = requests.get(url)

with open("sample.wav", "wb") as f:
    f.write(response.content)

# Load model
model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)

# Load audio
audio, sampling_rate = torchaudio.load("sample.wav")

# Resample if necessary
if sampling_rate != 16000:
    resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
    audio = resampler(audio)

# Feature extraction
inputs = feature_extractor(
    audio.squeeze().numpy(),
    sampling_rate=16000,
    return_tensors="pt"
)

# Prediction
with torch.no_grad():
    outputs = model(**inputs)

# Top predictions
logits = outputs.logits
probabilities = torch.nn.functional.softmax(logits, dim=1)

top_k = torch.topk(probabilities, k=5)

for score, idx in zip(top_k.values[0], top_k.indices[0]):
    label = model.config.id2label[idx.item()]
    print(f"{label}: {score.item():.4f}")




Error Handling

Common issues and solutions:

Error	Solution

ModuleNotFoundError	Install missing libraries
Audio file not found	Verify file path
Internet connection issue	Check network connectivity
Model loading failure	Verify model name
Unsupported audio format	Convert to WAV format


Example:

try:
    audio, sampling_rate = torchaudio.load("sample.wav")
except FileNotFoundError:
    print("Audio file not found.")




Contribution Guide

Contributions are welcome.

Steps:

1. Fork the repository.


2. Create a new branch.



git checkout -b feature-name

3. Commit changes.



git commit -m "Added new feature"

4. Push changes.



git push origin feature-name

5. Create a Pull Request.




Acknowledgements

[Hugging Face](https://huggingface.co/?utm_source=chatgpt.com) for providing pre-trained transformer models.

[PyTorch](https://pytorch.org/?utm_source=chatgpt.com) for the deep learning framework.

[Torchaudio](https://pytorch.org/audio/stable/?utm_source=chatgpt.com) for audio processing tools.

[MIT AST Model Repository](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593?utm_source=chatgpt.com) for the Audio Spectrogram Transformer model.

[ESC-50 Dataset](https://github.com/karolpiczak/ESC-50?utm_source=chatgpt.com) for providing environmental soundsamplesA


Conclusion

This project demonstrates the implementation of audio classification using the Hugging Face Audio Spectrogram Transformer (AST) model. By leveraging pre-trained deep learning models, the system accurately identifies environmental sounds without requiring additional model training. The combination of Python, PyTorch, Torchaudio, and Hugging Face Transformers makes the implementation simple, efficient, and easy to understand. This project serves as a strong foundation for developing intelligent audio recognition applications such as smart surveillance, wildlife monitoring, voice assistants, and multimedia analysis. It also highlights the effectiveness of transfer learning in building high-performance AI solutions with minimal development effort.