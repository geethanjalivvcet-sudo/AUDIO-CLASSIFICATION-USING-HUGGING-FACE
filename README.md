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

logits = outputs.logits
probs = torch.nn.functional.softmax(logits, dim=1)

top_k = torch.topk(probs, k=5)

for score, idx in zip(top_k.values[0], top_k.indices[0]):
    print(model.config.id2label[idx.item()], score.item())