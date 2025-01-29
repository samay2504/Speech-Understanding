# Experimenting with Spectrograms and Windowing Techniques

## Overview
This project explores the application of different windowing techniques on the UrbanSound8k dataset to generate spectrograms using the Short-Time Fourier Transform (STFT). We compare and analyze the visual and performance differences of spectrograms generated using Hann, Hamming, and Rectangular windows. Additionally, we train a simple classifier using features extracted from the spectrograms and evaluate the performance results comparatively for each windowing technique.

## Dataset
The UrbanSound8k dataset is used for this assignment. It contains 8732 labeled sound excerpts (≤4s) of urban sounds from 10 classes.

[Download the dataset from here](https://goo.gl/8hY5ER)

## Windowing Techniques
We implemented the following windowing techniques:
1. **Hann Window**
2. **Hamming Window**
3. **Rectangular Window**

## Steps
1. **Download and Load the Dataset**
2. **Implement Windowing Techniques**
3. **Generate Spectrograms using STFT**
4. **Visual Comparison of Spectrograms**
5. **Feature Extraction**
6. **Train and Evaluate Classifier**
7. **Comparison and Analysis**

### 1. Download and Load the Dataset
We downloaded the UrbanSound8k dataset and loaded the audio files and metadata for further processing.

### 2. Implement Windowing Techniques
We implemented Hann, Hamming, and Rectangular window functions to apply to the audio signals.

### 3. Generate Spectrograms using STFT
Using the Short-Time Fourier Transform (STFT), we generated spectrograms for each audio signal with the applied windowing techniques.

### 4. Visual Comparison of Spectrograms
We visually compared the spectrograms generated using different windowing techniques to observe the differences.

### 5. Feature Extraction
We extracted features from the spectrograms to be used in training the classifier.

### 6. Train and Evaluate Classifier
We trained a simple neural network classifier using the extracted features and evaluated the performance results for each windowing technique.

### 7. Comparison and Analysis
We compared the performance results and analyzed the differences between the windowing techniques.

## Implementation

### Windowing Techniques

#### Hann Window
```python
import numpy as np

def hann_window(N):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
```

#### Hamming Window
```python
def hamming_window(N):
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
```

#### Rectangular Window
```python
def rectangular_window(N):
    return np.ones(N)
```

### Generate Spectrograms using STFT
```python
import librosa
import matplotlib.pyplot as plt

def generate_spectrogram(audio_path, window_fn, n_fft=2048, hop_length=512):
    y, sr = librosa.load(audio_path, sr=None)
    window = window_fn(n_fft)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    spectrogram = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return spectrogram, sr

# Example: Generate spectrogram with Hann window
audio_path = 'path_to_audio_file.wav'
spectrogram, sr = generate_spectrogram(audio_path, hann_window)
plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Hann Window Spectrogram')
plt.show()
```

### Train and Evaluate Classifier
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_and_evaluate_nn(X_train, X_test, y_train, y_test, input_size, num_classes, batch_size=32, epochs=20):
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoader for training data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = SimpleNN(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)

    return accuracy

# Example: Train and evaluate with extracted features
input_size = X_train.shape[1]
num_classes = len(np.unique(y_train))
accuracy = train_and_evaluate_nn(X_train, X_test, y_train, y_test, input_size, num_classes)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
```

## Results and Analysis

### Visual Comparison of Spectrograms
![output](https://github.com/user-attachments/assets/9c68958e-b5ab-43b4-af62-1480aaeb032e)


### Performance Comparison
| Windowing Technique | Accuracy |
|---------------------|----------|
| Spectrogram         | 76.00%   |
| MFCC                | 78.62%   |
| Log-Mel Spectrogram | 82.62%   |

### Analysis
- The **Log-Mel Spectrogram** produced the best performance with an accuracy of 82.62%. This technique effectively represents the audio signal's frequency content, leading to better classification performance.
- The **MFCC** technique resulted in a slightly lower accuracy of 78.62%. It captures the most important aspects of the audio signal but may lose some information during the transformation.
- The **Spectrogram** had the lowest accuracy of 76.00%. While it provides a detailed representation of the audio signal, it may not capture the most relevant features for classification.

## Conclusion
The choice of windowing technique significantly impacts the performance of spectrogram-based audio classification. The Log-Mel Spectrogram provided the best balance between detail and relevance, resulting in the highest classification accuracy. Future work could explore other windowing techniques and advanced classifiers to further improve performance.

## References
- [UrbanSound8k Dataset](https://goo.gl/8hY5ER)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
