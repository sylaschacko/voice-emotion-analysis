# Emo-DB Emotion Recognition Project

## Overview
This project aims to build robust machine learning models for emotion recognition using the Berlin Database of Emotional Speech (Emo-DB) and additional datasets to enhance model performance. Our primary goal is to develop a reliable emotion classifier that can accurately detect and classify emotions from speech.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training and Hyperparameter Tuning](#training-and-hyperparameter-tuning)
- [Data Preprocessing](#data-preprocessing)
- [Results](#results)
- [Future Work](#future-work)
- [Contact](#contact)

## Project Description
Emotion recognition from speech is a challenging task that requires sophisticated data processing and robust model architecture. In this project, we designed and implemented three different neural network models to classify emotions from audio signals. Initially, we only used the Emo-DB dataset, which consists of 535 audio files. However, due to the limited dataset size, we are now integrating additional datasets—TESS, SAVEE, and RAVDESS—to increase the diversity and volume of data.

## Dataset
We utilize the following datasets:
- **Emo-DB** [Emo-DB](http://emodb.bilderbar.info/start.html)
- **TESS** [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
- **SAVEE** [SAVEE](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee)
- **RAVDESS** [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

Current features include:
- **MFCCs (Mel Frequency Cepstral Coefficients)**
- **Mel Spectrograms**
- **Chroma STFT (Short-Time Fourier Transform)**

We are working to add more features such as Zero-Crossing Rate, Tonal Centroid, Spectral Contrast, Root-Mean-Square, Constant-Q Chromagram, and Chroma Energy Normalized to improve the model's accuracy.

## Model Architecture
The core of our model is a custom neural network built from scratch using PyTorch. The architecture comprises:
- **Input Layer**: Accepts 153 features derived from the audio data.
- **Hidden Layers**: Includes two fully connected layers with ReLU activation and dropout layers to prevent overfitting.
- **Output Layer**: A softmax activation layer that outputs probabilities across different emotion classes.

```python
class EmotionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_prob=0.2):
        super(EmotionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return self.softmax(x)
```

## Training and Hyperparameter Tuning
The model is trained using PyTorch with the Adam optimizer and Cross-Entropy Loss. We employed Ray Tune for hyperparameter tuning, exploring different configurations for hidden layer size, learning rate, batch size, and number of epochs. Our training loop includes early stopping and batch normalization to enhance generalization and performance.

### Hyperparameters:
- **Hidden Layer Sizes**: [512, 640, 768, 896]
- **Learning Rate**: Log-uniformly sampled between \(1 \times 10^{-6}\) and \(5 \times 10^{-5}\)
- **Batch Sizes**: [8, 16, 24]
- **Epochs**: [30, 40, 50]

## Data Preprocessing
The data preprocessing pipeline involves:
1. **Feature Extraction**: Extracting MFCCs, Mel Spectrograms, and Chroma STFT features.
2. **Normalization**: Standardizing features to ensure consistent input for the model.
3. **One-Hot Encoding**: Converting categorical labels into numerical form suitable for model training.
4. **Data Augmentation**: Incorporating techniques like noise injection and time-stretching to enhance the dataset's variability.

The limited dataset size of Emo-DB is currently the primary factor impacting our accuracy. However, with the addition of more data from TESS, SAVEE, and RAVDESS, and the introduction of new features, we are actively working to improve the model's performance.

## Results
### Current Performance:
- **Training Accuracy**: 66.5%
- **Test Accuracy**: 55.5%

While these results reflect our initial efforts, we are confident that the integration of additional datasets and features will significantly boost our accuracy.

## Future Work
- **Feature Expansion**: Implementing Zero-Crossing Rate, Tonal Centroid, Spectral Contrast, and other advanced audio features.
- **Model Optimization**: Further fine-tuning hyperparameters and exploring deeper architectures to enhance classification accuracy.
- **Dataset Augmentation**: Continuously integrating more diverse datasets to provide a richer training environment for the model.

## Contact
We welcome any feedback or collaboration opportunities. Please feel free to reach out to us:

- **Sylas Chacko**: [sychacko@udel.edu](mailto:sychacko@udel.edu)
- **Ashley Chen**: [ashleychen908@gmail.com](mailto:ashleychen908@gmail.com)
- **Omari Motta**: [omotta223@gmail.com](mailto:omotta223@gmail.com)
