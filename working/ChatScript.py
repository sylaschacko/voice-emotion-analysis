import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
import os
from pathlib import Path

# Load and display an example audio file
audio_files = librosa.util.find_files(r'C:\Users\sylas\OneDrive\emo-db-project\wav', ext=['wav'])
ipd.display(ipd.Audio(audio_files[0]))

# Load audio file and plot waveform
y, sr = librosa.load(audio_files[0])
plt.figure(figsize=(10, 5))
plt.plot(y, label='Waveform')
plt.title('Raw Audio Example')
plt.legend()
plt.show()

# Compute and plot Spectrogram
D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
plt.figure(figsize=(10, 5))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram Example')
plt.show()

# Compute and plot Mel Spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
plt.figure(figsize=(10, 5))
librosa.display.specshow(S_db_mel, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram Example')
plt.show()

# Read feature data
df_MFCC = pd.read_csv(r'C:\Users\sylas\OneDrive\emo-db-project\working\MFCCs_features.csv')
df_Mel = pd.read_csv(r'C:\Users\sylas\OneDrive\emo-db-project\working\Mel_Spec_features.csv').drop('Emotion', axis=1)
df_Chr = pd.read_csv(r'C:\Users\sylas\OneDrive\emo-db-project\working\Chroma_STFT_features.csv').drop('Emotion', axis=1)
df = pd.concat([df_MFCC, df_Mel, df_Chr], axis=1)

# Shuffle and split data
def shuffle_dataframe(df, block_size):
    chunks = [df.iloc[i:i + block_size] for i in range(0, len(df), block_size)]
    shuffled_chunks = [chunk.sample(frac=1).reset_index(drop=True) for chunk in chunks]
    return pd.concat(shuffled_chunks).reset_index(drop=True)

df_shuffled = shuffle_dataframe(df, 87)
train_size = int(0.8 * len(df_shuffled))
train_df = df_shuffled.iloc[:train_size]
test_df = df_shuffled.iloc[train_size:]

X_train = np.array(train_df.drop('Emotion', axis=1))
y_train = np.array(train_df['Emotion'])
X_test = np.array(test_df.drop('Emotion', axis=1))
y_test = np.array(test_df['Emotion'])

input_size = X_train.shape[1]  # Determine the number of input features

# Define layers and model with appropriate methods
from emodb import Layer_Dense, Activation_ReLU, Activation_Softmax, Loss_CategoricalCrossentropy, Accuracy_Categorical, Optimizer_SGD, Layer_Input

model = Model()
model.add(Layer_Dense(input_size, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 32))
model.add(Activation_ReLU())
model.add(Layer_Dense(32, 10))  # Assuming 10 different emotions
model.add(Activation_Softmax())

model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_SGD(learning_rate=0.1, decay=1e-3, momentum=0.9),
    accuracy=Accuracy_Categorical()
)

model.train(X_train, y_train, epochs=10, batch_size=128)
