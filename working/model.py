# Imports

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import pickle as pkl

# Read in data
df_Mel = pd.read_csv(r'C:\Users\sylas\OneDrive\emo-db-project\working\Mel_Spec_features.csv')
df_Chr = pd.read_csv(r'C:\Users\sylas\OneDrive\emo-db-project\working\Chroma_STFT_features.csv')
df_MFCC = pd.read_csv(r'C:\Users\sylas\OneDrive\emo-db-project\working\MFCCs_features.csv')


# Extract features and labels
X = np.array(df_MFCC[['MFCC_1','MFCC_2','MFCC_3','MFCC_4','MFCC_5','MFCC_6','MFCC_7','MFCC_8','MFCC_9','MFCC_10','MFCC_11','MFCC_12','MFCC_13']].values)
y = np.array(df_MFCC['Emotion'].values)



'''
# Shuffle the dataset
indices = np.random.permutation(len(y))
X_padded, y = X_padded[indices], y[indices]

# Convert labels to categorical (one-hot encoding)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y_onehot = onehot_encoder.fit_transform(integer_encoded)

# Split the dataset into training and testing sets (80% train, 20% test)
split_idx = int(0.8 * len(y_onehot))
X_train, X_test = X_padded[:split_idx], X_padded[split_idx:]
y_train, y_test = y_onehot[:split_idx], y_onehot[split_idx:]

# Define the neural network architecture
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Initialize layers
input_size = X_train.shape[1] * X_train.shape[2]  # Number of features
dense1 = Layer_Dense(input_size, 4)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(4, 5)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(5, y_train.shape[1])
activation3 = Activation_Softmax()

# Forward pass through the network
X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten the input
dense1.forward(X_train_flat)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

# Output the shape of the final output to confirm correct dimensions
print("Output shape from the network:", activation3.output.shape)

# Debug: Output a few predictions
print("Sample predictions from the network:")
print(activation3.output[:5])
'''
