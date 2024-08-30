import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from glob import glob
import librosa 
import librosa.display
import IPython.display as ipd
import functools
from pathlib import Path
import pickle
import copy
import os
import sklearn.neural_network
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from functools import partial
import os
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray import train
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from ray.tune import CLIReporter


# Define your model as a class
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


# Function to load data and preprocess it
def load_data():
   df_Mel = pd.read_csv(r'C:\Users\sylas\OneDrive\Projects\emo-db-project\Mel_Spec_features.csv')
   df_Chr = pd.read_csv(r'C:\Users\sylas\OneDrive\Projects\emo-db-project\Chroma_STFT_features.csv')
   df_MFCC = pd.read_csv(r'C:\Users\sylas\OneDrive\Projects\emo-db-project\MFCCs_features.csv')


   df = pd.concat([df_MFCC, df_Mel.drop('Emotion', axis='columns'), df_Chr.drop('Emotion', axis='columns')], axis=1)


   X = df.iloc[:, 1:154].values
   y = df['Emotion'].values


   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


   labels_train, unique = pd.factorize(y_train)
   labels_test = pd.Series(y_test).map(lambda x: np.where(unique == x)[0][0]).values


   num_classes = len(unique)


   y_train_onehot = np.eye(num_classes)[labels_train]
   y_test_onehot = np.eye(num_classes)[labels_test]


   X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
   y_train_tensor = torch.tensor(y_train_onehot, dtype=torch.float32)


   X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
   y_test_tensor = torch.tensor(y_test_onehot, dtype=torch.float32)


   train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
   test_dataset = TensorDataset(X_test_tensor, y_test_tensor)


   return train_dataset, test_dataset, num_classes


# Define the training function
def train_emotion_classifier(config):
    train_dataset, test_dataset, num_classes = load_data()

    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=int(config["batch_size"]), shuffle=False)

    model = EmotionClassifier(input_size=153, hidden_size=int(config["hidden_size"]), num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(int(config["epochs"])):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate training accuracy
        model.eval()
        correct_train = 0
        total_train = 0

        with torch.no_grad():
            for inputs, labels in train_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(labels.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train

        # Test the model on the test set at the end of each epoch
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(labels.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_accuracy = 100 * correct_test / total_test

        # Report both accuracies to Ray Tune
        tune.report(train_accuracy=train_accuracy, accuracy=test_accuracy)

# Set the hyperparameter search space, scheduler, reporter, and run Ray Tune as before



# Set the hyperparameter search space
config = {
   "hidden_size": tune.choice([512, 640, 768, 896]),  # Focus on higher hidden sizes
   "lr": tune.loguniform(1e-6, 5e-5),  # Further narrow the learning rate range to fine-tune more delicately
   "batch_size": tune.choice([8, 16, 24]),  # Focus on smaller batch sizes which seem to perform better
   "epochs": tune.choice([30, 40, 50])  # Increase epochs range for potentially better training depth
}


# Set up the scheduler and reporter
scheduler = ASHAScheduler(
   metric="accuracy",
   mode="max",
   max_t=50,
   grace_period=1,
   reduction_factor=2
)


reporter = CLIReporter(
   metric_columns=["accuracy", "training_iteration"]
)


# Run Ray Tune
analysis = tune.run(
   train_emotion_classifier,
   resources_per_trial={"cpu": 1, "gpu": 0},
   config=config,
   num_samples=20,
   scheduler=scheduler,
   progress_reporter=reporter
)




# Output the best hyperparameters
best_config = analysis.best_config
print("Best hyperparameters found were: ", best_config)


# Re-train the model with the best configuration found
train_emotion_classifier(best_config)
