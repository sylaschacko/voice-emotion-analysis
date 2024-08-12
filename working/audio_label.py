import os
import pandas as pd

# Path to your audio files
audio_path = r'C:\Users\sylas\OneDrive\emo-db-project\wav'

# Mapping of emotion codes from filenames to emotion descriptions
emotion_map = {
    'W': 'anger',    # Wut
    'L': 'boredom',  # Langeweile
    'E': 'anxiety',  # Ekel
    'A': 'happiness', # Freude
    'T': 'sadness',  # Traurigkeit
    'F': 'disgust',  # Furcht
    'N': 'neutral'   # Neutral
}

# Mapping of speaker numbers to their details
speaker_map = {
    '03': 'male, 31 years old',
    '08': 'female, 34 years',
    '09': 'female, 21 years',
    '10': 'male, 32 years',
    '11': 'male, 26 years',
    '12': 'male, 30 years',
    '13': 'female, 32 years',
    '14': 'female, 35 years',
    '15': 'male, 25 years',
    '16': 'female, 31 years'
}

# Prepare to collect data
data = []

# Iterate over each file in the directory
for filename in os.listdir(audio_path):
    if filename.endswith('.wav'):
        # Extract components from the filename
        speaker = filename[:2]
        emotion_code = filename[5]

        # Map codes to descriptions
        emotion = emotion_map.get(emotion_code, 'Unknown')
        speaker_info = speaker_map.get(speaker, 'Unknown')

        # Append the details to the data list
        data.append([filename, speaker_info, emotion])

# Create a DataFrame
df = pd.DataFrame(data, columns=['Filename', 'Speaker Info', 'Emotion'])

# Save to CSV
csv_path = os.path.join(audio_path, 'audio_labels.csv')
df.to_csv(csv_path, index=False)

print(f'CSV file has been created at {csv_path}')
