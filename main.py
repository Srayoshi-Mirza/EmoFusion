import librosa
import numpy as np
import csv
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

##-Feature Extraction-###

# Define the file paths to the metadata CSV files
subesco_metadata_path = r'G:\Data Science Portfolio\Emotion & Sentimental Analysis\Augmented Datasets\SUBESCO\SUBESCO_metadata.csv'
ravdess_metadata_path = r'G:\Data Science Portfolio\Emotion & Sentimental Analysis\Augmented Datasets\RAVDESS\RAVDESS_metadata.csv'

def extract_features(file_path):
    # Load audio file
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    
    # Extract MFCCs (Mel-Frequency Cepstral Coefficients)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    
    # Calculate the mean of MFCCs along each feature dimension
    mfccs_mean = np.mean(mfccs, axis=1)
    
    return mfccs_mean

def process_dataset(metadata_path):
    features = []
    labels = []
    
    with open(metadata_path, 'r', newline='') as csvfile:
        metadata_reader = csv.reader(csvfile)
        next(metadata_reader)  # Skip header row
        for row in metadata_reader:
            file_path, gender, emotion, intensity = row
            # Extract features
            mfccs_mean = extract_features(file_path)
            features.append(mfccs_mean)
            # Use emotion as label
            labels.append(emotion)

    return np.array(features), np.array(labels)

# SUBESCO dataset
subesco_features, subesco_labels = process_dataset(subesco_metadata_path)

# RAVDESS dataset
ravdess_features, ravdess_labels = process_dataset(ravdess_metadata_path)
