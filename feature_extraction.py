import os
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

def process_dataset(metadata_path, output_csv):
    features = []
    labels = []
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)
    
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

    # Save features and labels to CSV
    save_features_to_csv(features, labels, output_csv)

def save_features_to_csv(features, labels, csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Label', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12', 'MFCC13'])  # Write header
        for feature, label in zip(features, labels):
            row = [label] + list(feature)
            writer.writerow(row)

# SUBESCO dataset
subesco_csv_file = r'G:\Data Science Portfolio\Emotion & Sentimental Analysis\Feature Extraction\subesco_features.csv'
process_dataset(subesco_metadata_path, subesco_csv_file)

# RAVDESS dataset
ravdess_csv_file = r'G:\Data Science Portfolio\Emotion & Sentimental Analysis\Feature Extraction\ravdess_features.csv'
process_dataset(ravdess_metadata_path, ravdess_csv_file)
