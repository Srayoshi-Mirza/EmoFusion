import os
import librosa
import csv
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

# Define augmentation transforms
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
])

def augment_audio(audio_data, sample_rate):
    # Apply augmentation transforms
    augmented_audio = augment(samples=audio_data, sample_rate=sample_rate)
    return augmented_audio

def convert_filename(filename, dataset_name):
    if dataset_name == 'SUBESCO':
        parts = filename.split('_')
        gender = parts[0]
        emotion = parts[-2]
        intensity = None  # Intensity information is not available in SUBESCO
    elif dataset_name == 'RAVDESS':
        parts = filename.split('-')
        emotion_dict = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
        gender = 'F' if parts[6].startswith('1') else 'M'  # Gender info based on the actor number
        emotion = emotion_dict[parts[2]]
        intensity = 'normal' if parts[3] == '01' else 'strong' if parts[3] == '02' else None

    return gender, emotion, intensity

def load_audio(file_path):
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    return audio_data, sample_rate

def process_dataset(dataset_folder, output_folder, dataset_name):
    metadata = []
    for file_name in os.listdir(dataset_folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(dataset_folder, file_name)
            gender, emotion, intensity = convert_filename(file_name, dataset_name)
            audio_data, sample_rate = load_audio(file_path)
            
            # Apply data augmentation
            augmented_audio = augment_audio(audio_data, sample_rate)
            
            # Save augmented audio using soundfile
            output_file_path = os.path.join(output_folder, file_name)
            try:
                sf.write(output_file_path, augmented_audio, sample_rate, format='wav')
                # Update metadata
                metadata.append((output_file_path, gender, emotion, intensity))
            except (PermissionError, FileNotFoundError) as e:
                print(f"Error writing file: {e}")
                continue
    
    # Save metadata to a CSV file
    csv_file_path = os.path.join(output_folder, f"{dataset_name}_metadata.csv")
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File Path', 'Gender', 'Emotion', 'Intensity'])
        writer.writerows(metadata)
    
    return metadata

output_folder_subesco = 'G:/Data Science Portfolio/Emotion & Sentimental Analysis/Augmented Datasets/SUBESCO'
output_folder_ravdess = 'G:/Data Science Portfolio/Emotion & Sentimental Analysis/Augmented Datasets/RAVDESS'

# Process SUBESCO dataset
subesco_folder = 'G:/Data Science Portfolio/Emotion & Sentimental Analysis/Datasets/SUBESCO'
subesco_metadata = process_dataset(subesco_folder, output_folder_subesco, 'SUBESCO')

# Process RAVDESS dataset
ravdess_folder = 'G:/Data Science Portfolio/Emotion & Sentimental Analysis/Datasets/RAVDESS'
ravdess_metadata = process_dataset(ravdess_folder, output_folder_ravdess, 'RAVDESS')

print("Metadata for SUBESCO dataset:")
for file_path, gender, emotion, intensity in subesco_metadata:
    print(f"File: {file_path}, Gender: {gender}, Emotion: {emotion}, Intensity: {intensity}")

print("\nMetadata for RAVDESS dataset:")
for file_path, gender, emotion, intensity in ravdess_metadata:
    print(f"File: {file_path}, Gender: {gender}, Emotion: {emotion}, Intensity: {intensity}")