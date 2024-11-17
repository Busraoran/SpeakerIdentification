import os
import librosa
import numpy as np

class Functions:
    @staticmethod
    def ExtractMfccFeatures(file_path, n_mfcc=13):
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean
    

    def PrepareDataset(data_dir, n_mfcc=13):
        features = []
        labels = []

        for speaker_id in os.listdir(data_dir):
            speaker_path = os.path.join(data_dir, speaker_id)
            if not os.path.isdir(speaker_path):
                continue

            for chapter_id in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, chapter_id)
                if not os.path.isdir(chapter_path):
                    continue

                for file_name in os.listdir(chapter_path):
                    if file_name.endswith(".flac"):
                        file_path = os.path.join(chapter_path, file_name)
                        mfcc_features = Functions.ExtractMfccFeatures(file_path, n_mfcc)
                        features.append(mfcc_features)
                        labels.append(speaker_id)

        return np.array(features), np.array(labels)
