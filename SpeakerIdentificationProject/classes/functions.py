import os
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
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

    def SvmModel(features,labels):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        # SVM Modeli
        svm = SVC(kernel='linear')  # Doğrusal çekirdek (linear kernel) kullanıyoruz
        svm.fit(X_train, y_train)  # Modeli eğit
        # Test seti ile tahmin yap
        y_pred_svm = svm.predict(X_test)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        confusion_svm = confusion_matrix(y_test, y_pred_svm)

        print("SVM Doğruluk Oranı:", accuracy_svm)
        print("SVM Karışıklık Matrisi:\n", confusion_svm)

        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_svm, annot=True, fmt='d', cmap='Blues')
        plt.title("SVM Karışıklık Matrisi")
        plt.xlabel("Tahmin Edilen Sınıf")
        plt.ylabel("Gerçek Sınıf")
        plt.show()

    