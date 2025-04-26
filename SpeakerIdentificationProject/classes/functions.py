import os
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import cv2
from sklearn.cluster import KMeans
from scipy.signal import argrelextrema


class Functions:
    #KNN
    @staticmethod
    def ExtractMfccFeatures(file_path, n_mfcc=13):
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1) #ortalama alır
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
                    if file_name.endswith((".flac", ".wav")):
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

    def KnnModel(features, labels, n_neighbors=5):
        #n_neighbors=5 parametresi, KNN algoritmasının en yakın kaç komşuya bakarak karar vereceğini belirler (varsayılan: 5)
        #Veriyi eğitim (%80) ve test (%20) olarak ikiye ayırırız. random_state=42 ifadesi, deneyin tekrarlanabilir olmasını sağlar.
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        
        #KNeighborsClassifier sınıfından bir KNN modeli oluşturuyoruz.
        #Burada kaç komşuya bakacağını daha önce belirttiğimiz n_neighbors belirliyor.
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        #KNN modelini (X_train, y_train) ile eğitiyoruz.
        knn.fit(X_train, y_train)
        #Eğitimden sonra modeli test verisiyle deniyoruz.
        #predict() fonksiyonu her test örneği için en yakın 5 komşuyu bulup çoğunluk oyu ile sınıf atar.
        y_pred_knn = knn.predict(X_test)

        #Tahmin edilen sınıflarla (y_pred_knn) gerçek sınıfları (y_test) karşılaştırarak doğruluk oranını hesaplıyoruz.
        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        #Karışıklık matrisi: Hangi sınıfın hangi sınıfa karıştırıldığını gösterir.
        confusion_knn = confusion_matrix(y_test, y_pred_knn)

        print("KNN Doğruluk Oranı:", accuracy_knn)
        print("KNN Karışıklık Matrisi:\n", confusion_knn)

        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_knn, annot=True, fmt='d', cmap='Greens')
        plt.title("KNN Karışıklık Matrisi")
        plt.xlabel("Tahmin Edilen Sınıf")
        plt.ylabel("Gerçek Sınıf")
        plt.show()

    def KnnModelNeighborsTest(features, labels, n_neighbors=5):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        for k in range(1, 21):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"K={k} için doğruluk: {acc:.2f}")
    #KNNEnd

    #CNN
    @staticmethod
    def ExtractMfccFeaturesCNN(file_path, n_mfcc=13):
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfcc  # Ortalama almadan
    
    def PrepareDatasetForCNN(data_dir, n_mfcc=13, max_len=100):
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
                    if file_name.endswith((".flac", ".wav")):
                        file_path = os.path.join(chapter_path, file_name)
                        mfcc = Functions.ExtractMfccFeaturesCNN(file_path, n_mfcc)
                        
                        if mfcc.shape[1] < max_len:
                            pad_width = max_len - mfcc.shape[1]
                            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
                        else:
                            mfcc = mfcc[:, :max_len]
                        
                        features.append(mfcc)
                        labels.append(speaker_id)

        X = np.array(features)[..., np.newaxis]  # (sample, height, width, channel)
        y = np.array(labels)
        return X, y


    def CnnModel(X, y):
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)

        X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(13, 100, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(y_categorical.shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        test_loss, test_acc = model.evaluate(X_test, y_test)
        print("CNN Test Doğruluğu:", test_acc)

        #karışıklık matrisi
        y_pred = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_true_labels, y_pred_labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
        plt.title("CNN Karışıklık Matrisi")
        plt.xlabel("Tahmin Edilen Sınıf")
        plt.ylabel("Gerçek Sınıf")
        plt.show()

    #CNNEnd

    #Açı- genlik dönüşümü 

    @staticmethod
    def find_local_extrema(signal):
        max_idx = argrelextrema(signal, np.greater)[0]
        min_idx = argrelextrema(signal, np.less)[0]
        return max_idx, min_idx

    @staticmethod
    def calculate_angle_amplitude(signal, peaks, valleys):
        points = np.sort(np.concatenate((peaks, valleys)))
        angle_list = []
        amplitude_list = []

        for i in range(1, len(points)-1):
            x1, y1 = points[i-1], signal[points[i-1]]
            x2, y2 = points[i], signal[points[i]]
            x3, y3 = points[i+1], signal[points[i+1]]

            left_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            right_dist = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)

            m1 = (y2 - y1) / (x2 - x1 + 1e-8)
            m2 = (y3 - y2) / (x3 - x2 + 1e-8)
            angle = np.degrees(np.arctan((m2 - m1) / (1 + m1 * m2 + 1e-8)))

            if left_dist < right_dist:
                amplitude = left_dist / (right_dist + 1e-8)
            else:
                amplitude = -(right_dist / (left_dist + 1e-8))

            angle_list.append(angle)
            amplitude_list.append(amplitude)

        return np.array(angle_list), np.array(amplitude_list)

    @staticmethod
    def generate_angle_amplitude_image(angle, amplitude, size=(300, 300)):
        img = np.zeros(size, dtype=np.uint8)
        center = (size[1]//2, size[0]//2)

        for a, amp in zip(angle, amplitude):
            x = int(center[0] + amp * center[0])
            y = int(center[1] - a * center[1] / 180)

            if 0 <= x < size[1] and 0 <= y < size[0]:
                img[y, x] = 255

        return img

    @staticmethod
    def extract_features(images, n_clusters=500):
        sift = cv2.SIFT_create()
        descriptor_list = []

        for img in images:
            keypoints = cv2.goodFeaturesToTrack(img, 500, 0.01, 10)
            if keypoints is not None:
                keypoints = [cv2.KeyPoint(float(x[0][0]), float(x[0][1]), 1) for x in keypoints]
                _, descriptors = sift.compute(img, keypoints)
                if descriptors is not None:
                    descriptor_list.extend(descriptors)

        descriptor_array = np.array(descriptor_list, dtype=np.float32)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(descriptor_array)

        features = []
        for img in images:
            keypoints = cv2.goodFeaturesToTrack(img, 500, 0.01, 10)
            if keypoints is not None:
                keypoints = [cv2.KeyPoint(float(x[0][0]), float(x[0][1]), 1) for x in keypoints]
                _, descriptors = sift.compute(img, keypoints)
                if descriptors is not None:
                    predict = kmeans.predict(descriptors.astype(np.float32))
                    hist, _ = np.histogram(predict, bins=np.arange(n_clusters+1))
                else:
                    hist = np.zeros(n_clusters)
            else:
                hist = np.zeros(n_clusters)
            features.append(hist)

        return np.array(features)

    @staticmethod
    def prepare_dataset(data_dir):
        images = []
        labels = []

        for root, dirs, files in os.walk(data_dir):
            for file_name in files:
                if not file_name.endswith('.flac'):
                    continue

                file_path = os.path.join(root, file_name)
                signal, sr = librosa.load(file_path, sr=None)

                peaks, valleys = Functions.find_local_extrema(signal)
                angle1, amplitude1 = Functions.calculate_angle_amplitude(signal, peaks, valleys)

                signal_smooth = librosa.effects.preemphasis(signal)
                peaks2, valleys2 = Functions.find_local_extrema(signal_smooth)
                angle2, amplitude2 = Functions.calculate_angle_amplitude(signal_smooth, peaks2, valleys2)

                img1 = Functions.generate_angle_amplitude_image(angle1, amplitude1)
                img2 = Functions.generate_angle_amplitude_image(angle2, amplitude2)

                combined_img = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
                images.append(combined_img)

                speaker_id = os.path.basename(os.path.dirname(file_path))
                labels.append(speaker_id)

        return images, labels

    @staticmethod
    def run_knn_pipeline(data_dir):
        images, labels = Functions.prepare_dataset(data_dir)
        features = Functions.extract_features(images)

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {acc * 100:.2f}%")

        # Karışıklık matrisi
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('KNN Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
