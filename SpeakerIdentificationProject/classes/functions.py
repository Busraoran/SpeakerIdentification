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
                    if file_name.endswith(".flac"):
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

    