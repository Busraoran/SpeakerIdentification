import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QScrollArea, QSizePolicy,QFrame
from PyQt5.QtCore import QSize,Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class Functions:
    @staticmethod
    def DrawSoundWaveForm(file_name, path):
        """
        Verilen dosya adını ve yolu birleştirerek ses dalga formunu çizen metot.
        """
        file_path = path + "\\" + file_name
        # Ses dosyasını yükle
        y, sr = librosa.load(file_path, sr=None)  # y: sinyal, sr: örnekleme hızı

        # Zaman ekseninde sinyali görselleştir
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr)
        plt.title(f"Ses Dalga Formu: {file_name}")
        plt.xlabel("Zaman (saniye)")
        plt.ylabel("Genlik")
        plt.show()

    def DrawAllSoundsWaveForm(file_names, path, single_window=True):
        """
        Birden fazla ses dosyasının dalga formlarını çizen metot.
        - file_names: Dosya adlarının listesi
        - path: Dosyaların bulunduğu dizin
        - single_window: True ise tüm grafikler tek bir pencerede, False ise ayrı pencerelerde açılır.
        """
        if single_window:
            app = QApplication(sys.argv)
            window = QWidget()
            window.setWindowTitle("Ses Dalga Formları")

            # Ana dikey düzen (Layout)
            layout = QVBoxLayout(window)

            # Scrollable alan oluşturuyoruz
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_widget = QWidget()
            scroll_layout = QVBoxLayout(scroll_widget)  # Dikey düzenek (layout)

            # Grafiklerin çizilmesi
            for file_name in file_names:
                file_path = path + "\\" + file_name
                y, sr = librosa.load(file_path, sr=None)

                # Matplotlib figürü oluşturuyoruz
                fig, ax = plt.subplots(figsize=(10, 4))
                librosa.display.waveshow(y, sr=sr, ax=ax)
                ax.set_title(f"Ses Dalga Formu: {file_name}")
                ax.set_xlabel("Zaman (saniye)")
                ax.set_ylabel("Genlik")

                # Matplotlib figürünü PyQt5'e entegre ediyoruz
                canvas = FigureCanvas(fig)
                canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Boyut politikasını ayarlıyoruz
                canvas.setMinimumSize(QSize(800, 400))  # Minimum boyutu ayarlıyoruz
                scroll_layout.addWidget(canvas)  # Canvas'ı scrollable layout'a ekliyoruz

            # Scrollable widget'ı ana pencereye ekliyoruz
            scroll_area.setWidget(scroll_widget)
            layout.addWidget(scroll_area)  # Scrollable alanı ana layout'a ekliyoruz

            window.setLayout(layout)
            window.resize(900, 700)  # Pencerenin başlangıç boyutunu ayarlıyoruz
            window.show()
            sys.exit(app.exec_())
        else:
            # Ayrı pencerelerde grafikler
            for file_name in file_names:
                file_path = path + "\\" + file_name
                y, sr = librosa.load(file_path, sr=None)

                # Her grafik için ayrı pencere
                plt.figure(figsize=(10, 4))
                librosa.display.waveshow(y, sr=sr)
                plt.title(f"Ses Dalga Formu: {file_name}")
                plt.xlabel("Zaman (saniye)")
                plt.ylabel("Genlik")
                plt.show(block=False)
            input("Grafikleri kapatmak için Enter tuşuna basın...")

    