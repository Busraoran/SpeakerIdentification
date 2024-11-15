import librosa
import librosa.display
import matplotlib.pyplot as plt
from classes.functions import Functions

# Ses dosyasını yükle
file_name = "0000.flac"
path = r"D:\PythonSoundProcessing\MainProject\SoundFiles"
#Functions.DrawSoundWaveForm(file_name, path)
fileNames= [f"{str(i)}.flac" for i in range(10)]
Functions.DrawAllSoundsWaveForm(fileNames,path,True)
#123123123123