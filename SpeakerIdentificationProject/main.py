from classes.functions import Functions

soundFilesDir="D:/PythonSoundProcessing/SpeakerIdentification/SoundFiles"
data_dir = soundFilesDir+"/train-clean-100"
features, labels = Functions.PrepareDataset(data_dir)
print("Özelliklerin Boyutu:", features.shape)
print("Etiketlerin Boyutu:", labels.shape)