from classes.functions import Functions

#soundFilesDir="D:/PythonSoundProcessing/SpeakerIdentification/SoundFiles"
soundFilesDir="C:/Users/Busra/Desktop/speakeridentification/train-clean.100tar/LibriSpeech"
data_dir = soundFilesDir+"/train"
#data_dir = soundFilesDir+"/train-voxceleb"
#data_dir = soundFilesDir+"/train-clean-100"
#KNN SVM

features, labels = Functions.PrepareDataset(data_dir)
print("Özelliklerin Boyutu:", features.shape)
print("Etiketlerin Boyutu:", labels.shape)

Functions.SvmModel(features,labels)
#Functions.KnnModel(features, labels)
#Functions.KnnModelNeighborsTest(features, labels)

#KNNSVMEnd

#CNN
#X_cnn, y_cnn = Functions.PrepareDatasetForCNN(data_dir)
#Functions.CnnModel(X_cnn, y_cnn)
#CNNEnd