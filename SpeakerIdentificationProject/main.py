from classes.functions import Functions
from sklearn.metrics import classification_report

# MFCC
#soundFilesDir="D:/PythonSoundProcessing/SpeakerIdentification/SoundFiles"
#soundFilesDir="C:/Users/Busra/Desktop/speakeridentification/train-clean.100tar/LibriSpeech"
#data_dir = soundFilesDir+"/train"
#data_dir = soundFilesDir+"/train-voxceleb"
#data_dir = soundFilesDir+"/train-clean-100"
#KNN SVM

#features, labels = Functions.PrepareDataset(data_dir)
#print("Özelliklerin Boyutu:", features.shape)
#print("Etiketlerin Boyutu:", labels.shape)

#Functions.SvmModel(features,labels)
#Functions.KnnModel(features, labels)
#Functions.KnnModelNeighborsTest(features, labels)

#KNNSVMEnd

#CNN
#X_cnn, y_cnn = Functions.PrepareDatasetForCNN(data_dir)
#Functions.CnnModel(X_cnn, y_cnn)
#CNNEnd

#########################################################################################################

# açı genlik dönüşümü 


if __name__ == "__main__":
    soundFilesDir = "C:/Users/Busra/Desktop/speakeridentification/train-clean.100tar/LibriSpeech"
    data_dir = soundFilesDir + "/train"

    Functions.run_knn_pipeline(data_dir)

    #Functions.run_svm_pipeline(data_dir) 

   # Functions.run_cnn_pipeline(data_dir)




