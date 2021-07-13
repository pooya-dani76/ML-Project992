print('Level 1: Initializing...')
import Initialize as ini
ini.initialize()

print('Level 2: Vectorization...')
import Vectorization as vec
vec.UnigramVectorize()
vec.UnigramTFIDF()
vec.BigramVectorize()
vec.BigramTFIDF()

print('Level 3: Choose Data Format...')
from ChooseDataFormat import TrainAndTest
TrainLabel = ini.TrainData['label'].values
TrainAndTest(vec.UnigramTrainText , TrainLabel , 'Unigram Counts')
TrainAndTest(vec.UnigramTFIDFTrainText , TrainLabel , ' Unigram TF-IDF')
TrainAndTest(vec.BigramTrainText , TrainLabel , 'Bigram Counts')
TrainAndTest(vec.BigramTFIDFTrainText , TrainLabel , 'Bigram TF-IDF')

print('Level 4: Set HyperParameters...')
from SetHyperParameters import SetHyperParameters
import os
if os.path.isfile('Classifiers/BestClassifier.joblib') == False:
    SetHyperParameters()

print('Level 5: Save Best Model...')
from SaveBest import SaveBestModel
SaveBestModel()

print('Level 6: Prediction...')
from Prediction import CreatePredictionFile
CreatePredictionFile()
