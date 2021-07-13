import pandas as pd
import os
import msvcrt
from SaveBest import BestClassifier
from Initialize import TrainData
from Vectorization import BigramVectorizer , BigramTFIDFTransformer

def CreatePredictionFile():
    try:
        PredictionSet = pd.read_csv('dataset/validation/Validation-Data.csv')
    except IOError as e:
        print(f'Error : {e.strerror}')
        return False

    PredictText = BigramVectorizer.transform(PredictionSet['text'].values)
    PredictText = BigramTFIDFTransformer.transform(PredictText)    
    Label = BestClassifier.predict(PredictText) 
    
    Predicted = []

    for x in PredictionSet.index:
        Lbl = Label[x]
        id = PredictionSet.loc[x , 'ID']
        Dict = {'ID' : id , 'label' : Lbl}
        Predicted.append(Dict)

    if os.path.isdir('Result') == False:
        os.system('mkdir Result')

    pd.DataFrame(Predicted).to_csv('Result/PredictResult.csv' , index=False) 
    print(f'Result .csv File Created At Result/PredictResult.csv')
    msvcrt.getch()