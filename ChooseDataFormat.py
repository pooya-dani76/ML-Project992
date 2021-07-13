from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from scipy.sparse import csc_matrix
import numpy as np



def TrainAndTest(X: csc_matrix,  y: np.array, title: str):
    '''
    Train Model And Test
    Calculate Model Score 
    '''
    TextTrain, TextValid, LabelTrain, LabelValid = train_test_split(
        X, y, train_size=0.75, stratify=y)
    Model = SGDClassifier()
    Model.fit(TextTrain, LabelTrain)
    TrainScore = Model.score(TextTrain, LabelTrain)
    TestScore = Model.score(TextValid, LabelValid)
    print(f'{title}\nTrain score: {round(TrainScore, 2)} ; Validation score: {round(TestScore, 2)}\n') 