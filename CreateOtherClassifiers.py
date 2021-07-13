from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier , RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from joblib import load , dump
from scipy.sparse import load_npz

def CreateLinearModel():
    print('Entered!')
    TrainSet = pd.read_csv('dataset/train/TrainDataFrame.csv')
    TrainLabel = TrainSet['label'].values
    print('loaded')
    BigramTFIDFTrainText = load_npz('Vectorized/BigramTFIDFTrainText.npz')

    XTrain, XValid, YTrain, YValid = train_test_split(
        BigramTFIDFTrainText, TrainLabel, train_size=0.75)

    LinearModel = LinearRegression()
    LinearModel.fit(XTrain , YTrain)

    validScore = LinearModel.score(XTrain , YTrain)
    TestScore = LinearModel.score(XValid , YValid)

    print(f'validScore = {validScore}  , TestScore = {TestScore}')

def CreateLogesticModel():
    TrainSet = pd.read_csv('dataset/train/TrainDataFrame.csv')
    TrainLabel = TrainSet['label'].values

    BigramTFIDFTrainText = load_npz('Vectorized/BigramTFIDFTrainText.npz')

    XTrain, XValid, YTrain, YValid = train_test_split(
        BigramTFIDFTrainText, TrainLabel, train_size=0.75)

    LogesticModel = LogisticRegression()
    LogesticModel.fit(XTrain , YTrain)

    validScore = LogesticModel.score(XTrain , YTrain)
    TestScore = LogesticModel.score(XValid , YValid)

    print(f'validScore = {validScore}  , TestScore = {TestScore}')    

def CreateSVMModel():
    print('LoadCSV...')
    TrainSet = pd.read_csv('dataset/train/TrainDataFrame.csv')
    TrainLabel = TrainSet['label'].values

    print('Loaded!')
    BigramTFIDFTrainText = load_npz('Vectorized/BigramTFIDFTrainText.npz')

    print('LoadNPZ...')
    XTrain, XValid, YTrain, YValid = train_test_split(
        BigramTFIDFTrainText, TrainLabel, train_size=0.75)
    
    print('Splited...' )
    print('Training...')  
    SVMModel = svm.SVC()
    SVMModel.fit(XTrain , YTrain)

    dump(SVMModel , 'Classifiers/SVMModel.joblib')
    print('SVMModel saved!')

    print('done!')
    validScore = SVMModel.score(BigramTFIDFTrainText , TrainLabel)
    TestScore = SVMModel.score(XValid , YValid)

    print(f'validScore = {validScore}  , TestScore = {TestScore}')  

def CreateAdaptiveBoostingModel():
    TrainSet = pd.read_csv('dataset/train/TrainDataFrame.csv')
    TrainLabel = TrainSet['label'].values

    BigramTFIDFTrainText = load_npz('Vectorized/BigramTFIDFTrainText.npz')

    XTrain, XValid, YTrain, YValid = train_test_split(
        BigramTFIDFTrainText, TrainLabel, train_size=0.75)

    LogesticModel = AdaBoostClassifier(n_estimators=100, random_state= 0)
    LogesticModel.fit(XTrain , YTrain)

    validScore = LogesticModel.score(XTrain , YTrain)
    TestScore = LogesticModel.score(XValid , YValid)

    print(f'validScore = {validScore}  , TestScore = {TestScore}') 

def CreateRandomForestModel():
    TrainSet = pd.read_csv('dataset/train/TrainDataFrame.csv')
    TrainLabel = TrainSet['label'].values

    BigramTFIDFTrainText = load_npz('Vectorized/BigramTFIDFTrainText.npz')

    XTrain, XValid, YTrain, YValid = train_test_split(
        BigramTFIDFTrainText, TrainLabel, train_size=0.75)

    LogesticModel = RandomForestClassifier(max_depth=3 , random_state=0)
    LogesticModel.fit(XTrain , YTrain)

    validScore = LogesticModel.score(XTrain , YTrain)
    TestScore = LogesticModel.score(XValid , YValid)

    print(f'validScore = {validScore}  , TestScore = {TestScore}')

CreateLinearModel()
CreateLogesticModel()
CreateSVMModel()
CreateAdaptiveBoostingModel()
CreateRandomForestModel()