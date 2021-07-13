from CreateDataFrame import JoinCSV
import os
import pandas as pd

DataPath1 = 'dataset/train/Train-Data.csv'
DataPath2 = 'dataset/train/Train-GT.csv'


def initialize():
    '''Create Folders and Files needed'''

    if os.path.isfile('dataset/train/TrainDataFrame.csv') == False:
        JoinCSV(DataPath1, DataPath2 , 'dataset/train/TrainDataFrame.csv')

    global TrainData
    TrainData = pd.read_csv('dataset/train/TrainDataFrame.csv')

    if os.path.isdir('Preprocessors') == False:
        os.system("mkdir Preprocessors")

    if os.path.isdir('Vectorized') == False:
        os.system("mkdir Vectorized")
