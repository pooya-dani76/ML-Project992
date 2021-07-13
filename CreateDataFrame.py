import msvcrt
import pandas as pd


def JoinCSV(FilePath1: str, FilePath2: str ,  DestenationPath : str):
    '''
    Get 2 .csv file path and merge into a .csv file
    '''
    try:
        File1 = pd.read_csv(FilePath1)
        File2 = pd.read_csv(FilePath2)
    except IOError as e:
        print(f'Error : {e.strerror}')
        return False

    DataList = []

    for x in File1.index:
        print(f'Reading Row {x}', end='\r')
        Text = File1.loc[x, 'text']
        Label = File2.loc[x, 'label']

        Dict = {'text': Text, 'label': Label}
        DataList.append(Dict)

    pd.DataFrame(DataList).to_csv(DestenationPath , index=False)
    print('File Train Data with TrainDataFrame.csv name created ... press any key to continue...')
    msvcrt.getch()
