from joblib import dump, load  
from sklearn.metrics import f1_score , accuracy_score , confusion_matrix , roc_auc_score
import pandas as pd


def PrintMetrics():
    '''
    print metrics of the model
    '''
    BigramVectorizer = load('Preprocessors/BigramVectorizer.joblib')
    BigramTFIDFTransformer = load('Preprocessors/BigramTFIDFTransformer.joblib')
    BestSGDClassifier = load('Classifiers/BestClassifier.joblib')

    TestText = pd.read_csv('dataset/test/Test-Data.csv')
    TestLabel = pd.read_csv('dataset/test/Test-GT.csv')

    TestText = TestText['text']
    TestLabel = TestLabel['label']
    
    PredictText = BigramVectorizer.transform(TestText.values)
    PredictText = BigramTFIDFTransformer.transform(PredictText) 
    Label = BestSGDClassifier.predict(PredictText)

    print(f'f1 Score : {f1_score(TestLabel.values , Label)}')
    print(f'Accuracy Score: {accuracy_score(TestLabel.values , Label)}')
    print(f'Confusion Matrix : {confusion_matrix(TestLabel.values , Label)}')
    print(f'roc_auc_score : {roc_auc_score(TestLabel.values , Label)}')


PrintMetrics()    