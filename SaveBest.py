import os
from joblib import dump, load

def SaveBestModel():
    global BestClassifier
    if os.path.isfile('Classifiers/BestClassifier.joblib') == False:
        os.system("mkdir Classifiers")
        from SetHyperParameters import RandomSearchCV
        BestClassifier = RandomSearchCV.best_estimator_
        dump(RandomSearchCV.best_estimator_, 'Classifiers/BestClassifier.joblib')
    BestClassifier = load('Classifiers/BestClassifier.joblib')
