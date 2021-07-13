from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from scipy.stats import uniform
from Vectorization import BigramTFIDFTrainText
from Initialize import TrainData
import os

TrainTxt = BigramTFIDFTrainText
TrainLbl = TrainData['label'].values

Classifier = SGDClassifier()

DistributionsPhase1 = dict(
    loss=['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
    learning_rate=['optimal', 'invscaling', 'adaptive'],
    eta0=uniform(loc=1e-7, scale=1e-2)
)

DistributionsPhase2 = dict(
    penalty=['l1', 'l2', 'elasticnet'],
    alpha=uniform(loc=1e-6, scale=1e-4)
)


def SetHyperParameters():
    global RandomSearchCV

    if os.path.isfile('Classifiers/BestClassifier.joblib') == False:
        # Phase 1: loss, learning rate and initial learning rate
        RandomSearchCV = RandomizedSearchCV(
            estimator=Classifier, param_distributions=DistributionsPhase1, cv=5, n_iter=50)
        RandomSearchCV.fit(TrainTxt, TrainLbl)

        print(f'Best Parameters For Phase 1: {RandomSearchCV.best_params_}')
        print(f'Best Score: {RandomSearchCV.best_score_}')

        # Phase 2: penalty and alpha
        RandomSearchCV = RandomizedSearchCV(
            estimator=Classifier, param_distributions=DistributionsPhase2, cv=5, n_iter=50)
        RandomSearchCV.fit(TrainTxt, TrainLbl)

        print(f'Best Parameters For Phase 2: {RandomSearchCV.best_params_}')
        print(f'Best Score: {RandomSearchCV.best_score_}')
