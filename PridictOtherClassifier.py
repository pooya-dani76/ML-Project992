from joblib import load
from sklearn.metrics import f1_score ,accuracy_score , roc_auc_score  ,confusion_matrix , classification_report , plot_confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
# from CreateDataFrame import JoinCSV

# JoinCSV('dataset/validation/Validation-Data.csv',
#         'dataset/validation/Validation-GT.csv', 'dataset/validation/ValidationDataSet.csv')

# JoinCSV('dataset/test/Test-Data.csv',
#         'dataset/test/Test-GT.csv', 'dataset/test/TestDataSet.csv')

def Report():
    print('Loading Models...')
    AdaBoostModel = load('Classifiers/AdaBoostModel.joblib')
    SGDModel = load('Classifiers/BestClassifier.joblib')
    LogesticModel = load('Classifiers/LogesticModel.joblib')
    SVMModel = load('Classifiers/SVMModel.joblib')
    RandomForestModel = load('Classifiers/RandomForestModel.joblib')

    print('Loading Data...')
    ValidationData = pd.read_csv('dataset/validation/ValidationDataSet.csv')
    ValidationText = ValidationData['text'].values
    ValidationLabel = ValidationData['label'].values

    TestData = pd.read_csv('dataset/test/TestDataSet.csv')
    TestText = TestData['text'].values
    TestLabel = TestData['label'].values

    BigramVectorizer = load('Preprocessors/BigramVectorizer.joblib')
    BigramTFIDFTransformer = load('Preprocessors/BigramTFIDFTransformer.joblib')

    ValidationText = BigramVectorizer.transform(ValidationText)
    ValidationText = BigramTFIDFTransformer.transform(ValidationText)

    TestText = BigramVectorizer.transform(TestText)
    TestText = BigramTFIDFTransformer.transform(TestText)

    plot_confusion_matrix(AdaBoostModel , ValidationText , ValidationLabel )
    plt.show() 
    plot_confusion_matrix(AdaBoostModel , TestText , TestLabel )
    plt.show()

    plot_confusion_matrix(SGDModel , ValidationText , ValidationLabel )
    plt.show()
    plot_confusion_matrix(SGDModel , TestText , TestLabel )
    plt.show()

    plot_confusion_matrix(LogesticModel , ValidationText , ValidationLabel )
    plt.show()
    plot_confusion_matrix(LogesticModel , TestText , TestLabel )
    plt.show()

    plot_confusion_matrix(SVMModel , ValidationText , ValidationLabel )
    plt.show()
    plot_confusion_matrix(SVMModel , TestText , TestLabel )
    plt.show()

    plot_confusion_matrix(RandomForestModel , ValidationText , ValidationLabel )
    plt.show()
    plot_confusion_matrix(RandomForestModel , TestText , TestLabel )
    plt.show()

    print('Adaptive boost report:')
    print('On Validation Data:')
    Y1 = AdaBoostModel.predict(ValidationText)
    print(classification_report(ValidationLabel ,Y1))
    print(f'Confusion Matrix : {confusion_matrix(ValidationLabel ,Y1)}')
    print(f'Accuracy : {accuracy_score(ValidationLabel ,Y1)}')
    print(f'f1_score : {f1_score(ValidationLabel ,Y1)}')
    print(f'roc_auc_score : {roc_auc_score(ValidationLabel ,Y1)}')

    print('On Test Data:')
    Y2 = AdaBoostModel.predict(TestText)
    print(classification_report(TestLabel ,Y2))
    print(f'Confusion Matrix : {confusion_matrix(TestLabel ,Y2)}')
    print(f'Accuracy : {accuracy_score(TestLabel ,Y2)}')
    print(f'f1_score : {f1_score(TestLabel ,Y2)}')
    print(f'roc_auc_score : {roc_auc_score(TestLabel ,Y2)}')

    ###########################################################################

    print('SGDModel report:')
    print('On Validation Data:')
    Y3 = SGDModel.predict(ValidationText)
    print(classification_report(ValidationLabel ,Y3))
    print(f'Confusion Matrix : {confusion_matrix(ValidationLabel ,Y3)}')
    print(f'Accuracy : {accuracy_score(ValidationLabel ,Y3)}')
    print(f'f1_score : {f1_score(ValidationLabel ,Y3)}')
    print(f'roc_auc_score : {roc_auc_score(ValidationLabel ,Y3)}')

    print('On Test Data:')
    Y4 = SGDModel.predict(TestText)
    print(classification_report(TestLabel ,Y4))
    print(f'Confusion Matrix : {confusion_matrix(TestLabel ,Y4)}')
    print(f'Accuracy : {accuracy_score(TestLabel ,Y4)}')
    print(f'f1_score : {f1_score(TestLabel ,Y4)}')
    print(f'roc_auc_score : {roc_auc_score(TestLabel ,Y4)}')

    ##########################################################################

    print('RandomForestModel report:')
    print('On Validation Data:')
    Y5 = RandomForestModel.predict(ValidationText)
    print(classification_report(ValidationLabel ,Y5))
    print(f'Confusion Matrix : {confusion_matrix(ValidationLabel ,Y5)}')
    print(f'Accuracy : {accuracy_score(ValidationLabel ,Y5)}')
    print(f'f1_score : {f1_score(ValidationLabel ,Y5)}')
    print(f'roc_auc_score : {roc_auc_score(ValidationLabel ,Y5)}')

    print('On Test Data:')
    Y6 = RandomForestModel.predict(TestText)
    print(classification_report(TestLabel ,Y6))
    print(f'Confusion Matrix : {confusion_matrix(TestLabel ,Y6)}')
    print(f'Accuracy : {accuracy_score(TestLabel ,Y6)}')
    print(f'f1_score : {f1_score(TestLabel ,Y6)}')
    print(f'roc_auc_score : {roc_auc_score(TestLabel ,Y6)}')

    ##########################################################################

    print('LogisticModel report:')
    print('On Validation Data:')
    Y7 = LogesticModel.predict(ValidationText)
    print(classification_report(ValidationLabel ,Y7))
    print(f'Confusion Matrix : {confusion_matrix(ValidationLabel ,Y7)}')
    print(f'Accuracy : {accuracy_score(ValidationLabel ,Y7)}')
    print(f'f1_score : {f1_score(ValidationLabel ,Y7)}')
    print(f'roc_auc_score : {roc_auc_score(ValidationLabel ,Y7)}')

    print('On Test Data:')
    Y8 = LogesticModel.predict(TestText)
    print(classification_report(TestLabel ,Y8))
    print(f'Confusion Matrix : {confusion_matrix(TestLabel ,Y8)}')
    print(f'Accuracy : {accuracy_score(TestLabel ,Y8)}')
    print(f'f1_score : {f1_score(TestLabel ,Y8)}')
    print(f'roc_auc_score : {roc_auc_score(TestLabel ,Y8)}')

    ###########################################################################

    print('SVMModel report:')
    print('On Validation Data:')
    Y9 = SVMModel.predict(ValidationText)
    print(classification_report(ValidationLabel ,Y9))
    print(f'Confusion Matrix : {confusion_matrix(ValidationLabel ,Y9)}')
    print(f'Accuracy : {accuracy_score(ValidationLabel ,Y9)}')
    print(f'f1_score : {f1_score(ValidationLabel ,Y9)}')
    print(f'roc_auc_score : {roc_auc_score(ValidationLabel ,Y9)}')

    print('On Test Data:')
    Y10 = SVMModel.predict(TestText)
    print(classification_report(TestLabel ,Y10))
    print(f'Confusion Matrix : {confusion_matrix(TestLabel ,Y10)}')
    print(f'Accuracy : {accuracy_score(TestLabel ,Y10)}')
    print(f'f1_score : {f1_score(TestLabel ,Y10)}')
    print(f'roc_auc_score : {roc_auc_score(TestLabel ,Y10)}')

Report()    