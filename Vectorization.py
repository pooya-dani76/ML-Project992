from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from joblib import dump, load  
from scipy.sparse import save_npz, load_npz
from Initialize import  TrainData
import os


def UnigramVectorize():
    global UnigramVectorizer
    if os.path.isfile('Preprocessors/UnigramVectorizer.joblib') == False:
        UnigramVectorizer = CountVectorizer(ngram_range=(1, 1))  # Only Unigram
        UnigramVectorizer.fit(TrainData['text'].values)
        dump(UnigramVectorizer, 'Preprocessors/UnigramVectorizer.joblib')

    print('Loading UnigramVectorizer.joblib...')    
    UnigramVectorizer = load('Preprocessors/UnigramVectorizer.joblib')
    print('Done!')

    global UnigramTrainText
    if os.path.isfile('Vectorized/UnigramTrainText.npz') == False:
        UnigramTrainText = UnigramVectorizer.transform(TrainData['text'].values)
        save_npz('Vectorized/UnigramTrainText.npz', UnigramTrainText)

    print('Loading UnigramTrainText.npz...')
    UnigramTrainText = load_npz('Vectorized/UnigramTrainText.npz')
    print('Done!')


def UnigramTFIDF():
    global UnigramTFIDFTransformer
    if os.path.isfile('Preprocessors/UnigramTFIDFTransformer.joblib') == False:
        UnigramTFIDFTransformer = TfidfTransformer()
        UnigramTFIDFTransformer.fit(UnigramTrainText)
        dump(UnigramTFIDFTransformer, 'Preprocessors/UnigramTFIDFTransformer.joblib')

    print('Loading UnigramTFIDFTransformer.joblib...')    
    UnigramTFIDFTransformer = load(
        'Preprocessors/UnigramTFIDFTransformer.joblib')
    print('Done!')

    global UnigramTFIDFTrainText
    if os.path.isfile('Vectorized/UnigramTFIDFTrainText.npz') == False:
        UnigramTFIDFTrainText = UnigramTFIDFTransformer.transform(UnigramTrainText)
        save_npz('Vectorized/UnigramTFIDFTrainText.npz', UnigramTFIDFTrainText)

    print('Loading UnigramTFIDFTrainText.npz...')     
    UnigramTFIDFTrainText = load_npz('Vectorized/UnigramTFIDFTrainText.npz')
    print('Done!')


def BigramVectorize():
    global BigramVectorizer
    if os.path.isfile('Preprocessors/BigramVectorizer.joblib') == False:
        BigramVectorizer = CountVectorizer(ngram_range=(1, 2))  # Unigram & Bigram
        BigramVectorizer.fit(TrainData['text'].values)
        dump(BigramVectorizer, 'Preprocessors/BigramVectorizer.joblib')

    print('Loading BigramVectorizer.joblib...')     
    BigramVectorizer = load('Preprocessors/BigramVectorizer.joblib')
    print('Done!')

    global BigramTrainText
    if os.path.isfile('Vectorized/BigramTrainText.npz') == False:
        BigramTrainText = BigramVectorizer.transform(TrainData['text'].values)
        save_npz('Vectorized/BigramTrainText.npz', BigramTrainText)

    print('Loading BigramTrainText.npz...')     
    BigramTrainText = load_npz('Vectorized/BigramTrainText.npz')
    print('Done!')


def BigramTFIDF():
    global BigramTFIDFTransformer
    if os.path.isfile('Preprocessors/BigramTFIDFTransformer.joblib') == False:
        BigramTFIDFTransformer = TfidfTransformer()
        BigramTFIDFTransformer.fit(BigramTrainText)
        dump(BigramTFIDFTransformer, 'Preprocessors/BigramTFIDFTransformer.joblib')

    print('Loading BigramTFIDFTransformer.joblib...')      
    BigramTFIDFTransformer = load(
        'Preprocessors/BigramTFIDFTransformer.joblib')
    print('Done!')

    global BigramTFIDFTrainText
    if os.path.isfile('Vectorized/BigramTFIDFTrainText.npz') == False:
        BigramTFIDFTrainText = BigramTFIDFTransformer.transform(BigramTrainText)
        save_npz('Vectorized/BigramTFIDFTrainText.npz', BigramTFIDFTrainText)

    print('Loading BigramTFIDFTrainText.npz...')      
    BigramTFIDFTrainText = load_npz('Vectorized/BigramTFIDFTrainText.npz')
    print('Done!')
