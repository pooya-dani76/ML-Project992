# ML992 : Text Sentiment Analysis on IMDB Movie Reviews Dataset

Task is Implement a machine learning method for prediction on some movie reviews.

## Datasets:

**Train Dataset:**

Training dataset stored at `dataset/train/Train-Data.csv` contained 40000 Reviews with long sentence(most of them are longer than 200 words) & its ID.
Also `dataset/train/Train-GT.csv` contained their ID & label(0 for Negative & 1 For Positive)
We merge these 2 files into `dataset/train/TrainDataFrame.csv`. This file containes 40000 Review and its labels.

**Validation Dataset:**
Validation dataset (Phase 1 Test) stored at `dataset/validation/Validation-Data.csv` contained 5000 Reviews with long sentence & its ID.
Also `dataset/validation/Validation-GT.csv` contained their ID & label(0 for Negative & 1 For Positive)
We merge these 2 files into `dataset/validation/ValidationDataSet.csv`. This file containes 5000 Review and its labels.

**Test Dataset:**
Test dataset (Phase 2) stored at `dataset/test/Test-Data.csv` contained 5000 Reviews with long sentence & its ID.
Also `dataset/test/Test-GT.csv` contained their ID & label(0 for Negative & 1 For Positive)
We merge these 2 files into `dataset/test/TestDataSet.csv`. This file containes 5000 Review and its labels.

## Preprocessing & Vectorization:
   We test all of these task:
   - Unigram
   - Unigram + TDIDF
   - Bigram
   - Bigram + TDIDF
   And The best one is `Bigram + TDIDF`

## Training:
   For Training Model(s) we split 75% of `dataset/train/TrainDataFrame.csv` & remaining use for phase 1 validation (25%). 

## Methods reviewed & their results:
### 1. Adaptive Boosting:
   <p align = "center">
   <img src = "https://github.com/pooya-dani76/ML-Project992/blob/main/Reports/AdaBoosting.PNG">
   </p>
   **Confusion Matrix:**

   <p align = "center">
   <img src="https://github.com/pooya-dani76/ML-Project992/blob/main/Confiusion%20Matrix/AdaBoost-Val.png" width = "395"  height = "320">
   </p>
   
### 2. SGDClassifier(Lineaer Model + Stochastic Gradient Descent):
   <p align = "center">
   <img src = "https://github.com/pooya-dani76/ML-Project992/blob/main/Reports/SGD.PNG">
   </p>
   **Confusion Matrix:**
 
   <p align = "center">
   <img src="https://github.com/pooya-dani76/ML-Project992/blob/main/Confiusion%20Matrix/SGD-Val.png" width = "395"  height = "320">
   </p>   
   

## Run Model:
   Just Run `Main.py` File.


## For Use For Another Train and Test Dataset:

  - Put your dataset in path `dataset/train` and rename ID & text file to `Train-Data.csv` and rename ID & Label data to `Train-GT.csv`
  - Put file you want predict on it to path `dataset/validation` to `Validation-Data.csv`
  - After prediction , predicted file will create in Result folder named  `PredictResult.csv`


