# random forest for feature importance on a classification problem
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
from joblib import dump, load

# DATASET 
vectorizer = TfidfVectorizer(lowercase=False, analyzer='char', ngram_range=(2,5))

df = pd.read_csv('processed_data_nospace.csv')
X, y = df.phonetic_transcription.fillna(' '), df.language_classification

# make a train/test split here (before fitting the vectorizers on the data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# VECTORIZATION
# fitting and transforming the train data
X_train = vectorizer.fit_transform(X_train)

# transforming the test data
X_test = vectorizer.transform(X_test)

# Encoding classes:
Encoder = LabelEncoder()
y_train = Encoder.fit_transform(y_train)
y_test = Encoder.fit_transform(y_test)


# MODEL
try:
    model = load('randomforest_test.joblib') 
except:
    # define the model
    print('Initiating Random Forest...')
    model = RandomForestClassifier(n_estimators=4, random_state=42, verbose=5, n_jobs=4)

    # fit the model
    print('Fitting data on Random Forest...')
    model.fit(X_train, y_train)

    # save the model
    dump(model, 'randomforest_test.joblib')

# predicting test labels
print('Predicting test labels')
y_pred = model.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print('ACCURACY: ', accuracy_score(y_test, y_pred))

# FEATURE IMPORTANCE
# get importance and feature ranking
n = 20

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

features_by_index = vectorizer.get_feature_names()

print("Feature ranking:")

for f in range(30):
    print("%d. %s (%f)" % (f + 1, features_by_index[indices[f]], importances[indices[f]]))