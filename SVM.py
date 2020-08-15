import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder


count_vect = CountVectorizer(lowercase='false', analyzer='char', ngram_range=(1,4))
tfidf_transformer = TfidfTransformer()

df = pd.read_csv('processed_data.csv')
X, y = df.phonetic_transcription.fillna(' '), df.language_classification
'''print('X shape: ', X.shape)
print('y shape: ', y.shape)
print()'''

# IMPORTANT: Make a train/test split here (before fitting the vectorizers on the data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Vectorization: Bag of Words vs. TF-IDF
X_train_counts = count_vect.fit_transform(list(X_train))
print('X train counts: ', X_train_counts.shape)

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print('X train tfidf:  ', X_train_tfidf.shape)
print()

X_test_counts = count_vect.transform(list(X_test))
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
print('X test counts:  ', X_test_counts.shape)
print('X test tfidf:   ', X_test_tfidf.shape)

print(tfidf_transformer.vocabulary_)

# Encoding classes:
Encoder = LabelEncoder()
y_train = Encoder.fit_transform(y_train)
y_test = Encoder.fit_transform(y_test)

print(y_train[:20])

# Training a random forest:
'''print('Initiating Random Forest...')
text_clf_rf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=5, n_jobs=4)

print('Fitting data on Random Forest...')
text_clf_rf.fit(X_train_tfidf, y_train)

print('Predicting test labels')
y_pred = text_clf_rf.predict(X_test_tfidf)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))'''

# Training an SVM:

text_clf_svm = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
_ = text_clf_svm.fit(X_train_tfidf, y_train)
y_pred = text_clf_svm.predict(X_test_tfidf)
print('SVM accuracy: ', np.mean(y_pred == y_test))

# Plotting a confusion matrix (calculating precision, recall and F-score)


# Checking important features:

'''from matplotlib import pyplot as plt
from sklearn import svm

def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

features_names = ['input1', 'input2']
svm = svm.SVC(kernel='linear', verbose=5)
svm.fit(X_train_tfidf, y_train)
f_importances(svm.coef_, features_names)'''