import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

# get data
print('Step 1: getting training and testing data...')
df = pd.read_csv('processed_data_nospace.csv')
X, y = df.phonetic_transcription.fillna(' '), df.language_classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

######################################################################################################

# Linear SVM
# define pipeline: tfidf_vectorizer + SGDClassifier
print('Step 2a: defining the pipeline: tfidf_vectorizer + SGDClassifier...')

svm = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=False, analyzer='char', ngram_range=(1,5), max_df=0.7)),
    ('svm', SGDClassifier(class_weight='balanced'))
    ])

print('Step 3a: fitting pipeline')
svm.fit(X_train, y_train)

# predict labels and report metrics
y_pred = svm.predict(X_test)
cr = classification_report(y_test, y_pred)
print('Step 4a: report metrics (LinearSGD):\n', cr)
print()


######################################################################################################

#RANDOM FOREST
# define pipeline: tfidf_vectorizer + random forest classifier
print('Step 2b: defining the pipeline: tfidf_vectorizer + RandomForest...')

rf = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=False, analyzer='char', ngram_range=(1,5), max_df=0.7)),
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42, verbose=2, n_jobs=4))
    ])

print('Step 3b: fitting pipeline')
rf.fit(X_train, y_train)

# predict labels and report metrics
y_pred = rf.predict(X_test)
cr = classification_report(y_test, y_pred)
print('Step 4b: report metrics (RandomForest):\n', cr)
print()

######################################################################################################

# MULTINOMIAL NAIVE BAYES
# define pipeline: tfidf_vectorizer + Multinomial Naive Bayes
print('Step 2c: defining the pipeline: tfidf_vectorizer + Multinomial NB...')

mnb = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=False, analyzer='char', ngram_range=(1,5), max_df=0.7)),
    ('Multinomial NB', MultinomialNB())
    ])

print('Step 3c: fitting pipeline')
mnb.fit(X_train, y_train)

# predict labels and report metrics
y_pred = mnb.predict(X_test)
cr = classification_report(y_test, y_pred)
print('Step 4c: report metrics (MultinomialNB):\n', cr)
print()
