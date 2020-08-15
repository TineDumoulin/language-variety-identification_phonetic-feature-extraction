# Randomized Search Cross Validation for Random Forest:
import pandas as pd
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.metrics import classification_report, f1_score

# define pipeline: tfidf_vectorizer + random forest classifier
print('Step 1: defining the pipeline: tfidf_vectorizer + random forest classifier...')
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=False, analyzer='char')),
    ('rf', RandomForestClassifier(random_state=42, verbose=1, n_jobs=3)),
])

# define grid
print('Step 2: defining the randomized search grid...')

tfidf_ngram_range=[(1,2), (2,3), (2,4), (2,5)]

rf_n_estimators = [800, 1500, 2500]
rf_max_features = ['auto', 'sqrt', 'log2']
rf_max_depth = [10, 20, 30, 40, 50]
rf_max_depth.append(None)
rf_min_samples_split = [2, 5, 10, 15, 20]
rf_min_samples_leaf = [1, 2, 5, 10, 15]

'''# testing
tfidf_ngram_range=[(1,2), (2,3)]

rf_n_estimators = [500, 800]
rf_max_features = ['auto']
rf_max_depth = [10, 20]
rf_max_depth.append(None)
rf_min_samples_split = [2, 5]
rf_min_samples_leaf = [1, 2]'''

grid_param =   {'tfidf__ngram_range': tfidf_ngram_range,
                'rf__n_estimators': rf_n_estimators, 
                'rf__max_features' : rf_max_features, 
                'rf__max_depth': rf_max_depth, 
                'rf__min_samples_split': rf_min_samples_split, 
                'rf__min_samples_leaf': rf_min_samples_leaf
                }

# get data
print('Step 3: getting training and testing data...')
df = pd.read_csv('processed_data_nospace.csv')
X, y = df.phonetic_transcription.fillna(' '), df.language_classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# fit model
if os.path.isfile('randomsearchcv_randomforest.pkl'):
    print('Step 4: loading model...')
    search = joblib.load('randomsearchcv_randomforest.pkl')
else:    
    print('Step 4: fitting randomized search grid...')
    t0 = time.time()
    search = RandomizedSearchCV(pipeline, param_distributions=grid_param, verbose=5)
    search.fit(X_train, y_train)

    print(f"Fitting took {time.time() - t0:0.3f}s.")
    print()
    print('The best parameters are: ')
    print(search.best_params_)

    # saving the best model
    joblib.dump(search.best_estimator_, 'randomsearchcv_randomforest.pkl')

# predict and report metrics
y_pred = search.predict(X_test)

print('F1-score: ', f1_score(y_test, y_pred, pos_label='pos'))

cr = classification_report(y_test, y_pred)
print(cr)