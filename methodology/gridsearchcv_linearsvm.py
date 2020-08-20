# Grid Search Cross Validation for LinearSVM:
import pandas as pd
import numpy as np
import time
import os
import scipy.stats as stats
import eli5

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import SGDClassifier

# define pipeline: tfidf_vectorizer + random forest classifier
print('Step 1: defining the pipeline: tfidf_vectorizer + random forest classifier...')

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=False, analyzer='char', max_df=0.7)),
    ('svm', SGDClassifier(loss='hinge', penalty='elasticnet', fit_intercept=True, class_weight='balanced', average=False, alpha=1e-5))
    ])

# define grid
print('Step 2: defining the randomized search grid...')

tfidf__ngram_range=[(1,2), (1,3), (1,4), (1,5), (1,6), (1,7)]
svm__l1_ratio = [0.2, 0.4, 0.6, 0.8]


grid_param =   {'tfidf__ngram_range': tfidf__ngram_range,
                'svm__l1_ratio' : svm__l1_ratio
                }

# get data
print('Step 3: getting training and testing data...')
df = pd.read_csv('processed_data_nospace.csv')
X, y = df.phonetic_transcription.fillna(' '), df.language_classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# fit model
if os.path.isfile('gridsearchcv_linearsvm.pkl'):
    print('Step 4: loading model...')
    best_estimator = joblib.load('gridsearchcv_linearsvm.pkl')
else:    
    print('Step 4: fitting search grid...')
    t0 = time.time()
    search = GridSearchCV(pipeline, param_grid=grid_param, verbose=5, cv=5)
    search.fit(X_train, y_train)

    print(f"Fitting took {time.time() - t0:0.3f}s.")
    print()
    print('The best parameters are: ')
    print(search.best_params_)

    # saving the best model
    best_estimator = search.best_estimator_
    joblib.dump(best_estimator, 'gridsearchcv_linearsvm.pkl')

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

report(search.cv_results_, n_top=5)

# predict labels and report metrics
y_pred = best_estimator.predict(X_test)
cr = classification_report(y_test, y_pred)
print(cr)

# checking n most important features
feature_names = best_estimator.steps[0][1].get_feature_names()
eli5.explain_weights(best_estimator.steps[1][1], top=50, feature_names=feature_names, target_names=['NL', 'VL'])