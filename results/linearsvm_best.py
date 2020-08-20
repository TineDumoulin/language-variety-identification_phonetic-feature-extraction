# Best SVM after grid search: {'svm__l1_ratio': 0.2, 'tfidf__ngram_range': (1, 5)}

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import eli5

# pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=False, analyzer='char', max_df=0.7, ngram_range=(1,5))),
    ('svm', SGDClassifier(loss='hinge', penalty='elasticnet', fit_intercept=True, class_weight='balanced', average=False, alpha=1e-5, l1_ratio=0.2))
    ])

# data
df = pd.read_csv('processed_data_nospace.csv')
X, y = df.phonetic_transcription.fillna(' '), df.language_classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#fitting
model = pipeline.fit(X_train, y_train)

# predicting & evaluating
y_pred = model.predict(X_test)
cr = classification_report(y_test, y_pred)
print(cr)

# 50 most important features
feature_names = model.steps[0][1].get_feature_names()
eli5.explain_weights(model.steps[1][1], top=50, feature_names=feature_names, target_names=['NL', 'VL'])
