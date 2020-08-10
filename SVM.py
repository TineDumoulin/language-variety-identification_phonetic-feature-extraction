import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count_vect = CountVectorizer(lowercase='false', analyzer='char', ngram_range=(1,4))
tfidf_transformer = TfidfTransformer()

df = pd.read_csv('data.csv')
X, y = df.phonetic_transcription.fillna(' '), df.language_classification
print('X shape: ', X.shape)
print('y shape: ', y.shape)
print()

X_train_counts = count_vect.fit_transform(list(X))
print('X train counts: ', X_train_counts.shape)

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print('X train tfidf: ', X_train_tfidf.shape)
print(X_train_tfidf[:10])