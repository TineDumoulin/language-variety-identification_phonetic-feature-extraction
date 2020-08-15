import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt  
import seaborn as sns


count_vect = CountVectorizer(lowercase='false', analyzer='char', ngram_range=(1,4))
tfidf_transformer = TfidfTransformer()

df = pd.read_csv('processed_data.csv')
X, y = df.phonetic_transcription.fillna(' '), df.language_classification
print('X shape: ', X.shape)
print('y shape: ', y.shape)
print()

# IMPORTANT: Make a train/test split here (before fitting the vectorizers on the data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Define a baseline
'''A baseline could be an accuracy of 50% (since by randomly guessing the right classification, 
we should reach an accuracy of 50% statistically speaking). 
Zero-R baseline: sklearn dummyclassifier with strategy "most_frequent"'''

baseline = {'most_frequent': 0.672540932383452, 'stratified': 0.560867391576053, 'uniform': 0.5012498437695289, 'constant': 0.672540932383452}

if baseline:
    print(baseline)
else:
    strategies = ['most_frequent', 'stratified', 'uniform', 'constant'] 
    
    test_scores = [] 
    for s in strategies: 
        if s =='constant': 
            dclf = DummyClassifier(strategy = s, random_state = 0, constant ='NL') 
        else: 
            dclf = DummyClassifier(strategy = s, random_state = 0) 
        dclf.fit(X_train, y_train) 
        score = dclf.score(X_test, y_test) 
        test_scores.append(score) 

    ax = sns.stripplot(strategies, test_scores); 
    ax.set(xlabel ='Strategy', ylabel ='Test Score') 
    plt.show()