
# coding: utf-8

# In[16]:


import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import movie_reviews

import sklearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem.snowball import SnowballStemmer

dir = "/Users/anujkapur/Downloads/review_polarity/txt_sentoken"
movie_train = load_files('/Users/anujkapur/Downloads/review_polarity/txt_sentoken', shuffle=True)


# In[67]:


from nltk.stem.snowball import FrenchStemmer

normCountMatrix = TfidfVectorizer().fit_transform(movie_train['data'])


# In[68]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
docs_train, docs_test, y_train, y_test = train_test_split(
    normCountMatrix, movie_train.target, test_size = 0.20, random_state = 12)

model = MultinomialNB().fit(docs_train, y_train)

y_pred = model.predict(docs_test)
sklearn.metrics.accuracy_score(y_test, y_pred)

