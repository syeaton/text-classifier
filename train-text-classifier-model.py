# This is an example text classification model in Skafos, based in part on the following examples:
# https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
# https://github.com/javedsha/text-classification
# https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html

from skafossdk import *
import pickle
import logging
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

ska = Skafos()

# Select training and testing data. This creates newsgroups_train and newsgroups_test as sklearn.utils.Bunch objects
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle='True')
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), shuffle='True')

# Use training data and scikit-learn feature extraction functions to create feature vectors from the text data
# Want TF-IDF weightings
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(newsgroups_train.data)

# Use a naive bayes classifier to predict targets
clf = MultinomialNB(alpha=0.05).fit(X_train_tfidf, newsgroups_train.target)
X_test_tfidf = vectorizer.transform(newsgroups_test.data)
news_pred = clf.predict(X_test_tfidf)
accuracy = metrics.accuracy_score(newsgroups_test.target, news_pred)
# Print training accuracy here

# Save model to Skafos using save_model. Note that we need to save both the model and the vectorizer
pickledVectorizer = pickle.dumps(vectorizer)
saved_vectorizer = ska.engine.save_model("Vectorizer", pickledVectorizer, tags=["Vectorizer", "latest"])
ska.log(f"Tfidf vectorizer saved to Cassandra: {saved_vectorizer} \n", labels=["model saving"], level=logging.INFO)

pickledModel = pickle.dumps(clf)
saved_model = ska.engine.save_model("Classifier", pickledModel, tags=["Classifier", "latest"])
ska.log(f"Text classifier saved to Cassandra: {saved_model} \n", labels=["model saving"], level=logging.INFO)
