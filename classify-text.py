from skafossdk import *
import pickle
import logging
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import argparse

from common.modeling import TARGET_LIST

ska = Skafos()

# Read in data to score from the command line
parser = argparse.ArgumentParser("Text classification")
parser.add_argument(
  '--text',
  type=str,
  help='Text to classify.')
args = parser.parse_args()

if __name__ == '__main__':

    # Make sure text argument is a string
    if isinstance(args.text, str) == False:
        ska.log("Must pass a string argument")
        sys.exit()

    # Read in model and vectorizer from what we saved in the training run
    pickledModel = ska.engine.load_model("Classifier", tag="latest").result()
    model_id = int(pickledModel['meta']['version'])
    ska.log(f"Classifier model id from unpickling: {model_id}", labels=["classifier id"])
    clf = pickle.loads(pickledModel['data'])

    pickledVectorizer = ska.engine.load_model("Vectorizer", tag="latest").result()
    vectorizer_id = int(pickledVectorizer['meta']['version'])
    ska.log(f"Vectorizer model id from unpickling: {vectorizer_id}", labels=["vectorizer id"])
    vectorizer = pickle.loads(pickledVectorizer['data'])

    # Classify text and log output
    example_text = [args.text]
    ex_tfidf = vectorizer.transform(example_text)
    ex_predict = clf.predict(ex_tfidf)
    group = TARGET_LIST[ex_predict[0]]
    ska.log(f"Text is in group: {group}")
