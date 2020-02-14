#!/usr/bin/env python3
"""
ENLP A2: Perceptron

Usage: python perceptron.py NITERATIONS

@author: Alan Ritter, Nathan Schneider

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""
import sys, os, glob
from collections import Counter
from math import log
from numpy import mean
import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from evaluation import Eval
import csv
# nltk.download('averaged_perceptron_tagger')


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def load_docs(direc, lemmatize, normalize, bigram, labelMapFile='labels.csv'):
    """Return a list of word-token-lists, one per document.
    Words are optionally lemmatized with WordNet."""

    labelMap = {}   # docID => gold label, loaded from mapping file
    with open(os.path.join(direc, labelMapFile)) as inF:
        for ln in inF:
            docid, label = ln.strip().split(',')
            assert docid not in labelMap
            labelMap[docid] = label

    # create parallel lists of documents and labels
    docs, labels = [], []
    for file_path in sorted(glob.glob(os.path.join(direc, '*.txt'))):
        filename = os.path.basename(file_path)
        # open the file at file_path, construct a list of its word tokens,
        # and append that list to 'docs'.
        # look up the document's label and append it to 'labels'.
        with open(file_path) as f:
            l = word_tokenize(f.read())
            # add lemmatization feature
            if lemmatize:
                res = []
                for word, pos in pos_tag(l):
                    wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
                    res.append(WordNetLemmatizer().lemmatize(word=word, pos=wordnet_pos))
                l = res
            #  add normalize feature
            if normalize:
                l = [x.lower() for x in l]
            # add bigram feature
            if bigram:
                prev = "<s>"
                length = len(l)
                for i in range(length):
                    l.append(prev + "|" + l[i])
                    prev = l[i]
                l.append(l[length - 1] + "|" + "</s>")
            docs.append(l)
            labels.append(labelMap[filename])

    return docs, labels


def extract_feats(doc):
    """
    Extract input features (percepts) for a given document.
    Each percept is a pairing of a name and a boolean, integer, or float value.
    A document's percepts are the same regardless of the label considered.
    """
    ff = Counter(doc)
    return ff


def load_featurized_docs(datasplit, i):
    if i == 0:
        rawdocs, labels = load_docs(datasplit, lemmatize=False, normalize=False, bigram=False)
    elif i == 1:
        rawdocs, labels = load_docs(datasplit, lemmatize=True, normalize=True, bigram=True)
    elif i == 2:
        rawdocs, labels = load_docs(datasplit, lemmatize=False, normalize=True, bigram=True)
    elif i == 3:
        rawdocs, labels = load_docs(datasplit, lemmatize=True, normalize=False, bigram=True)
    else:
        rawdocs, labels = load_docs(datasplit, lemmatize=True, normalize=True, bigram=False)

    assert len(rawdocs)==len(labels)>0,datasplit
    # print(Counter(labels))
    featdocs = []
    for d in rawdocs:
        featdocs.append(extract_feats(d))
    return featdocs, labels


class Perceptron:
    def __init__(self, train_docs, train_labels, MAX_ITERATIONS=100, dev_docs=None, dev_labels=None):
        self.CLASSES = ['ARA', 'DEU', 'FRA', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR', 'ZHO']
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.dev_docs = dev_docs
        self.dev_labels = dev_labels
        self.weights = {l: Counter() for l in self.CLASSES}
        self.learn(train_docs, train_labels)

    def copy_weights(self):
        """
        Returns a copy of self.weights.
        """
        return {l: Counter(c) for l,c in self.weights.items()}

    def learn(self, train_docs, train_labels):
        """
        Train on the provided data with the perceptron algorithm.
        Up to self.MAX_ITERATIONS of learning.
        At the end of training, self.weights should contain the final model
        parameters.
        """
        lr = 1
        max_acc = 0
        max_weight = self.copy_weights()
        for i in range(self.MAX_ITERATIONS):
            for t in range(len(train_labels)):
                for c in self.CLASSES:
                    y_hat = 0
                    # calculate total loss
                    for token, freq in train_docs[t].items():
                        if token not in self.weights[c]:
                            self.weights[c][token] = 0
                        y_hat += 1 * self.weights[c][token]
                    # add bias term
                    if "bias_term" not in self.weights[c]:
                        self.weights[c]["bias_term"] = 0
                    y_hat += 1 * self.weights[c]["bias_term"]
                    # update weight if y_hat * y <= 0
                    y = 1 if train_labels[t] == c else -1
                    if y_hat * y <= 0:
                        for token, freq in train_docs[t].items():
                            self.weights[c][token] += lr * 1 * y
                        self.weights[c]["bias_term"] += lr * y

            train_acc = self.test_eval(train_docs, train_labels)
            dev_acc = self.test_eval(self.dev_docs, self.dev_labels)
            print("iteration: {}, trainAcc: {}, devAcc: {}".format(i, train_acc, dev_acc), file=sys.stderr)
            if dev_acc > max_acc:
                max_acc = dev_acc
                max_weight = self.copy_weights()
            if train_acc == 1 or i == 30:
                break
        self.weights = max_weight

    def score(self, doc, label):
        """
        Returns the current model's score of labeling the given document
        with the given label.
        """
        value = 0
        for token, freq in doc.items():
            value += 1 * self.weights[label][token]
        value += self.weights[label]["bias_term"]
        return value

    def predict(self, doc):
        """
        Return the highest-scoring label for the document under the current model.
        """
        scores = []
        for c in self.CLASSES:
            scores.append(self.score(doc, c))
        return self.CLASSES[scores.index(max(scores))]

    def confusion_matrix(self, pred_labels, test_labels):
        """
        Store confusion matrix to a file.
        """
        length = len(self.CLASSES)
        matrix = [[0 for i in range(length)] for j in range(length)]
        for pred, real in zip(pred_labels, test_labels):
            matrix[self.CLASSES.index(real)][self.CLASSES.index(pred)] += 1
        with open('confusion matrix.csv', "w") as wf:
            csv_writer = csv.writer(wf, delimiter=",")
            csv_writer.writerows(matrix)
        return

    def top_features(self):
        """
        Store bias and top 10 most common and least common features with their weights to a file.
        """
        with open('top features.txt', 'w') as wf:
            for c in self.CLASSES:
                wf.write("Language: {}\n".format(c))
                wf.write("bias: ")
                wf.write(str(self.weights[c]['bias_term'])+"\n")
                wf.write("10 most common features and their weights:\n")
                wf.write(str(Counter(self.weights[c]).most_common(10)))
                wf.write("\n10 least common features and their weights:\n")
                wf.write(str(Counter(self.weights[c]).most_common()[:-11:-1]))
                wf.write("\n")

        return

    def test_eval(self, test_docs, test_labels, toFile=False):
        """
        Calculate Test Accuracy.
        """
        pred_labels = [self.predict(d) for d in test_docs]
        ev = Eval(test_labels, pred_labels)
        if toFile:
            self.confusion_matrix(pred_labels, test_labels)
            self.top_features()
        return ev.accuracy()


if __name__ == "__main__":
    args = sys.argv[1:]
    niters = int(args[0])

    features = ["Base line features", "All features, including unigram, bigram, normalization and lemmatization",
                "All features except for lemmatization", "All features except for normalization",
                "All features except for bigram"]
    for i in range(0, 5):
        print("Selected features: {}".format(features[i]), file=sys.stderr)
        train_docs, train_labels = load_featurized_docs('train', i)
        print(len(train_docs), 'training docs with',
            sum(len(d) for d in train_docs)/len(train_docs), 'percepts on avg', file=sys.stderr)

        dev_docs,  dev_labels  = load_featurized_docs('dev', i)
        print(len(dev_docs), 'dev docs with',
            sum(len(d) for d in dev_docs)/len(dev_docs), 'percepts on avg', file=sys.stderr)

        test_docs,  test_labels  = load_featurized_docs('test', i)
        print(len(test_docs), 'test docs with',
            sum(len(d) for d in test_docs)/len(test_docs), 'percepts on avg', file=sys.stderr)

        ptron = Perceptron(train_docs, train_labels, MAX_ITERATIONS=niters, dev_docs=dev_docs, dev_labels=dev_labels)
        acc = ptron.test_eval(test_docs, test_labels)
        print(acc, file=sys.stderr)

    # Error Analysis with best model
    print("Model with best performance is the one with {}".format(features[3]), file=sys.stderr)
    train_docs, train_labels = load_featurized_docs('train', 3)
    dev_docs, dev_labels = load_featurized_docs('dev', 3)
    test_docs, test_labels = load_featurized_docs('test', 3)
    ptron = Perceptron(train_docs, train_labels, MAX_ITERATIONS=niters, dev_docs=dev_docs, dev_labels=dev_labels)
    acc = ptron.test_eval(test_docs, test_labels, toFile=True)
