#!/usr/bin/python3 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np



import sys, time, glob, re


d = pd.read_table('sentiment-analysis.txt')

labels = d['class'] == 'depression'
x = d[['tok_vneg', 'tok_neg', 'tok_neut', 'tok_pos', 'tok_vpos',
       'sent_vneg', 'sent_neg', 'sent_neut', 'sent_pos', 'sent_vpos']]
x = x.as_matrix()

kf = StratifiedKFold(labels, n_folds=10)

precision = []
recall = []
fscore = []
accuracy = []
for trn_i, tst_i in kf:
    trn_i = np.array(trn_i)
    tst_i = np.array(tst_i)
    x_train, x_test = x[trn_i], x[tst_i]
    y_train, y_test = labels[trn_i], labels[tst_i]
    #m = LogisticRegression(class_weight='balanced') # this gives more balanced p/r
    #m = LogisticRegression()
    m = LinearSVC(class_weight='balanced')
    m.fit(x_train, y_train)
    y_pred = m.predict(x_test)
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    recall.append(r)
    precision.append(p)
    fscore.append(f)
    accuracy.append(accuracy_score(y_test, y_pred))

print("Precision: {}, Recall: {}, F-score: {}, Accuracy: {}".format(
            np.array(precision).mean(),
            np.array(recall).mean(),
            np.array(fscore).mean(),
            np.array(accuracy).mean())
)
