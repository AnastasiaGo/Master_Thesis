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
import numpy as np
import pandas as pd
import scipy

import sys, time, glob, re

dirpath1='depression/'
dirpath2='non_depression/'


sent_d = pd.read_table('sentiment-analysis.txt')

tokenizer_re = re.compile("\w+|\S")

labels = []     # 1 for depression, 0 for non-depression
documents = []
sentiments = np.empty((0,10))
for filename in glob.glob(dirpath1 + '*.txt'):
   with open(filename, "r", encoding="latin-1") as fp:
       doc = fp.read().strip()
       labels.append(1)
       documents.append(doc)
       xmlfile = filename + '.xml'
       sdata = sent_d.loc[sent_d['file'] == xmlfile][['tok_vneg', 'tok_neg', 'tok_neut', 'tok_pos', 'tok_vpos',
                                            'sent_vneg', 'sent_neg', 'sent_neut', 'sent_pos', 'sent_vpos']]
       if len(sdata) == 0:
           sdata = np.zeros((1,10))
       sentiments = np.vstack((sentiments,  sdata))

for filename in glob.glob(dirpath2 + '*.txt'):
   with open(filename, "r", encoding="latin-1") as fp:
       doc = fp.read().strip()
       labels.append(0)
       documents.append(doc)

       xmlfile = filename + '.xml'
       sdata = sent_d.loc[sent_d['file'] == xmlfile][['tok_vneg', 'tok_neg', 'tok_neut', 'tok_pos', 'tok_vpos',
                                            'sent_vneg', 'sent_neg', 'sent_neut', 'sent_pos', 'sent_vpos']]
       if len(sdata) == 0:
           sdata = np.zeros((1,10))
       sentiments = np.vstack((sentiments,  sdata))

labels = np.array(labels)


v = TfidfVectorizer(ngram_range=(1,2), min_df=2,
        tokenizer=lambda s: tokenizer_re.findall(s))


v.fit(documents)
x = v.transform(documents)

print(x.shape, sentiments.shape)
x = scipy.sparse.hstack((x, sentiments)).tocsr()

print(x.shape)

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
    m = LogisticRegression()
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

