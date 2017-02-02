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



import sys, time, glob, re

dirpath1='depression/'
dirpath2='non_depression/'

tokenizer_re = re.compile("\w+|\S")

labels = []     # 1 for depression, 0 for non-depression
documents = []
for filename in glob.glob(dirpath1 + '*.txt'):
   with open(filename, "r", encoding="latin-1") as fp:
       doc = fp.read().strip()
       labels.append(1)
       documents.append(doc)

for filename in glob.glob(dirpath2 + '*.txt'):
   with open(filename, "r", encoding="latin-1") as fp:
       doc = fp.read().strip()
       labels.append(0)
       documents.append(doc)

labels = np.array(labels)


v = TfidfVectorizer(ngram_range=(1,2), min_df=2,
        tokenizer=lambda s: tokenizer_re.findall(s))


v.fit(documents)
x = v.transform(documents)

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

