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

#
# again define a special tokenizer.
# this time we want to experiment with documents of different lengths
#
def my_tokenizer(s, doclen=None):
    tokens = tokenizer_re.findall(s)
    if doclen is not None and len(tokens) > doclen:
        return tokens[:doclen]
    else:
        return tokens

for n in (10, 20, 30, 50, 100):
    v = TfidfVectorizer(ngram_range=(1,2), min_df=2,
            tokenizer=lambda s: my_tokenizer(s, doclen=n))

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
        m = LogisticRegression(class_weight='balanced') # this gives more balanced p/r
        #m = LogisticRegression()
        m.fit(x_train, y_train)
        y_pred = m.predict(x_test)
        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        recall.append(r)
        precision.append(p)
        fscore.append(f)
        accuracy.append(accuracy_score(y_test, y_pred))

    print("Length: {} Precision: {:.4f}, Recall: {:.4f}, F-score: {:.4f}, Accuracy: {:.4f}".format(
                n,
                np.array(precision).mean(),
                np.array(recall).mean(),
                np.array(fscore).mean(),
                np.array(accuracy).mean())
    )

