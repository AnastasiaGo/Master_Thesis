#!/usr/bin/python3 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC


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

# unigrams only
#v = CountVectorizer(ngram_range=(1,2))

# use tf-idf instead (this should always be better)
#v = TfidfVectorizer(ngram_range=(1,1))

# also include bigrams
# v = TfidfVectorizer(ngram_range=(1,2))

# do not lowercase - case difference may be relevant to `mood'
#v = TfidfVectorizer(ngram_range=(1,2), lowercase=False)

# this replaces the built-in tokenizer of the vectorizer.
# prevents removing short strings, punctuation (exclamation, question mark
# etc. may be helpful)
v = TfidfVectorizer(ngram_range=(1,2),
        tokenizer=lambda s: tokenizer_re.findall(s))

v.fit(documents)
x = v.transform(documents)


# we do cross-validation no need for extra split
#x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2)

# this may work better if one of the classes has more instances
#m = LogisticRegression(class_weight='balanced')

m = LogisticRegression()

# you can try SVM instead of logistic regression - in general SVMs tend to perform better
#m = LinearSVC(dual=False)

# cross_val_score will fit the model, no need to fit here.
#m.fit(x_train, y_train)
#m_accuracy = cross_validation.cross_val_score(m, x_test, y_test, cv=10, scoring='accuracy')

m_accuracy = cross_validation.cross_val_score(m, x, labels, cv=10, scoring='accuracy')

print (m_accuracy.mean())

# disabled 'coefficient analysis' stuff for now, feel free to enable & analyze
# with some of the chosen models.
#
#d = dict(zip(v.vocabulary_, m.coef_[0]))
#dictionary=sorted([(k,v) for (k,v) in d.items()], key=lambda x: x[1])
  

#with open ("coefficients.csv", "w", encoding="latin-1") as fp:
#   for k,v in dictionary:
#      print ("{}\t{}".format(k,v), file=fp)
