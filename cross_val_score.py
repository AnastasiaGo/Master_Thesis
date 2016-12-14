from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

import sys, time, glob

dirpath1='./depression_copy/'
dirpath2='./nondepression_copy/'

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

v = CountVectorizer(ngram_range=(1,1))
v.fit(documents)
x = v.transform(documents)


x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2)

m = LogisticRegression()
m.fit(x_train, y_train)
m_accuracy = cross_validation.cross_val_score(m, x_test, y_test, cv=10, scoring='accuracy')

print (m_accuracy.mean())



d = dict(zip(v.vocabulary_, m.coef_[0]))
dictionary=sorted([(k,v) for (k,v) in d.items()], key=lambda x: x[1])
  

with open ("coefficients.csv", "w", encoding="latin-1") as fp:
   for k,v in dictionary:
      print ("{}\t{}".format(k,v), file=fp)
