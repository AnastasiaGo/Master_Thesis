#!/usr/bin/python3
import os
from lxml import etree
import glob
import sys

#for depression files change the name of the folder
os.chdir("/Users/anastasia/Documents/Tuebingen/Stylometry/Master_Thesis/blogs_out/blogs_nondepression_out/")
print("file\tneg\tpos\tdepression")
for f in glob.glob('./*.out'):
     scount = {'Negative': 0,
             'Neutral': 0,
             'Positive': 0,
               'Verynegative': 0,
               'Verypositive': 0
               }
     with open(f, 'r', encoding='utf-8') as fp:
          xml = etree.parse(fp)
          for sent in xml.findall("//sentence"):
               sentiment = sent.get("sentiment")
               if sentiment is not None:
                    scount[sentiment] += 1
                    if sentiment == "Verynegative":
                        scount["Negative"]+= 1
                    if sentiment == "Verypositive":
                        scount["Positive"]+= 1
          scount={'Negative': scount ['Negative'],
                  'Positive' : scount ['Positive']}  
               
     print ("{}".format(f), end="")
     for k in sorted(scount):
          print("\t{}".format(scount[k]), end="")
     if scount['Negative'] > scount ['Positive']:
          print("\tTrue")
     elif scount['Negative'] == scount['Positive']:
          print("\tNeutral")
     else:
          print("\tFalse")
