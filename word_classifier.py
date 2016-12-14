#!/usr/bin/python3
import os
from lxml import etree
import glob
import sys      

#for depression files change the name of the folder
os.chdir("/Users/anastasia/Documents/Tuebingen/Stylometry/Master_Thesis/blogs_out/blogs_nondepression_out/")
print("file\tnegative\tpositive\tdepression")
for f in glob.glob('./*.out'):
     scount = {'Negative': 0,
               'Neutral': 0,
               'Positive': 0,
               'Very negative':0,
               'Very positive':0
                    }
          
     with open(f, 'r', encoding='utf-8') as fp:
          xml= etree.parse(fp)
          for x in xml.findall("//token/sentiment"):
              sentiment = x.text
              if sentiment is not None:
                   scount[sentiment] += 1
                   if sentiment == "Very negative":
                        scount["Negative"]+= 1
                   if sentiment == "Very positive":
                        scount["Positive"]+= 1
          scount={'Negative': scount ['Negative'],
                  'Positive' : scount ['Positive']}     
                   
              #else:
#                 scount['Unknown'] += 1
                    
     print ("{}".format(f), end="")
     for k in sorted(scount):
          print("\t{}".format(scount[k]), end="")
     if scount['Positive'] > scount['Negative']:     
          print("\tFalse")
     elif scount['Negative'] == scount['Positive']:
          print("\tNeutral")     
     else:
          print("\tTrue")
