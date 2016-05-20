import json
import pandas as pd
import argparse
import codecs
import time
import sys
import os, re, glob
import nltk
from collections import defaultdict
from random import shuffle, randint
import numpy as np
from numpy import array, arange, zeros, hstack, argsort
import unicodedata
from scipy.sparse import csr_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import gensim

f = open('/Users/KristinDay/OneDrive/0Education/0IU_MS_Data_Science/0Spring_2016/Sentiment_Analysis/yelp_data_processed.json')
line = f.readline()
stars = {}
pos = {}
text = {}
review_id = {}

i=0
while i<1000:
    #print line
    line = f.readline()
    review=json.loads(line)
    
    stars[i] = review["stars"]
    review_id[i] = review["review_id"]
    text[i] = review["text"]
    pos[i] = review["pos"]
    
   
    i=i+1
f.close

yelp_df = pd.DataFrame(columns=['stars', 'review_ID', 'text', 'pos'])
pos_data = pd.DataFrame(columns=['stars', 'review_ID', 'text', 'pos'])
neg_data = pd.DataFrame(columns=['stars', 'review_ID', 'text', 'pos'])

for i in stars:
    yelp_df.loc[len(yelp_df)] = [stars[i], review_id[i], text[i], pos[i]]
    
    
#split into pos and neg tweets
i = 0
for index, l in yelp_df.iterrows():
    x = yelp_df.iloc[i]['stars'] 
    if x > 3:
        pos_data.loc[len(pos_data)] = [l['stars'], l['review_ID'],  l['text'], l['pos']]
    else:
        neg_data.loc[len(neg_data)] = [l['stars'], l['review_ID'],  l['text'], l['pos']]
    i += 1

frames1 = [pos_data[:200], neg_data[:200]]
frames2 = [pos_data[200:400], neg_data[200:400]]

train_data = pd.concat(frames1, ignore_index=True)
test_data = pd.concat(frames2, ignore_index=True)


def get_dict(train_data):
    new_dict = defaultdict(int)
    for index, l in train_data.iterrows():
        for w in l['text'].split():
            if w not in new_dict:
                new_dict[w] = len(new_dict)
    return new_dict

yelp_dict = get_dict(train_data)



def get_sparse_vec(data_point, space):
    sparse_vec = np.zeros((len(space)))
    for w in set(data_point.split()):
        try:
            sparse_vec[space[w]] = 1
        except:
            continue
    return sparse_vec


train_vecs = [get_sparse_vec(l['text'], yelp_dict) for index, l in train_data.iterrows()]
    
test_vecs = [get_sparse_vec(l['text'], yelp_dict) for index, l in test_data.iterrows()]


train_tags = [1.0 for i in range(200)] + [0.0 for i in range(200)]
test_tags = [1.0 for i in range(200)] + [0.0 for i in range(200)]

train_vecs = np.array(train_vecs)
train_tags = np.array(train_tags)

n_jobs = 2

clf = OneVsRestClassifier(SVC(C=1, kernel = 'linear', gamma=1, verbose= False, probability=False))
clf.fit(train_vecs, train_tags)
print "\nDone fitting classifier on training data...\n"

#------------------------------------------------------------------------------------------
print "="*50, "\n"
print "Results with 5-fold cross validation:\n"
print "="*50, "\n"
#------------------------------------------------------------------------------------------
predicted = cross_validation.cross_val_predict(clf, train_vecs, train_tags, cv=5)
print "*"*20
print "\t accuracy_score\t", metrics.accuracy_score(train_tags, predicted)
print "*"*20
print "precision_score\t", metrics.precision_score(train_tags, predicted)
print "recall_score\t", metrics.recall_score(train_tags, predicted)
print "\nclassification_report:\n\n", metrics.classification_report(train_tags, predicted)
print "\nconfusion_matrix:\n\n", metrics.confusion_matrix(train_tags, predicted)
                                           