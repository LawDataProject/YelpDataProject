#Yelp data project
#Kristin Day February 27, 2016
#Sentiment Analysis Z604

import collections
from collections import namedtuple

all_data = []
DataDoc = namedtuple('DataDoc', 'tag words')
with open ('/Users/KristinDay/OneDrive/0Education/0IU_MS_Data_Science/0Spring_2016/Sentiment_Analysis/Yelp_Challenge/yelp_academic_dataset_business.json') as alldata:
    for line_no, line in enumerate(alldata):
        label=line.split()[0]
        word_list=line.lower().split()[1:]
        all_data.append(DataDoc(label, word_list))
        #print my_data[line_no]
        #break
train_data_all = all_data[:40000]
test_data_all = all_data[40001:]

train_data=train_data_all[:100]+train_data_all[20000:20100]
test_data=test_data_all[:100]+test_data_all[17000:17100]



#Get a dictionary of all the words in training_data
#These will be our bag-of-words features
#We won't need this function, since we will use gensim's built-in-method "Dictionary"
#from the corpus module
#--> corpora.Dictionary, but we provide this so that you are clear on one way of how to do this.

from collections import defaultdict
def get_space(train_data):
    """
    input is a list of namedtuples
    get a dict of word space
    key=word
    value=len of the dict at that point
    (that will be the index of the word and it is unique since the dict grows as we loop)
    """
    word_space=defaultdict(int)
    for doc in train_data:
        for w in doc.words:
            #indexes of words won't be in sequential order as they occur
            #in data (can you tell why?),
            #but that doesn't matter
            if w not in word_space:
                word_space[w]=len(word_space)
    return word_space
    
word_space=get_space(train_data)
print(len(word_space))
print word_space["love"]

import numpy as np
def get_sparse_vector(data_point, space):
    #create empty vector
    sparse_vec=np.zeros((len(space)))
    for w in set(data_point.words):
        #use exception handling such that this funciton can also be used to vectorize
        #data with words not in train (i.e., text and dev data)
        try:
            sparse_vec[space[w]]=1
        except:
            continue
    return sparse_vec

train_vecs=[get_sparse_vector(data_point, word_space) for data_point in train_data]
test_vecs=[get_sparse_vector(data_point, word_space) for data_point in test_data]
#test_vecs=get_sparse_vector(test_data, word_space)

#print train_vecs, test_vecs[0]
print(train_data[12500:12600])
print(train_vecs[:25])
print(test_vecs[:25])

#We should usually get tags automatically based on input data file.
#Can use stars for that purpose in this dataset.  1-3 stars = neg; 4-5 stars = pos
