import json
import collections
from collections import namedtuple

f = open('/Users/KristinDay/Desktop/OneDrive/0Education/0IU_MS_Data_Science/0Spring_2016/Sentiment_Analysis/yelp_data_processed.json')
line = f.readline()
stars = []
pos = []
text = []
review_id = []

i=0
while i<2000:
    line = f.readline()
    review=json.loads(line)
    
    stars.append(review["stars"])
    review_id.append(review["review_id"])
    text.append(review["text"])
    pos.append(review["pos"])
    
    i=i+1
f.close

train_data = text[:1600]
dev_data = text[1600:1800]
test_data = text[1800:2000]

#sort positive reviews from negative by star ratings
pos_data = {}
neg_data = {}
i = 0
while i<1600:
    if stars[i] >=4:
        pos_data[i] = train_data[i]
    else:
        neg_data[i] = train_data[i]
    i = i = 1




        

from collections import defaultdict
def get_space(train_data):
    """
    input is a list of texts
    get a dict of word space
    key=word
    value=len of the dict at that point
    (that will be the index of the word and it is unique since the dict grows as we loop)
    """
    word_space=defaultdict(int)
    for l in train_data:
        for w in l.split():
            #indexes of words won't be in sequential order as they occur
            #in data (can you tell why?),
            #but that doesn't matter
            if w not in word_space:
                word_space[w]=len(word_space)
    return word_space
    
word_space=get_space(train_data)
print(len(word_space))


import numpy as np
def get_sparse_vector(data_point, space):
    #create empty vector
    sparse_vec=np.zeros((len(space)))
    for w in set(data_point.split()):
        #use exception handling such that this funciton can also be used to vectorize
        #data with words not in train (i.e., text and dev data)
        try:
            sparse_vec[space[w]]=1
        except:
            continue
    return sparse_vec

train_vecs=[get_sparse_vector(data_point, word_space) for data_point in train_data]
test_vecs=[get_sparse_vector(data_point, word_space) for data_point in dev_data]
#test_vecs=get_sparse_vector(test_data, word_space)

#We should usually get tags automatically based on input data file.
#Can use stars for that purpose in this dataset.  1-3 stars = neg; 4-5 stars = pos